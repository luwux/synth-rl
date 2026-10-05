"""GRPO with LoRA for Qwen Omni thinkers on synth-rl tasks.

Each step samples fresh procedural tasks, draws `group` completions per task at temperature 1,
scores every completion by applying its edits in Vital and rendering, and takes one policy-gradient
step with group-normalized advantages. Sampling and the update use the same weights (one update per
batch), so the PPO ratio is 1 and the clipped objective reduces to advantage-weighted log-likelihood.

Single process on CUDA, MPS, or CPU; multi-GPU with `torchrun --nproc_per_node N -m synth_rl.grpo ...`
(data parallel: every rank holds the model, LoRA gradients are averaged with all-reduce).
The run is resumable: rerunning the same command continues from `<out>/ckpt/latest`.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import multiprocessing as mp
import os
import random
import shutil
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist

from .agents import hide_values, parse_edits, prompt_text
from .omni import OmniPolicy, to_model_audio
from .procedural import random_patch
from .rl import FORMAT_FAIL
from .tasks import EDIT_SPECS, MONO_SPECS, Task, make_edit_task, make_match_task, make_repair_task, score
from .vital_env import EDITABLE, VitalEnv

LORA_TARGETS = r".*model\.layers\.\d+\.(self_attn\.(q|k|v|o)_proj|mlp\.(gate|up|down)_proj)"

# ---------------------------------------------------------------- tasks and rewards


def sample_tasks(env: VitalEnv, rng: random.Random, n: int, kinds: list[str], specs: list[str]) -> list[Task]:
    """Fresh procedural tasks, cycling through kinds from a random offset so every batch is balanced."""
    tasks = []
    offset = rng.randrange(len(kinds))
    while len(tasks) < n:
        state = random_patch(env, rng)
        kind = kinds[(offset + len(tasks)) % len(kinds)]
        if kind == "edit":
            task = make_edit_task(env, state, rng, specs)
        elif kind == "repair":
            task = make_repair_task(env, state, rng)
        else:
            task = make_match_task(env, state)
        if task:
            tasks.append(task)
    return tasks


KINDS = ("edit", "repair", "match")

_worker_env: VitalEnv | None = None


def _init_worker() -> None:
    global _worker_env
    torch.set_num_threads(1)
    _worker_env = VitalEnv()


def _score(item: tuple[dict, str]) -> dict:
    task_dict, text = item
    edits = {k: v for k, v in parse_edits(text).items() if k in EDITABLE}
    if not edits:
        return {"reward": FORMAT_FAIL, "valid": False, "n_changed": 0}
    out = score(_worker_env, Task(**task_dict), edits)
    return {"reward": float(out["reward"]), "valid": True, "n_changed": int(out["n_changed"])}


def task_dict(t: Task) -> dict:
    return {"kind": t.kind, "state": t.state, "instruction": t.instruction, "spec_id": t.spec_id,
            "target_state": t.target_state, "perturbed": t.perturbed}


# ---------------------------------------------------------------- prompts and log-probs


class Prompts:
    """Chat texts and model-rate audio clips for a list of tasks; `mute` replaces every clip with silence."""

    def __init__(self, env: VitalEnv, policy: OmniPolicy, tasks: list[Task], mute: bool = False, blind: bool = False):
        self.texts, self.clips = [], []
        for t in tasks:
            env.load_state(t.state)
            clips = [env.render(t.state)] + ([env.render(t.target_state)] if t.target_state else [])
            clips = [to_model_audio(c) for c in clips]
            if mute:
                clips = [np.zeros_like(c) for c in clips]
            params = hide_values(env.params()) if blind else env.params()
            self.texts.append(policy.chat(prompt_text(params, t.instruction), len(clips)))
            self.clips.append(clips)

    def select(self, idx: list[int]) -> tuple[list[str], list[np.ndarray]]:
        return [self.texts[i] for i in idx], [c for i in idx for c in self.clips[i]]


def completion_mask(ids: torch.Tensor, stop_ids: list[int]) -> torch.Tensor:
    """1 for generated tokens up to and including the first stop token, 0 for the padding after it."""
    is_stop = torch.zeros_like(ids, dtype=torch.bool)
    for s in stop_ids:
        is_stop |= ids == s
    ended_before = (torch.cumsum(is_stop.int(), dim=1) - is_stop.int()) > 0
    return (~ended_before).float()


def token_logprobs(policy: OmniPolicy, texts: list[str], clips: list[np.ndarray], comp: torch.Tensor,
                   mask: torch.Tensor) -> torch.Tensor:
    """Log-probability of each completion token given its prompt, shape (n, completion length)."""
    inp = policy.encode(texts, clips)
    inp["input_ids"] = torch.cat([inp["input_ids"], comp], dim=1)
    inp["attention_mask"] = torch.cat([inp["attention_mask"], mask.to(inp["attention_mask"].dtype)], dim=1)
    with policy.last_logits(comp.shape[1] + 1):
        logits = policy.model(**inp).logits[:, :-1].float()
    return torch.log_softmax(logits, dim=-1).gather(-1, comp.unsqueeze(-1)).squeeze(-1)


# ---------------------------------------------------------------- distributed helpers


def setup_dist() -> tuple[int, int, str | None]:
    if "LOCAL_RANK" not in os.environ:
        return 0, 1, None
    local = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local)
    dist.init_process_group("nccl")
    return dist.get_rank(), dist.get_world_size(), f"cuda:{local}"


def all_sum(values: list[float], world: int, device) -> list[float]:
    if world == 1:
        return [float(v) for v in values]
    t = torch.tensor(values, dtype=torch.float64, device=device)
    dist.all_reduce(t)
    return t.tolist()


def all_mean(values: list[float], world: int, device) -> list[float]:
    return [v / world for v in all_sum(values, world, device)]


def barrier(world: int) -> None:
    if world > 1:
        dist.barrier()


# ---------------------------------------------------------------- evaluation


@torch.no_grad()
def evaluate(policy, env, pool, tasks, rank, world, batch, mute=False, blind=False, max_new_tokens=512) -> dict:
    """Greedy decoding on a fixed task set, sharded across ranks."""
    policy.model.eval()
    mine = tasks[rank::world]
    sums = {k: [0.0, 0, 0] for k in KINDS}  # reward sum, count, valid count
    i = 0
    while i < len(mine):
        chunk = mine[i:i + batch]
        prompts = Prompts(env, policy, chunk, mute=mute, blind=blind)
        try:
            texts, _, _ = policy.generate(*prompts.select(list(range(len(chunk)))),
                                          max_new_tokens=max_new_tokens, sample=False)
        except torch.OutOfMemoryError:
            if batch == 1:
                raise
            batch //= 2
            continue
        i += len(chunk)
        for t, res in zip(chunk, pool.map(_score, [(task_dict(t), x) for t, x in zip(chunk, texts)])):
            acc = sums[t.kind]
            acc[0] += res["reward"]
            acc[1] += 1
            acc[2] += res["valid"]
    flat = all_sum([v for k in KINDS for v in sums[k]], world, policy.device)
    per = {k: flat[3 * j:3 * j + 3] for j, k in enumerate(KINDS)}
    total, count, n_valid = (sum(v[i] for v in per.values()) for i in range(3))
    return {"reward": total / max(count, 1), "valid": n_valid / max(count, 1), "n": int(count),
            "by_kind": {k: round(v[0] / v[1], 4) for k, v in per.items() if v[1]}}


# ---------------------------------------------------------------- checkpoints


def save_checkpoint(model, optim, step: int, best: float, ckpt_dir: Path) -> None:
    tmp, latest, old = ckpt_dir / ".tmp", ckpt_dir / "latest", ckpt_dir / ".old"
    shutil.rmtree(tmp, ignore_errors=True)
    model.save_pretrained(tmp)
    torch.save({"optim": optim.state_dict(), "step": step, "best": best}, tmp / "trainer.pt")
    shutil.rmtree(old, ignore_errors=True)
    if latest.exists():
        latest.rename(old)
    tmp.rename(latest)
    shutil.rmtree(old, ignore_errors=True)


def load_checkpoint(model, optim, ckpt: Path) -> tuple[int, float]:
    from peft import set_peft_model_state_dict
    from safetensors.torch import load_file

    set_peft_model_state_dict(model, load_file(ckpt / "adapter_model.safetensors"))
    state = torch.load(ckpt / "trainer.pt", map_location="cpu", weights_only=False)
    optim.load_state_dict(state["optim"])
    return state["step"], state["best"]


# ---------------------------------------------------------------- training


def train(args) -> None:
    from peft import LoraConfig, get_peft_model

    rank, world, device = setup_dist()
    out = Path(args.out)
    ckpt_dir = out / "ckpt"
    if rank == 0:
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        (out / "config.json").write_text(json.dumps(vars(args), indent=2))
    barrier(world)
    log = (out / "log.jsonl").open("a") if rank == 0 else None
    samples_log = (out / "samples.jsonl").open("a") if rank == 0 else None

    def say(msg: str) -> None:
        if rank == 0:
            print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

    torch.manual_seed(args.seed + rank)
    policy = OmniPolicy(args.model, device)
    for p in policy.model.parameters():
        p.requires_grad_(False)
    lora = LoraConfig(r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=0.0, target_modules=LORA_TARGETS)
    policy.model = get_peft_model(policy.model, lora)
    if args.grad_checkpointing:
        policy.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    params = [p for p in policy.model.parameters() if p.requires_grad]
    say(f"model {Path(args.model).name} on {policy.device} x{world}; "
        f"LoRA params {sum(p.numel() for p in params) / 1e6:.1f}M")
    optim = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0, betas=(0.9, 0.99))

    step, best = 0, -math.inf
    if (ckpt_dir / "latest" / "trainer.pt").exists():
        step, best = load_checkpoint(policy.model, optim, ckpt_dir / "latest")
        say(f"resumed from step {step}")

    env = VitalEnv()
    pool = mp.get_context("spawn").Pool(args.reward_workers, initializer=_init_worker)
    specs = MONO_SPECS if args.specs == ["mono"] else args.specs
    unknown = sorted(set(specs or []) - set(EDIT_SPECS))
    if unknown:
        raise SystemExit(f"unknown --specs {unknown}; choose from {sorted(EDIT_SPECS)} or 'mono'")
    kinds = args.kinds
    eval_tasks = sample_tasks(env, random.Random(f"eval-{args.seed}"), args.eval_n, kinds, specs)

    def run_eval(tag: str) -> dict:
        res = {"normal": evaluate(policy, env, pool, eval_tasks, rank, world, args.eval_batch, blind=args.blind,
                                  max_new_tokens=args.max_new_tokens)}
        if args.eval_mute:
            res["mute"] = evaluate(policy, env, pool, eval_tasks, rank, world, args.eval_batch, mute=True,
                                   blind=args.blind, max_new_tokens=args.max_new_tokens)
        if rank == 0:
            log.write(json.dumps({"eval": tag, "step": step, **res}) + "\n")
            log.flush()
            say(f"eval {tag} step {step}: " + ", ".join(
                f"{k} {v['reward']:.3f} (valid {v['valid']:.2f}; "
                + " ".join(f"{kind} {r:.3f}" for kind, r in v["by_kind"].items()) + ")" for k, v in res.items()))
        return res

    if step == 0 and args.eval_every > 0:
        run_eval("start")

    def shrink(name: str) -> bool:
        """After running out of memory, halve args.<name> so the caller can retry; False once it is 1.
        Retrying is local to this rank: ranks only meet at the gradient all-reduce, which waits for it."""
        optim.zero_grad(set_to_none=True)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        size = getattr(args, name)
        if size == 1:
            return False
        setattr(args, name, size // 2)
        say(f"step {step + 1}: out of memory, retrying with {name.replace('_', '-')} {size // 2}")
        return True

    started = time.time()
    while step < args.steps:
        over = bool(args.max_hours) and time.time() - started > args.max_hours * 3600
        if all_sum([float(over)], world, policy.device)[0] > 0:  # every rank stops at the same step
            say("time budget reached")
            break
        t0 = time.time()
        rng = random.Random(f"{args.seed}-{rank}-{step}")
        tasks = sample_tasks(env, rng, args.prompts, kinds, specs)
        prompts = Prompts(env, policy, tasks, blind=args.blind)
        order = [i for i in range(len(tasks)) for _ in range(args.group)]

        # 1. sample a group of completions per task
        policy.model.eval()
        while True:
            try:
                texts, comps, masks = [], [], []
                for i in range(0, len(order), args.gen_batch):
                    idx = order[i:i + args.gen_batch]
                    t_out, c_ids, _ = policy.generate(*prompts.select(idx), max_new_tokens=args.max_new_tokens,
                                                      temperature=args.temperature, sample=True)
                    texts += t_out
                    comps += list(c_ids.cpu())
                    masks += list(completion_mask(c_ids, policy.stop_ids).cpu())
                break
            except torch.OutOfMemoryError:
                if not shrink("gen_batch"):
                    raise
        t_gen = time.time() - t0

        # 2. render and score every completion
        scored = pool.map(_score, [(task_dict(tasks[i]), x) for i, x in zip(order, texts)])
        rewards = torch.tensor([s["reward"] for s in scored]).view(len(tasks), args.group)
        t_rew = time.time() - t0 - t_gen

        # 3. group-normalized advantages; groups with identical rewards carry no signal and are skipped
        mean, std = rewards.mean(1, keepdim=True), rewards.std(1, keepdim=True)
        adv = (rewards - mean) / (std + 1e-4) if args.scale_std else rewards - mean
        adv = adv.flatten()
        live = [j for j in range(len(order)) if std[j // args.group] > 1e-6]

        # 4. policy-gradient step, token-level mean over all live completion tokens on this rank
        policy.model.train()
        n_tokens = max(1.0, float(sum(masks[j].sum() for j in live)))
        while True:
            try:
                optim.zero_grad(set_to_none=False)
                for p in params:
                    if p.grad is None:
                        p.grad = torch.zeros_like(p)
                loss_sum, kl_sum = 0.0, 0.0
                for i in range(0, len(live), args.micro_batch):
                    idx = live[i:i + args.micro_batch]
                    width = int(max(masks[j].sum() for j in idx))
                    comp = torch.stack([comps[j][:width] for j in idx]).to(policy.device)
                    mask = torch.stack([masks[j][:width] for j in idx]).to(policy.device)
                    texts_mb, clips_mb = prompts.select([order[j] for j in idx])
                    logp = token_logprobs(policy, texts_mb, clips_mb, comp, mask)
                    a = adv[idx].to(policy.device).unsqueeze(1)
                    per_token = -a * logp
                    if args.kl > 0:
                        with torch.no_grad(), policy.model.disable_adapter():
                            ref = token_logprobs(policy, texts_mb, clips_mb, comp, mask)
                        kl = torch.exp(ref - logp) - (ref - logp) - 1
                        per_token = per_token + args.kl * kl
                        kl_sum += (kl * mask).sum().item()
                    loss = (per_token * mask).sum() / n_tokens
                    loss.backward()
                    loss_sum += loss.item()
                break
            except torch.OutOfMemoryError:
                logp = per_token = loss = None
                if not shrink("micro_batch"):
                    raise
        if world > 1:
            for p in params:
                dist.all_reduce(p.grad)
                p.grad /= world
        grad_norm = float(torch.nn.utils.clip_grad_norm_(params, args.max_grad_norm))
        optim.step()

        step += 1
        valid = [s["valid"] for s in scored]
        stats = all_mean([float(rewards.mean()), float(rewards.std()), float(np.mean(valid)),
                          float(np.mean([s["n_changed"] == 0 for s in scored])),
                          1 - len(live) / len(order), float(np.mean([m.sum() for m in masks])),
                          loss_sum, kl_sum / n_tokens, grad_norm], world, policy.device)
        row = dict(zip(["reward", "reward_std", "valid", "no_change", "skipped_groups", "length", "loss", "kl",
                        "grad_norm"], stats))
        per = all_sum([x for k in KINDS for x in (sum(float(rewards[i].sum()) for i, t in enumerate(tasks) if t.kind == k),
                                                  args.group * sum(t.kind == k for t in tasks))], world, policy.device)
        row["by_kind"] = {k: round(per[2 * j] / per[2 * j + 1], 4) for j, k in enumerate(KINDS) if per[2 * j + 1]}
        row.update(step=step, secs=round(time.time() - t0, 1), gen_secs=round(t_gen, 1), reward_secs=round(t_rew, 1))
        if rank == 0:
            log.write(json.dumps(row) + "\n")
            log.flush()
            for j in range(0, len(order), max(1, len(order) // 2))[:2]:
                samples_log.write(json.dumps({"step": step, "kind": tasks[order[j]].kind,
                                              "instruction": tasks[order[j]].instruction,
                                              "completion": texts[j], "reward": scored[j]["reward"]}) + "\n")
            samples_log.flush()
            say(f"step {step}: reward {row['reward']:.3f} valid {row['valid']:.2f} skipped {row['skipped_groups']:.2f}"
                f" len {row['length']:.0f} grad {grad_norm:.2f} ({row['secs']}s, gen {row['gen_secs']}s) "
                + " ".join(f"{k} {r:.3f}" for k, r in row["by_kind"].items()))

        if args.eval_every > 0 and step % args.eval_every == 0:
            res = run_eval("periodic")
            if rank == 0 and res["normal"]["reward"] > best:
                best = res["normal"]["reward"]
                shutil.rmtree(out / "best", ignore_errors=True)
                policy.model.save_pretrained(out / "best")
        if step % args.save_every == 0 or step == args.steps:
            if rank == 0:
                save_checkpoint(policy.model, optim, step, best, ckpt_dir)
            barrier(world)

    if rank == 0:
        save_checkpoint(policy.model, optim, step, best, ckpt_dir)
        policy.model.save_pretrained(out / "final")
    if args.eval_every > 0:
        run_eval("final")
    if rank == 0:
        (out / "status.json").write_text(json.dumps({"done": step >= args.steps, "step": step, "best": best}))
    pool.close()
    if world > 1:
        dist.destroy_process_group()


def main() -> None:
    p = argparse.ArgumentParser(prog="synth-rl-grpo", description=__doc__.split("\n\n")[0])
    p.add_argument("--model", required=True, help="Qwen2.5-Omni or Qwen3-Omni folder")
    p.add_argument("--out", default="outputs/grpo")
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--max-hours", type=float, default=0, help="stop and save after this wall time (0: no limit)")
    p.add_argument("--prompts", type=int, default=8, help="tasks per step on each rank")
    p.add_argument("--group", type=int, default=8, help="completions sampled per task")
    p.add_argument("--kinds", nargs="+", default=["edit", "repair", "match"], choices=KINDS,
                   help="task mix, sampled uniformly; repair and match (rebuild a target sound) need listening")
    p.add_argument("--specs", nargs="*", default=["mono"], help="edit families; 'mono' drops stereo width")
    p.add_argument("--blind", action="store_true", help="hide current parameter values so the policy must listen")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--max-new-tokens", type=int, default=512, help="match replies can set every parameter")
    p.add_argument("--gen-batch", type=int, default=64, help="sequences per generate call")
    p.add_argument("--micro-batch", type=int, default=4, help="sequences per backward pass")
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--lora-r", type=int, default=32)
    p.add_argument("--lora-alpha", type=int, default=64)
    p.add_argument("--kl", type=float, default=0.0, help="KL penalty to the base model (0: off)")
    p.add_argument("--scale-std", action=argparse.BooleanOptionalAction, default=True,
                   help="divide advantages by the group std (GRPO); --no-scale-std keeps raw differences")
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--grad-checkpointing", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--reward-workers", type=int, default=max(1, min(16, (os.cpu_count() or 2) // 2)))
    p.add_argument("--eval-every", type=int, default=25)
    p.add_argument("--eval-n", type=int, default=64)
    p.add_argument("--eval-batch", type=int, default=16)
    p.add_argument("--eval-mute", action=argparse.BooleanOptionalAction, default=True,
                   help="also evaluate with silent audio, to see how much the policy relies on listening")
    p.add_argument("--save-every", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    train(p.parse_args())


if __name__ == "__main__":
    main()
