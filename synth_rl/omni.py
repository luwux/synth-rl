"""Qwen Omni thinkers (Qwen2.5-Omni, Qwen3-Omni) as policies: prompt building, generation, log-probs.

Only the Thinker (text output) is loaded; the talker and the vision encoder are not needed.
Audio is downmixed to mono and resampled to 16 kHz, which is what these models accept.
"""

from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
from scipy.signal import resample_poly

from .agents import SYSTEM, first_json, parse_edits, prompt_text
from .vital_env import RENDER_DUR, SR

OMNI_SR = 16000
FAMILIES = {
    "qwen2.5omni": ("Qwen2_5OmniProcessor", "Qwen2_5OmniThinkerForConditionalGeneration"),
    "qwen3omni": ("Qwen3OmniMoeProcessor", "Qwen3OmniMoeThinkerForConditionalGeneration"),
}


def family_of(path: str | Path) -> str:
    model_type = json.loads((Path(path) / "config.json").read_text()).get("model_type", "")
    return "qwen3omni" if "qwen3_omni" in model_type else "qwen2.5omni"


def to_model_audio(audio: np.ndarray) -> np.ndarray:
    mono = audio.mean(axis=0) if audio.ndim == 2 else audio
    return resample_poly(mono, 160, 441).astype(np.float32)


def default_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class OmniPolicy:
    def __init__(self, path: str | Path, device: str | None = None, dtype=torch.bfloat16):
        import transformers as T

        self.path = str(path)
        self.family = family_of(path)
        proc_cls, model_cls = FAMILIES[self.family]
        self.proc = getattr(T, proc_cls).from_pretrained(self.path)
        self.proc.tokenizer.padding_side = "left"
        tok = self.proc.tokenizer
        self.stop_ids = [tok.convert_tokens_to_ids("<|im_end|>")] + ([tok.eos_token_id] if tok.eos_token_id is not None else [])
        self.device = device or default_device()
        # Load straight onto the device: one CPU copy per rank would need hundreds of GB of RAM for 30B on 8 GPUs.
        self.model = getattr(T, model_cls).from_pretrained(self.path, dtype=dtype, device_map=self.device)
        if hasattr(self.model, "visual"):
            self.model.visual = None
        self.model.eval()
        # The Omni thinkers have no `logits_to_keep` and compute vocabulary logits at every position; only the
        # last positions are needed, and the rest cost more memory than the KV cache.
        self._keep = 0
        self.model.lm_head.register_forward_pre_hook(self._keep_last)

    def _keep_last(self, module, args):
        if self._keep and args[0].shape[1] > self._keep:
            return (args[0][:, -self._keep:],)

    @contextmanager
    def last_logits(self, n: int):
        """Inside this block, forward passes return logits for the last `n` positions only."""
        self._keep = n
        try:
            yield
        finally:
            self._keep = 0

    def chat(self, user_text: str, n_audio: int, extra_text: str = "Second clip: target sound.") -> str:
        content = [{"type": "text", "text": user_text}, {"type": "audio", "audio": "current.wav"}]
        if n_audio == 2:
            content += [{"type": "text", "text": extra_text}, {"type": "audio", "audio": "target.wav"}]
        msgs = [{"role": "system", "content": [{"type": "text", "text": SYSTEM}]},
                {"role": "user", "content": content}]
        return self.proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)

    def encode(self, texts: list[str], audios: list[np.ndarray]):
        """`audios` is the flat list of every clip, in prompt order. Padding stops at the clip length."""
        inp = self.proc(text=texts, audio=audios, sampling_rate=OMNI_SR, return_tensors="pt", padding=True,
                        audio_kwargs={"max_length": int(RENDER_DUR * OMNI_SR) + 1600})
        inp = inp.to(self.device)
        inp["input_features"] = inp["input_features"].to(self.model.dtype)
        return inp

    @torch.no_grad()
    def generate(self, texts: list[str], audios: list[np.ndarray], max_new_tokens: int = 512,
                 temperature: float = 1.0, sample: bool = True) -> tuple[list[str], torch.Tensor, dict]:
        """Returns decoded completions, completion token ids (right-padded), and the encoded prompt."""
        inp = self.encode(texts, audios)
        pad = self.proc.tokenizer.pad_token_id
        kwargs = {"do_sample": sample, "max_new_tokens": max_new_tokens, "eos_token_id": self.stop_ids,
                  "pad_token_id": pad if pad is not None else self.stop_ids[0]}
        if sample:
            kwargs.update(temperature=temperature, top_p=1.0, top_k=0)
        with self.last_logits(1):
            out = self.model.generate(**inp, **kwargs)
        sequences = out.sequences if hasattr(out, "sequences") else out
        completion = sequences[:, inp["input_ids"].shape[1]:]
        text = self.proc.tokenizer.batch_decode(completion, skip_special_tokens=True)
        return text, completion, inp


class OmniAgent:
    """Zero-shot or fine-tuned Omni thinker used as a benchmark agent (greedy decoding)."""

    def __init__(self, path: str, adapter: str | None = None, mute_audio: bool = False, device: str | None = None):
        self.policy = OmniPolicy(path, device)
        if adapter:
            from peft import PeftModel

            self.policy.model = PeftModel.from_pretrained(self.policy.model, adapter).eval()
        self.mute_audio = mute_audio
        self.name = Path(path).name + ("+lora" if adapter else "") + ("+mute" if mute_audio else "")

    def propose(self, instruction: str, params: dict, audio: np.ndarray, target: np.ndarray | None = None) -> dict:
        clips = [audio] + ([target] if target is not None else [])
        clips = [np.zeros_like(c) if self.mute_audio else c for c in clips]
        text = self.policy.chat(prompt_text(params, instruction), len(clips))
        out, _, _ = self.policy.generate([text], [to_model_audio(c) for c in clips], sample=False)
        reason = (first_json(out[0]) or {}).get("reason", "")
        return {"edits": parse_edits(out[0]), "reason": str(reason), "raw": out[0]}
