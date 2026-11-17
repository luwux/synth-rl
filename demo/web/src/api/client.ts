import { EventSourceParserStream } from "eventsource-parser/stream";
import type { AgentInfo, Api, ChatEvent, ChatRequest, State } from "./types";

export class ApiError extends Error {
  status: number;
  constructor(status: number, message: string) {
    super(message);
    this.status = status;
  }
}

async function readError(res: Response): Promise<string> {
  const text = await res.text().catch(() => "");
  try {
    const body = JSON.parse(text);
    // FastAPI puts the message in "detail"; the contract's error event uses "message".
    const detail = body?.detail ?? body?.message;
    if (typeof detail === "string") return detail;
    if (detail) return JSON.stringify(detail);
  } catch {
    /* not JSON */
  }
  return text.slice(0, 300) || res.statusText || `HTTP ${res.status}`;
}

async function request<T>(method: "GET" | "POST", path: string, body?: unknown): Promise<T> {
  let res: Response;
  try {
    res = await fetch(path, {
      method,
      headers: body === undefined ? undefined : { "Content-Type": "application/json" },
      body: body === undefined ? undefined : JSON.stringify(body),
    });
  } catch {
    throw new ApiError(0, "The synth server is not reachable.");
  }
  if (!res.ok) throw new ApiError(res.status, await readError(res));
  return (await res.json()) as T;
}

const EVENT_NAMES = new Set(["start", "listening", "proposal", "revision", "done", "error"]);

async function* chatStream(body: ChatRequest, signal: AbortSignal): AsyncIterable<ChatEvent> {
  let res: Response;
  try {
    res = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json", Accept: "text/event-stream" },
      body: JSON.stringify(body),
      signal,
    });
  } catch (err) {
    if (signal.aborted) return;
    throw new ApiError(0, "The synth server is not reachable.");
  }
  if (!res.ok || !res.body) throw new ApiError(res.status, await readError(res));
  const events = res.body
    .pipeThrough(new TextDecoderStream())
    .pipeThrough(new EventSourceParserStream({ onError: "terminate" }));
  const reader = events.getReader();
  try {
    for (;;) {
      const { value, done } = await reader.read();
      if (done) return;
      const name = value.event ?? "message";
      if (!EVENT_NAMES.has(name)) continue;
      let data: unknown;
      try {
        data = JSON.parse(value.data);
      } catch {
        continue;
      }
      yield { event: name, data } as ChatEvent;
    }
  } finally {
    reader.releaseLock();
  }
}

export const httpApi: Api = {
  state: () => request<State>("GET", "/api/state"),
  agents: () => request<AgentInfo[]>("GET", "/api/agents"),
  newPatch: (body) => request<State>("POST", "/api/new_patch", body),
  set: (edits) => request<State>("POST", "/api/set", { edits }),
  undo: () => request<State>("POST", "/api/undo"),
  redo: () => request<State>("POST", "/api/redo"),
  openInVital: (rev) => request<{ ok: boolean }>("POST", "/api/open_in_vital", { rev }),
  audioUrl: (rev) => rev.audio_url,
  presetUrl: (rev) => `/api/preset/${rev}.vital`,
  chat: chatStream,
};
