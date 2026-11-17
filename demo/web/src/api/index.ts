import { httpApi } from "./client";
import { createMockApi } from "./mock";
import type { Api } from "./types";

/** The real server unless the page is opened with ?mock=1 (development fixtures, no server needed). */
export const isMock = new URLSearchParams(window.location.search).get("mock") === "1";
export const api: Api = isMock ? createMockApi() : httpApi;
export * from "./types";
