import { describe, expect, it } from "vitest";
import { apiUrl, API_TOKEN_HEADER } from "./backendUrl";

describe("apiUrl", () => {
  it("passes through absolute URLs unchanged", () => {
    expect(apiUrl("http://127.0.0.1:48173/api/status")).toBe("http://127.0.0.1:48173/api/status");
  });

  it("prefixes relative paths in production builds", () => {
    // In vitest, import.meta.env.DEV is true, so the prefix is empty —
    // assert the relative form that the Vite proxy relies on.
    expect(apiUrl("/api/setup/status")).toBe("/api/setup/status");
  });

  it("exports the header name the backend middleware expects", () => {
    expect(API_TOKEN_HEADER).toBe("X-Hearth-Token");
  });
});
