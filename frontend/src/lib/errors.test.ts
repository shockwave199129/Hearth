import { describe, expect, it, vi } from "vitest";
import { friendlyActionError, friendlyFetchError } from "./errors";

describe("friendlyFetchError", () => {
  it("logs the real error and returns a fixed user-facing string", () => {
    const spy = vi.spyOn(console, "error").mockImplementation(() => undefined);
    expect(friendlyFetchError(new Error("status 503"), "useProfile")).toBe(
      "Couldn't reach the companion — is the backend running?",
    );
    expect(spy).toHaveBeenCalledOnce();
    spy.mockRestore();
  });
});

describe("friendlyActionError", () => {
  it("returns the caller-supplied fallback, never the raw status", () => {
    const spy = vi.spyOn(console, "error").mockImplementation(() => undefined);
    expect(friendlyActionError(new Error("status 500"), "save", "Couldn't save.")).toBe(
      "Couldn't save.",
    );
    spy.mockRestore();
  });
});
