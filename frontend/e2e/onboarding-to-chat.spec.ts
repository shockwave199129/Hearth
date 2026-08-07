/**
 * Onboarding → Chat smoke test with a mocked backend.
 *
 * CI cannot run the real STT/LLM/TTS stack, and the review's medium-term
 * item only asks for one Playwright path that proves the first-run flow
 * reaches Talk. Every `/api/*` call is fulfilled here; the WebSocket is
 * left to fail closed (Chat tolerates a disconnected socket).
 */
import { expect, test, type Page, type Route } from "@playwright/test";

const SETUP_COMPLETE = {
  hardware: { ram_gb: 16, cpu_count: 8, gpu_name: null, vram_gb: 0 },
  tier: "C",
  tts_engine: "kokoro",
  gpu_vendor: "none",
  complete: true,
};

const TIER_STATUS = {
  tier: "C",
  llm_gguf: "fake.gguf",
  stt_model: "fake-stt",
  tts_engine: "kokoro",
  hardware: { ram_gb: 16, gpu_name: null, vram_gb: 0 },
};

let activeProfile: Record<string, unknown> | null = null;

async function mockBackend(page: Page) {
  await page.route("**/api/**", async (route: Route) => {
    const req = route.request();
    const url = new URL(req.url());
    const path = url.pathname;
    const method = req.method();

    if (path === "/api/setup/status" && method === "GET") {
      return route.fulfill({ json: SETUP_COMPLETE });
    }
    if (path === "/api/setup/progress" && method === "GET") {
      return route.fulfill({ json: { step: "done", log_tail: [], error: null } });
    }
    if (path === "/api/profile" && method === "GET") {
      if (!activeProfile) {
        return route.fulfill({ status: 404, json: { detail: "no profile saved yet" } });
      }
      return route.fulfill({ json: activeProfile });
    }
    if (path === "/api/onboarding" && method === "POST") {
      const body = req.postDataJSON() as Record<string, unknown>;
      activeProfile = {
        user_id: "e2e-user",
        created_at: new Date().toISOString(),
        relationship_general_trust: 0,
        relationship_vulnerability_trust: 0,
        relationship_advice_trust: 0,
        relationship_consistency_confidence: 0,
        relationship_boundaries: "normal",
        relationship_life_model: "unknown",
        emoji_usage: "minimal",
        ...body,
      };
      return route.fulfill({ json: activeProfile });
    }
    if (path === "/api/status" && method === "GET") {
      return route.fulfill({ json: TIER_STATUS });
    }
    if (path === "/api/chat_history" && method === "GET") {
      return route.fulfill({ json: { items: [], has_more: false } });
    }
    if (path === "/api/checkin" && method === "GET") {
      return route.fulfill({
        json: { last_checkin_at: null, days_since_last_checkin: null },
      });
    }
    if (path === "/api/safety/status" && method === "GET") {
      return route.fulfill({
        json: {
          recent_crisis_events: 0,
          last_escalation_at: null,
          safety_log_retention_policy: "test",
          safety_log_entries_retained: 0,
        },
      });
    }
    if (path === "/api/memories" && method === "GET") {
      return route.fulfill({ json: { items: [], has_more: false, limit: 50, offset: 0 } });
    }
    if (path === "/api/skills" && method === "GET") {
      return route.fulfill({ json: [] });
    }

    return route.fulfill({ status: 404, json: { detail: `unmocked ${method} ${path}` } });
  });
}

test.beforeEach(async ({ page }) => {
  activeProfile = null;
  await mockBackend(page);
});

test("onboarding completes and lands on chat", async ({ page }) => {
  await page.goto("/");

  await expect(page.getByRole("heading", { name: "Let's get acquainted" })).toBeVisible({
    timeout: 15_000,
  });

  await page.getByPlaceholder("Your name").fill("Ada");
  await page.getByPlaceholder("e.g. Sage, River, Companion").fill("Ember");
  await page.getByRole("button", { name: "Continue" }).click();

  // Skip optional steps 1–4 (About / Stressors / Style / Voice).
  for (let i = 0; i < 4; i++) {
    await page.getByRole("button", { name: "Continue" }).click();
  }

  await expect(page.getByRole("heading", { name: /safety net/i })).toBeVisible();
  await page.getByRole("button", { name: "Start talking" }).click();

  await expect(page.getByText("Ember").first()).toBeVisible({ timeout: 10_000 });
  await expect(page).toHaveURL(/\/chat/);
});
