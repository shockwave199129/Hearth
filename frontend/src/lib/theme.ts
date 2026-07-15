export type Theme = "system" | "dark" | "light";

const STORAGE_KEY = "companion:theme";

export function getStoredTheme(): Theme {
  const stored = localStorage.getItem(STORAGE_KEY);
  return stored === "dark" || stored === "light" ? stored : "system";
}

/** Mutates document.documentElement directly — deliberately not tied to
 * React state, so it can run once at app boot (main.tsx) before any route
 * renders, not just while Settings happens to be mounted. */
export function applyTheme(theme: Theme): void {
  if (theme === "system") document.documentElement.removeAttribute("data-theme");
  else document.documentElement.setAttribute("data-theme", theme);
}

export function setStoredTheme(theme: Theme): void {
  localStorage.setItem(STORAGE_KEY, theme);
  applyTheme(theme);
}
