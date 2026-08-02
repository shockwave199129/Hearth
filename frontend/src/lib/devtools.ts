/** Blocks the keyboard shortcuts that open the developer tools window.
 *
 * This is the second of two layers, and the weaker one. What actually keeps
 * devtools out of a shipped build is desktop/src-tauri/Cargo.toml declaring
 * `tauri = { features = [] }` — without the `devtools` feature, Tauri never
 * compiles the inspector into a release binary at all, and no keypress can
 * summon what isn't there. This handler covers the gap in between: the
 * webview still routes these chords somewhere, and on Linux/WebKitGTK the
 * inspector can be reachable when developer extras are on.
 *
 * It cannot block F12 in a plain browser tab (`pnpm dev`), where the browser
 * itself handles the key before the page ever sees it. That's a dev-only
 * surface, not something an installed Hearth exposes.
 */

/** Chords across the platforms/webviews Hearth ships on: Ctrl+Shift+I/J/C on
 * Windows and Linux, Cmd+Alt+I/J/C on macOS, plus bare F12. */
function isDevtoolsChord(event: KeyboardEvent): boolean {
  if (event.key === "F12") return true;

  const key = event.key.toUpperCase();
  if (key !== "I" && key !== "J" && key !== "C") return false;

  const windowsChord = event.ctrlKey && event.shiftKey;
  const macChord = event.metaKey && event.altKey;
  return windowsChord || macChord;
}

export function blockDevtoolsShortcuts(): void {
  document.addEventListener(
    "keydown",
    (event) => {
      if (!isDevtoolsChord(event)) return;
      event.preventDefault();
      event.stopPropagation();
    },
    { capture: true },
  );
}
