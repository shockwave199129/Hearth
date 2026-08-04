/** Suppresses the right-click context menu across the UI, except inside
 * text fields.
 *
 * Done in the webview rather than in Tauri config because the native menu
 * is only reliably suppressed by Tauri on some platforms in release builds.
 * WebKitGTK (Linux) and every `tauri dev` run still show it, and this also
 * covers the plain-browser dev server. One listener at the document root
 * with capture:true, so nothing downstream can re-enable it.
 *
 * Text fields keep their menu on purpose: this is the app where someone
 * types out what's weighing on them, sometimes pasting in something they
 * drafted elsewhere, and taking away paste and spellcheck suggestions there
 * is friction in exactly the wrong place.
 */

/** input types that aren't text entry — a right-click on a checkbox has
 * nothing worth offering, so those stay blocked with everything else. */
const NON_TEXT_INPUT_TYPES = new Set([
  "button",
  "checkbox",
  "color",
  "file",
  "image",
  "radio",
  "range",
  "reset",
  "submit",
]);

function isTextEntry(target: EventTarget | null): boolean {
  const field = (target as Partial<HTMLElement> | null)?.closest?.("input, textarea");
  if (field instanceof HTMLTextAreaElement) return true;
  if (field instanceof HTMLInputElement) return !NON_TEXT_INPUT_TYPES.has(field.type);
  return false;
}

export function disableContextMenu(): void {
  document.addEventListener(
    "contextmenu",
    (event) => {
      if (isTextEntry(event.target)) return;
      event.preventDefault();
    },
    { capture: true },
  );
}
