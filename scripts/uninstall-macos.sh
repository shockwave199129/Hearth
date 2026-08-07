#!/bin/sh
# Manual macOS fallback when Hearth.app was already moved to the Trash.
#
# Prefer Settings → Uninstall Hearth while the app is installed: only the
# bundled backend knows how to export the encrypted profile identity before
# removing conversations, memories, models, and packages. This script can
# run that same backend from an app bundle supplied with --app.

set -eu

APP_PATH="/Applications/Hearth.app"
if [ "${1:-}" = "--app" ]; then
  APP_PATH="${2:?Usage: uninstall-macos.sh [--app /path/to/Hearth.app]}"
fi

BACKEND="$APP_PATH/Contents/Resources/resources/backend/hearth-backend"
if [ ! -x "$BACKEND" ]; then
  echo "Hearth.app was not found at: $APP_PATH" >&2
  echo "To preserve your profile identity, mount the Hearth DMG or restore the app," >&2
  echo "then run: $0 --app /path/to/Hearth.app" >&2
  exit 1
fi

"$BACKEND" --uninstall-cleanup
echo "Hearth's models, packages, memories, conversations, and crash logs were removed."
echo "Profile identity was retained for the next installation."
