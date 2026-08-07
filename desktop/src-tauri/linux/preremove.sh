#!/bin/sh
# Called by dpkg/rpm before Hearth's package files are removed. Preserve the
# current profile identity but delete downloaded runtime assets and history.
#
# Debian invokes prerm during package upgrades too; never wipe user data then.
case "${1:-}" in
  upgrade|failed-upgrade|abort-install|abort-upgrade) exit 0 ;;
esac

BACKEND=""
for candidate in \
  /usr/lib/Hearth/resources/backend/hearth-backend \
  /usr/lib/hearth/resources/backend/hearth-backend \
  /opt/Hearth/resources/backend/hearth-backend; do
  if [ -x "$candidate" ]; then
    BACKEND="$candidate"
    break
  fi
done

[ -n "$BACKEND" ] || exit 0

# apt/dpkg generally run maintainer scripts as root. Execute as the invoking
# desktop user so $HOME and the Secret Service keyring point at their Hearth
# data rather than /root. No SUDO_USER is available for every package-manager
# invocation, so root removals safely become a no-op instead of touching an
# unknown user's profile.
if [ "$(id -u)" -eq 0 ]; then
  if [ -n "${SUDO_USER:-}" ] && [ "$SUDO_USER" != "root" ]; then
    user_home="$(getent passwd "$SUDO_USER" | cut -d: -f6)"
    exec su -s /bin/sh "$SUDO_USER" -c "HOME='$user_home' '$BACKEND' --uninstall-cleanup" || true
  fi
  exit 0
fi

"$BACKEND" --uninstall-cleanup || true
