#!/usr/bin/env bash
# Build a SIGNED + NOTARIZED MeshThatWorks.dmg that anyone can download and open
# with zero Gatekeeper warnings.
#
# ─── ONE-TIME SETUP (only you can do these — they need your Apple login) ──────
#
# 1. Create a "Developer ID Application" certificate:
#      Xcode → Settings → Accounts → (your team) → Manage Certificates →
#      "+" → "Developer ID Application".   (You must be the Account Holder.)
#    Confirm:  security find-identity -v -p codesigning | grep "Developer ID"
#
# 2. Store your notarization password in the keychain ONCE, so it never has to
#    be pasted again (and never appears in this file):
#      xcrun notarytool store-credentials "MTW_NOTARY" \
#        --apple-id "you@appleid.com" \
#        --team-id  "UFAJBW55T6" \
#        --password "xxxx-xxxx-xxxx-xxxx"     # app-specific pw from appleid.apple.com
#
# Then just run:  ./scripts/release-dmg.sh
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

NOTARY_PROFILE="${NOTARY_PROFILE:-MTW_NOTARY}"

HERE="$(cd "$(dirname "$0")/.." && pwd)"
cd "$HERE"

# Auto-detect the Developer ID Application identity from the keychain.
IDENTITY="$(security find-identity -v -p codesigning | sed -n 's/.*"\(Developer ID Application:[^"]*\)".*/\1/p' | head -1)"
if [[ -z "$IDENTITY" ]]; then
  echo "✋ No 'Developer ID Application' certificate found in your keychain."
  echo "   Create one: Xcode → Settings → Accounts → Manage Certificates → + → Developer ID Application"
  echo "   (Step 1 in this script's header.)"
  exit 1
fi
echo "==> Signing identity: $IDENTITY"
export APPLE_SIGNING_IDENTITY="$IDENTITY"

if ! xcrun notarytool history --keychain-profile "$NOTARY_PROFILE" >/dev/null 2>&1; then
  echo "✋ No notarization profile '$NOTARY_PROFILE' stored."
  echo "   Run the 'notarytool store-credentials' command in this script's header (Step 2)."
  exit 1
fi

echo "==> Rebuilding the mtw sidecar so it's fresh…"
./scripts/sync-sidecar.sh

echo "==> Building + signing the app (hardened runtime + entitlements)…"
pnpm tauri build

DMG="$(ls -1 src-tauri/target/release/bundle/dmg/*.dmg | head -1)"
APP="src-tauri/target/release/bundle/macos/MeshThatWorks.app"

echo "==> Notarizing $DMG (waits for Apple — usually 1–5 min)…"
xcrun notarytool submit "$DMG" --keychain-profile "$NOTARY_PROFILE" --wait

echo "==> Stapling the ticket so it works offline…"
xcrun stapler staple "$APP"
xcrun stapler staple "$DMG"

echo
echo "==> Gatekeeper check (should say 'accepted / source=Notarized Developer ID'):"
spctl -a -vvv -t install "$APP" || true

echo
echo "✅ Ship this — opens cleanly on any Apple Silicon Mac:"
echo "    $DMG"
