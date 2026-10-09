#!/usr/bin/env bash
# SUPERFLY for macOS and Linux. One line in a terminal:
#   curl -fsSL https://raw.githubusercontent.com/gondwanagenesis/fly-brain/claude/trusting-newton-ip3rl0/installer/superfly-install.sh | bash
# (the Mac app in dist/SUPERFLY-Mac.zip runs the same thing).
# Installs into ~/.superfly (no admin rights): the code, a private Python via
# uv, the packages; then starts the Lab, which fetches the fly's connectome and
# its language model on first run. Run it again to start SUPERFLY later.
set -euo pipefail
ROOT="${SUPERFLY_HOME:-$HOME/.superfly}"
APP="$ROOT/app"
BRANCH="claude/trusting-newton-ip3rl0"
TARBALL="https://github.com/gondwanagenesis/fly-brain/archive/refs/heads/$BRANCH.tar.gz"
say() { printf '\n\033[1;33m== %s\033[0m\n' "$1"; }
printf '\n\033[1;35m   S U P E R F L Y   -  an uplifted fruit fly, still a fly\033[0m\n   installing into %s\n' "$ROOT"
mkdir -p "$ROOT"
for a in "$@"; do [ "$a" = "--update" ] && rm -rf "$APP"; done
if [ ! -f "$APP/superfly/quickstart.py" ]; then
  say "Downloading SUPERFLY"
  rm -rf "$ROOT/unpack" && mkdir -p "$ROOT/unpack"
  curl -fL --progress-bar "$TARBALL" | tar -xz -C "$ROOT/unpack"
  mv "$ROOT/unpack"/* "$APP" && rm -rf "$ROOT/unpack"
fi
UV="$ROOT/bin/uv"
if [ ! -x "$UV" ]; then
  say "Getting a private Python manager (uv)"
  curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR="$ROOT/bin" UV_NO_MODIFY_PATH=1 INSTALLER_NO_MODIFY_PATH=1 sh >/dev/null
fi
export UV_PYTHON_INSTALL_DIR="$ROOT/python" UV_CACHE_DIR="$ROOT/cache"
cd "$APP"
if [ ! -x .venv/bin/python ]; then
  say "Setting up Python 3.12 (private to SUPERFLY)"
  "$UV" venv --python 3.12 .venv
fi
if [ ! -f .venv/superfly_installed ]; then
  say "Installing packages (PyTorch and friends, about 1 GB, first run only)"
  "$UV" pip install --python .venv/bin/python --index-strategy unsafe-best-match -r requirements-superfly.txt
  touch .venv/superfly_installed
fi
say "Starting SUPERFLY (the first start also downloads the fly and its language model, ~6.5 GB)"
ARGS=()
for a in "$@"; do [ "$a" != "--update" ] && ARGS+=("$a"); done
exec .venv/bin/python -m superfly.quickstart "${ARGS[@]+"${ARGS[@]}"}"
