#!/usr/bin/env bash
# SUPERFLY launcher for macOS and Linux:  ./start-superfly.sh   (or double-click Start-SUPERFLY-Mac.command)
# First run: sets up a private Python environment, installs the packages,
# downloads the fly's connectome and its language model, then opens the Lab.
set -e
cd "$(dirname "$0")"
PY=$(command -v python3 || command -v python || true)
if [ -z "$PY" ]; then
  echo "SUPERFLY needs Python 3.10 or newer: https://www.python.org/downloads/"
  exit 1
fi
if [ ! -x .venv/bin/python ]; then
  echo "Setting up SUPERFLY's Python environment (first run only)..."
  "$PY" -m venv .venv
fi
. .venv/bin/activate
if [ ! -f .venv/superfly_installed ]; then
  python -m pip install --upgrade pip
  python -m pip install -r requirements-superfly.txt
  touch .venv/superfly_installed
fi
exec python -m superfly.quickstart "$@"
