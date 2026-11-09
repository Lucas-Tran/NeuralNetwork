#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-}"

if [[ -n "$PYTHON" ]]; then
	if ! command -v "$PYTHON" >/dev/null 2>&1; then
		printf 'Python interpreter not found: %s\n' "$PYTHON" >&2
		exit 1
	fi
	if ! "$PYTHON" -c 'import sys; sys.exit(sys.version_info < (3, 10))' >/dev/null 2>&1; then
		printf 'Python 3.10 or newer is required.\n' >&2
		exit 1
	fi
else
	for candidate in python3.12 python3.11 python3.10 python3; do
		if command -v "$candidate" >/dev/null 2>&1 && \
			"$candidate" -c 'import sys; sys.exit(sys.version_info < (3, 10))' >/dev/null 2>&1; then
			PYTHON="$candidate"
			break
		fi
	done
fi

if [[ -z "$PYTHON" ]]; then
	printf 'Python 3.10 or newer is required. Set PYTHON to its executable and retry.\n' >&2
	exit 1
fi

printf 'Creating local environment with %s (%s)\n' "$PYTHON" "$($PYTHON --version)"
"$PYTHON" -m venv "$ROOT_DIR/.venv"
"$ROOT_DIR/.venv/bin/python" -m pip install --upgrade pip
"$ROOT_DIR/.venv/bin/python" -m pip install -r "$ROOT_DIR/requirements.txt"

ACTIVATE_LINK="$ROOT_DIR/activate"
ACTIVATE_TARGET=".venv/bin/activate"
if [[ -L "$ACTIVATE_LINK" ]]; then
	if [[ "$(readlink "$ACTIVATE_LINK")" != "$ACTIVATE_TARGET" ]]; then
		printf 'The activate symlink points to an unexpected target: %s\n' "$(readlink "$ACTIVATE_LINK")" >&2
		exit 1
	fi
elif [[ -e "$ACTIVATE_LINK" ]]; then
	printf 'Cannot create activate symlink: %s already exists and is not a symlink.\n' "$ACTIVATE_LINK" >&2
	exit 1
else
	ln -s "$ACTIVATE_TARGET" "$ACTIVATE_LINK"
fi

printf '\nSetup complete. Activate the environment and start the app with:\n'
printf 'source "%s/activate"\n' "$ROOT_DIR"
printf 'python main.py\n'