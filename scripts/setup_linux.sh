#!/usr/bin/env bash
set -euo pipefail

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e '.[dev,audio]'

cat <<MSG
Setup complete.
Activate with: source .venv/bin/activate
Run with: python -m cortana
MSG
