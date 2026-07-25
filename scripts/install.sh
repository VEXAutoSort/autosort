#!/usr/bin/env bash
# One-shot setup: a Python 3.12 venv + every dependency.
# Run once:  ./scripts/install.sh
set -e
cd "$(dirname "$0")/.."

command -v python3.12 >/dev/null || { echo "Install Python 3.12 first (brew install python@3.12)"; exit 1; }

python3.12 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e .

echo
echo "Done. Next:"
echo "  source .venv/bin/activate"
echo "  ./scripts/find_ports.sh      # get your USB ports, put them in config.yaml"
echo "  python run.py --dry-run      # test the whole loop with no hardware"
