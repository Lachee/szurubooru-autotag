#!/usr/bin/env bash
set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

info()    { echo -e "${GREEN}[setup]${NC} $*"; }
warn()    { echo -e "${YELLOW}[setup]${NC} $*"; }
die()     { echo -e "${RED}[setup] ERROR:${NC} $*" >&2; exit 1; }

# --- Check Python ---
if ! command -v python3 &>/dev/null; then
    die "python3 is not installed. Install Python 3.10+ and re-run."
fi

PYTHON_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
PYTHON_MAJOR=$(python3 -c 'import sys; print(sys.version_info.major)')
PYTHON_MINOR=$(python3 -c 'import sys; print(sys.version_info.minor)')

info "Found Python $PYTHON_VERSION"

if [[ "$PYTHON_MAJOR" -lt 3 || ("$PYTHON_MAJOR" -eq 3 && "$PYTHON_MINOR" -lt 10) ]]; then
    die "Python 3.10+ is required (found $PYTHON_VERSION)."
fi

# --- Check pip ---
if ! python3 -m pip --version &>/dev/null; then
    die "pip is not available. Install pip and re-run."
fi
info "pip is available"

# --- Check huggingface-cli (hf) ---
if ! command -v hf &>/dev/null; then
    die "huggingface-cli not found — installing huggingface_hub[cli]..."
    curl -LsSf https://hf.co/cli/install.sh | bash
else
    HF_CMD="huggingface-cli"
    info "huggingface-cli is available"
fi

# --- Download taggerine model ---
TAGGERINE_DIR="$(dirname "$0")/taggerine"

if [[ -d "$TAGGERINE_DIR" && -n "$(ls -A "$TAGGERINE_DIR" 2>/dev/null)" ]]; then
    warn "taggerine/ directory already exists and is non-empty — skipping download."
    warn "Delete or empty the taggerine/ directory to re-download."
else
    info "Downloading lodestones/taggerine into taggerine/..."
    $HF_CMD download lodestones/taggerine --exclude tagger_ui --local-dir "$TAGGERINE_DIR"
    info "Download complete."
fi

# --- Install taggerine requirements ---
TAGGERINE_REQS="$TAGGERINE_DIR/requirements.txt"

if [[ -f "$TAGGERINE_REQS" ]]; then
    info "Installing taggerine/requirements.txt..."
    python3 -m pip install -r "$TAGGERINE_REQS"
else
    warn "taggerine/requirements.txt not found — skipping."
fi

# --- Install project requirements ---
PROJECT_REQS="$(dirname "$0")/requirements.txt"

if [[ -f "$PROJECT_REQS" && -s "$PROJECT_REQS" ]]; then
    info "Installing requirements.txt..."
    python3 -m pip install -r "$PROJECT_REQS"
else
    warn "requirements.txt is missing or empty — skipping."
fi

info "Setup complete."
