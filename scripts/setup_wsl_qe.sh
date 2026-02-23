#!/usr/bin/env bash
# =============================================================================
# SHALOM — WSL2 Ubuntu QE Setup Script
# Run this INSIDE WSL Ubuntu:
#   wsl -d Ubuntu-22.04
#   bash /mnt/c/.../SHALOM/scripts/setup_wsl_qe.sh
#
# The script auto-detects SHALOM_ROOT from its own location.
# =============================================================================
set -e

# Auto-detect SHALOM root (parent of scripts/)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SHALOM_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PSEUDO_DIR="${SHALOM_PSEUDO_DIR:-$HOME/pseudopotentials}"

echo "============================================================"
echo "  SHALOM — QE Environment Setup for WSL2 Ubuntu"
echo "============================================================"
echo ""
echo "  SHALOM_ROOT:  $SHALOM_ROOT"
echo "  PSEUDO_DIR:   $PSEUDO_DIR"
echo ""

# ---------------------------------------------------------------------------
# 1. System update + QE install
# ---------------------------------------------------------------------------
echo "[1/5] Updating package list and installing Quantum ESPRESSO..."
sudo apt-get update -qq
sudo apt-get install -y quantum-espresso

echo ""
echo "  pw.x version: $(pw.x --version 2>&1 | head -1 || echo 'check manually')"
echo ""

# ---------------------------------------------------------------------------
# 2. Miniconda (if not present)
# ---------------------------------------------------------------------------
echo "[2/5] Checking Miniconda..."
if command -v conda &>/dev/null; then
    echo "  conda already installed: $(conda --version)"
else
    echo "  Downloading Miniconda..."
    wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh
    bash /tmp/miniconda.sh -b -p "$HOME/miniconda3"
    eval "$($HOME/miniconda3/bin/conda shell.bash hook)"
    conda init bash
    echo "  Miniconda installed. Reloading shell..."
    source ~/.bashrc || true
fi

# Ensure conda is on PATH for this script
if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
fi

echo ""

# ---------------------------------------------------------------------------
# 3. Create shalom-env from environment.yml
# ---------------------------------------------------------------------------
echo "[3/5] Setting up shalom-env (Python 3.11)..."
if conda env list | grep -q "^shalom-env"; then
    echo "  shalom-env already exists, updating..."
    conda env update -f "$SHALOM_ROOT/environment.yml" -n shalom-env --prune
else
    conda env create -f "$SHALOM_ROOT/environment.yml"
fi

echo ""

# ---------------------------------------------------------------------------
# 4. Download Si pseudopotential for test
# ---------------------------------------------------------------------------
echo "[4/5] Downloading Si pseudopotential..."
mkdir -p "$PSEUDO_DIR"
SI_UPF="Si.pbe-n-rrkjus_psl.1.0.0.UPF"
SI_URL="https://pseudopotentials.quantum-espresso.org/upf_files/$SI_UPF"

if [ -f "$PSEUDO_DIR/$SI_UPF" ]; then
    echo "  Si UPF already present."
else
    wget -q "$SI_URL" -O "$PSEUDO_DIR/$SI_UPF"
    echo "  Downloaded: $SI_UPF"
fi

echo ""
echo "  SHALOM_PSEUDO_DIR=$PSEUDO_DIR"

# ---------------------------------------------------------------------------
# 5. Add env vars to ~/.bashrc
# ---------------------------------------------------------------------------
echo "[5/5] Configuring environment variables in ~/.bashrc..."

add_if_missing() {
    local line="$1"
    grep -qxF "$line" ~/.bashrc || echo "$line" >> ~/.bashrc
}

add_if_missing "export SHALOM_PSEUDO_DIR=$PSEUDO_DIR"
add_if_missing "export SHALOM_WORKSPACE=~/Desktop/shalom-runs"

echo "  Added to ~/.bashrc:"
echo "    export SHALOM_PSEUDO_DIR=$PSEUDO_DIR"
echo "    export SHALOM_WORKSPACE=~/Desktop/shalom-runs"

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
echo ""
echo "============================================================"
echo "  Setup complete! Run the following to test:"
echo "============================================================"
echo ""
echo "  conda activate shalom-env"
echo "  export SHALOM_PSEUDO_DIR=$PSEUDO_DIR"
echo "  cd $SHALOM_ROOT"
echo ""
echo "  # Check QE environment"
echo "  python -m shalom setup-qe --elements Si"
echo ""
echo "  # Run Si SCF calculation"
echo "  python -m shalom run Si --backend qe --calc scf --execute"
echo ""
echo "  Output will be in: ~/Desktop/shalom-runs/"
echo "============================================================"
