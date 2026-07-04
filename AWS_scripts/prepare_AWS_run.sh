#!/usr/bin/env bash
set -e

# Helper script to prepare an EC2 GPU instance for running AToE / NCC-PINN.
# Run this *on the EC2 machine* after you SSH in.
#
# Usage:
#   bash ~/AToE/AWS_scripts/prepare_AWS_run.sh
#   # or if copied elsewhere:
#   bash prepare_AWS_run.sh

# === Repo selection ===
GITHUB_USER="assafzimand"
REPOS=("AToE" "NCC-PINN")

echo "=== Select repository ==="
for i in "${!REPOS[@]}"; do
  if [ "$i" -eq 0 ]; then
    echo "  $((i+1))) ${REPOS[$i]} (default)"
  else
    echo "  $((i+1))) ${REPOS[$i]}"
  fi
done

echo
read -p "Enter repo number [1]: " REPO_CHOICE
if [ -z "$REPO_CHOICE" ]; then
  REPO_CHOICE=1
fi
if ! [[ "$REPO_CHOICE" =~ ^[0-9]+$ ]] || [ "$REPO_CHOICE" -lt 1 ] || [ "$REPO_CHOICE" -gt ${#REPOS[@]} ]; then
  echo "Invalid choice. Using repo: ${REPOS[0]}"
  REPO_CHOICE=1
fi
SELECTED_REPO="${REPOS[$((REPO_CHOICE-1))]}"
echo
echo "Selected repo: $SELECTED_REPO"

# === Configuration ===
REPO_URL="https://github.com/${GITHUB_USER}/${SELECTED_REPO}.git"
REPO_DIR="$HOME/${SELECTED_REPO}"
VENV_DIR="$HOME/.venv_$(echo "$SELECTED_REPO" | tr '[:upper:]-' '[:lower:]_')"

echo "=== Updating apt and installing dependencies (python3, venv, git, git-lfs, screen) ==="
sudo apt update
sudo apt install -y python3 python3-venv git git-lfs screen

echo
echo "=== Initializing Git LFS ==="
git lfs install

echo
echo "=== Creating Python virtual environment (if missing) ==="
if [ ! -d "$VENV_DIR" ]; then
  python3 -m venv "$VENV_DIR"
fi

echo "=== Activating virtual environment ==="
# shellcheck disable=SC1090
source "$VENV_DIR/bin/activate"

echo
echo "=== Cloning or updating $SELECTED_REPO repo ==="
if [ ! -d "$REPO_DIR" ]; then
  git clone "$REPO_URL" "$REPO_DIR"
fi

cd "$REPO_DIR"

echo "  Fetching latest changes from GitHub..."
git fetch origin --prune

echo
echo "=== Select branch to use ==="
# Get list of remote branches (excluding HEAD)
BRANCHES=($(git branch -r | grep -v HEAD | sed 's/origin\///' | tr -d ' '))

if [ ${#BRANCHES[@]} -eq 0 ]; then
  echo "ERROR: No branches found!"
  exit 1
fi

echo "Available branches:"
for i in "${!BRANCHES[@]}"; do
  # Mark main/master as default
  if [[ "${BRANCHES[$i]}" == "main" ]] || [[ "${BRANCHES[$i]}" == "master" ]]; then
    echo "  $((i+1))) ${BRANCHES[$i]} (default)"
  else
    echo "  $((i+1))) ${BRANCHES[$i]}"
  fi
done

echo
read -p "Enter branch number [1]: " BRANCH_CHOICE

# Default to first branch if no input
if [ -z "$BRANCH_CHOICE" ]; then
  BRANCH_CHOICE=1
fi

# Validate input
if ! [[ "$BRANCH_CHOICE" =~ ^[0-9]+$ ]] || [ "$BRANCH_CHOICE" -lt 1 ] || [ "$BRANCH_CHOICE" -gt ${#BRANCHES[@]} ]; then
  echo "Invalid choice. Using branch: ${BRANCHES[0]}"
  BRANCH_CHOICE=1
fi

SELECTED_BRANCH="${BRANCHES[$((BRANCH_CHOICE-1))]}"
echo
echo "Selected branch: $SELECTED_BRANCH"

echo "  Force updating to match origin/$SELECTED_BRANCH (discards local changes)..."
git checkout -B "$SELECTED_BRANCH" "origin/$SELECTED_BRANCH"
git reset --hard "origin/$SELECTED_BRANCH"

echo
echo "=== Clearing Python cache ==="
echo "  Removing __pycache__ directories and .pyc files..."
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find . -name "*.pyc" -delete 2>/dev/null || true
echo "  Cache cleared"

echo
echo "=== Installing Python dependencies ==="
pip install --upgrade pip
if [ -f "requirements.txt" ]; then
  echo "  Installing from requirements.txt..."
  pip install -r requirements.txt
  
  echo
  echo "=== Verifying critical dependencies ==="
  python3 -c "import torch; print(f'  ✓ PyTorch {torch.__version__}')" || echo "  ✗ PyTorch not found!"
  python3 -c "from scimba_torch.optimizers.ssbroyden import SSBroyden; print('  ✓ scimba SSBroyden available')" || echo "  ✗ scimba SSBroyden not found!"
  python3 -c "import scipy; print('  ✓ scipy installed')" || echo "  ✗ scipy not found!"
else
  echo "WARNING: requirements.txt not found in $REPO_DIR"
fi

echo
echo "=== Environment ready ==="
echo "To start working on this instance next time, run:"
echo "  source $VENV_DIR/bin/activate"
echo "  cd $REPO_DIR"
echo
echo "To launch experiments with screen (recommended for long runs):"
echo "  screen -S ncc_experiment"
echo "  source $VENV_DIR/bin/activate"
echo "  cd $REPO_DIR"
echo "  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
echo "  python run_experiments.py"
echo "  # Press Ctrl+A then D to detach and disconnect safely"
echo
echo "To reattach to a running screen session:"
echo "  screen -r ncc_experiment"
echo
echo "To list all screen sessions:"
echo "  screen -ls"
echo
echo "Or use run_and_terminate.sh (includes auto-shutdown):"
echo "  bash $REPO_DIR/AWS_scripts/run_and_terminate.sh"
echo
echo "Or run directly without screen (will stop if you disconnect):"
echo "  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
echo "  python run_experiments.py"


