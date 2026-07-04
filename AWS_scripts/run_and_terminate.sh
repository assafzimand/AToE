#!/usr/bin/env bash
set -e

# =============================================================================
# run_and_shutdown.sh - Run experiments and auto-shutdown EC2 instance
# =============================================================================
# This script:
#   1. Runs python run_experiments.py
#   2. Shuts down the EC2 instance (STOPS it to save money!)
#
# Your results are SAFE - they stay on the disk (EBS volume) and will be
# there when you restart the instance.
#
# Usage:
#   screen -S atoe_experiment
#   bash ~/AToE/AWS_scripts/run_and_terminate.sh [repo_dir]
#   # Detach with Ctrl+A, D - safe to disconnect!
#
# After experiments complete:
#   1. Restart instance from AWS Console
#   2. SSH in and download results with download_AWS_results.ps1
# =============================================================================

# Repo dir: first argument, else first existing of ~/AToE, ~/NCC-PINN
if [ -n "$1" ]; then
    REPO_DIR="$1"
elif [ -d "$HOME/AToE" ]; then
    REPO_DIR="$HOME/AToE"
else
    REPO_DIR="$HOME/NCC-PINN"
fi
REPO_NAME="$(basename "$REPO_DIR")"
VENV_DIR="$HOME/.venv_$(echo "$REPO_NAME" | tr '[:upper:]-' '[:lower:]_')"

# === Colors for output ===
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# === Helper functions ===
log_info() {
    echo -e "${CYAN}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# === Main script ===
echo "============================================================================="
echo "  $REPO_NAME: Run Experiments and Auto-Shutdown"
echo "============================================================================="
echo ""

# Activate virtual environment
log_info "Activating virtual environment..."
if [ ! -d "$VENV_DIR" ]; then
    log_error "Virtual environment not found at $VENV_DIR"
    log_error "Please run prepare_AWS_run.sh first!"
    exit 1
fi
source "$VENV_DIR/bin/activate"

# Change to repo directory
cd "$REPO_DIR"
log_info "Working directory: $(pwd)"
echo ""

# === Run experiments ===
echo "============================================================================="
log_info "Starting experiments..."
echo "============================================================================="
echo ""

# Set PyTorch CUDA memory allocator to use expandable segments
# This helps avoid memory fragmentation during training with high-order derivatives
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
log_info "CUDA memory allocator: expandable_segments enabled"
echo ""

EXPERIMENT_SUCCESS=true
python run_experiments.py || EXPERIMENT_SUCCESS=false

echo ""
if [ "$EXPERIMENT_SUCCESS" = true ]; then
    log_success "Experiments completed successfully!"
else
    log_warn "Experiments completed with errors (some may have failed)"
fi
echo ""

# === Show where results are saved ===
echo "============================================================================="
log_success "Results saved to:"
echo "============================================================================="
echo ""
log_info "  $REPO_DIR/outputs/"
log_info "  $REPO_DIR/checkpoints/"
echo ""
log_info "These files will be preserved when the instance stops."
log_info "To download: restart instance, SSH in, run download_AWS_results.ps1"
echo ""

# === Shutdown ===
echo "============================================================================="
log_warn "Instance will SHUT DOWN in 60 seconds to save costs!"
log_warn "Press Ctrl+C to cancel shutdown"
echo "============================================================================="
echo ""
log_info "After shutdown:"
log_info "  - Instance will be STOPPED (not terminated)"
log_info "  - No more compute charges (~\$0.53/hour saved)"
log_info "  - All results preserved on disk"
log_info "  - Restart anytime from AWS Console to download results"
echo ""

# Give user a chance to cancel
for i in {60..1}; do
    echo -ne "\r  Shutting down in $i seconds... (Ctrl+C to cancel)  "
    sleep 1
done
echo ""
echo ""

log_info "Initiating shutdown..."
sudo shutdown -h now
