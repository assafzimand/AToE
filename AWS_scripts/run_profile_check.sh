#!/usr/bin/env bash
set -e

# =============================================================================
# run_profile_check.sh - short GPU profiling smoke, then print the reports
# =============================================================================
# Runs experiment_plans/profile_soap_gpu.yaml: ONE burgers run, all three
# phases (root -> phase 3 -> fine_tune), 500 epochs each, at the same scale as
# the real campaign. Each segment records a 50-step torch.profiler window and
# writes profile_<segment>.txt next to metrics.json.
#
# Unlike run_and_terminate.sh this does NOT shut the instance down -- the whole
# point is to read the reports afterwards.
#
# Usage:
#   bash ~/AToE/AWS_scripts/run_profile_check.sh [repo_dir]
# =============================================================================

PLAN="experiment_plans/profile_soap_gpu.yaml"

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

GREEN='\033[0;32m'; YELLOW='\033[1;33m'; CYAN='\033[0;36m'; RED='\033[0;31m'; NC='\033[0m'
log_info()    { echo -e "${CYAN}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
log_warn()    { echo -e "${YELLOW}[WARNING]${NC} $1"; }
log_error()   { echo -e "${RED}[ERROR]${NC} $1"; }

echo "============================================================================="
echo "  $REPO_NAME: GPU profiling check (short run, no shutdown)"
echo "============================================================================="
echo ""

if [ ! -d "$VENV_DIR" ]; then
    log_error "Virtual environment not found at $VENV_DIR"
    log_error "Run prepare_AWS_run.sh first."
    exit 1
fi
source "$VENV_DIR/bin/activate"
cd "$REPO_DIR"
log_info "Working directory: $(pwd)"

if [ ! -f "$PLAN" ]; then
    log_error "Plan not found: $PLAN"
    exit 1
fi

# Confirm the GPU is actually visible -- if this says False, the profile will
# be CPU-only and the whole exercise measures the wrong machine.
log_info "Checking CUDA..."
python - <<'PY'
import torch
print(f"  torch            : {torch.__version__}")
print(f"  cuda available   : {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  device           : {torch.cuda.get_device_name(0)}")
    print(f"  capability       : {torch.cuda.get_device_capability(0)}")
else:
    print("  !! CUDA NOT AVAILABLE - the profile will be CPU-only !!")
PY
echo ""

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
log_info "CUDA allocator: expandable_segments enabled"
echo ""

log_info "Running $PLAN ..."
echo "============================================================================="
RUN_OK=true
python run_experiments.py "$PLAN" || RUN_OK=false
echo "============================================================================="
echo ""

if [ "$RUN_OK" = true ]; then
    log_success "Run finished."
else
    log_warn "Run reported errors - profile reports may still exist below."
fi
echo ""

# === Print every profile report produced by the newest experiment dir ===
LATEST="$(ls -1dt outputs/experiments/profile_soap_gpu_check_* 2>/dev/null | head -1)"
if [ -z "$LATEST" ]; then
    log_warn "No profile_soap_gpu_check_* output directory found."
    exit 0
fi
log_info "Results: $LATEST"
echo ""

FOUND=0
while IFS= read -r f; do
    FOUND=$((FOUND+1))
    echo "============================================================================="
    echo "  $f"
    echo "============================================================================="
    cat "$f"
    echo ""
done < <(find "$LATEST" -name 'profile_*.txt' | sort)

if [ "$FOUND" -eq 0 ]; then
    log_warn "No profile_*.txt files were written."
    log_warn "Check the log for '[Profiler]' lines - it disables itself on error."
    log_warn "grep -n 'Profiler' \$(find $LATEST -name training_logs.log)"
else
    log_success "$FOUND profile report(s) printed above."
    log_info "Read the 'Self CUDA %' column; absolute times in a profiled window are inflated."
fi
echo ""
log_info "Instance left RUNNING (no auto-shutdown). Stop it from the console when done."
