# AWS Scripts Usage

This folder contains helper scripts to run NCC-PINN experiments on an AWS EC2 GPU instance with **automatic shutdown** to save money.

---

## Quick Start

1. **SSH into your EC2 instance**
2. **First time only**: Run setup script
   ```bash
   bash ~/NCC-PINN/AWS_scripts/prepare_AWS_run.sh
   ```
3. **Activate venv** (every time you connect):
   ```bash
   source ~/.venv_ncc_pinn/bin/activate
   cd ~/NCC-PINN
   ```
4. **Run experiments with auto-shutdown**:
   ```bash
   screen -S ncc_experiment
   bash ~/NCC-PINN/AWS_scripts/run_and_terminate.sh
   # Detach: Ctrl+A, then D
   ```
5. **After experiments complete**: Instance auto-stops → no more charges!
6. **Download results**: Restart instance, SSH in, activate venv, run download script

---

## Scripts Overview

| Script | Purpose | Run On |
|--------|---------|--------|
| `prepare_AWS_run.sh` | Setup EC2 environment (Python, venv, repo) | EC2 |
| `run_and_terminate.sh` | Run experiments + auto-shutdown | EC2 |
| `download_AWS_results.ps1` | Download results from EC2 via SSH | Windows |

---

## How It Works

```
┌─────────────────────────────────────────────────────────────────┐
│ 1. SSH into EC2                                                 │
│    ssh -i .\NCC-PINN-ASSAF.pem ubuntu@<IP>                     │
├─────────────────────────────────────────────────────────────────┤
│ 2. Run experiments with auto-shutdown                           │
│    screen -S ncc_experiment                                     │
│    bash ~/NCC-PINN/AWS_scripts/run_and_terminate.sh            │
│    (Ctrl+A, D to detach - safe to disconnect!)                 │
├─────────────────────────────────────────────────────────────────┤
│ 3. Instance runs experiments...                                 │
│    → Experiments complete                                       │
│    → Instance STOPS automatically (saves ~$0.53/hour!)         │
│    → All results preserved on disk                             │
├─────────────────────────────────────────────────────────────────┤
│ 4. Later: Restart instance from AWS Console                     │
│    → Note the new IP address                                   │
│    → SSH in and download results                               │
│    → Stop instance again when done                             │
└─────────────────────────────────────────────────────────────────┘
```

---

## Data Safety

When instance **STOPS** (not terminate):
- ✅ All files on disk are preserved
- ✅ `outputs/` with all results → safe
- ✅ `checkpoints/` → safe
- ✅ Your venv and repo → safe
- 💰 Compute charges stop (~$0.53/hour saved)
- 💵 Only pay small storage cost (~$4/month for 50GB)

**Tip**: Enable **Termination Protection** in AWS Console to prevent accidental deletion:
- EC2 Console → Select instance → Actions → Instance settings → Change termination protection → Enable

---

## Creating a New EC2 Instance

If your instance was terminated or you need a new one:

### Step 1: Launch Instance

1. Go to [EC2 Console](https://console.aws.amazon.com/ec2/) → **"Launch instance"**

2. **Name**: `NCC-PINN-GPU`

3. **Application and OS Images (AMI)**:
   - Search for: `Deep Learning AMI GPU PyTorch`
   - Select: **Deep Learning AMI GPU PyTorch 2.x (Ubuntu 20.04)**

4. **Instance type**: `g4dn.xlarge` (~$0.53/hour with GPU)

5. **Key pair**: Select `NCC-PINN-ASSAF` (or create new)

6. **Network settings**: Allow SSH from your IP

7. **Storage**: 50 GB gp3

8. Click **"Launch instance"**

### Step 2: Enable Termination Protection (Recommended)

1. Select instance → **Actions** → **Instance settings** → **Change termination protection**
2. Check **Enable** → Save

### Step 3: Note the Public IP

Copy the **Public IPv4 address** from EC2 Console.

### Instance Type Reference

| Type | GPU | Cost/hour | Best For |
|------|-----|-----------|----------|
| `t3.medium` | None | ~$0.04 | Testing |
| `g4dn.xlarge` | T4 (16GB) | ~$0.53 | Training (recommended) |
| `g4dn.2xlarge` | T4 (16GB) | ~$0.75 | Larger batches |

---

## Script Details

### 1. `prepare_AWS_run.sh` – Setup on EC2

Run this **the first time** after SSHing into EC2:

```bash
bash ~/NCC-PINN/AWS_scripts/prepare_AWS_run.sh
```

This will:
- Install Python, venv, git, screen
- Create virtualenv at `~/.venv_ncc_pinn`
- Clone/update the NCC-PINN repo from GitHub
- **Install all dependencies** from `requirements.txt` (including `torchmin` for SSBroyden optimizer)

**Important**: The script activates the venv **during setup only**. After the script finishes, you need to **manually activate** the venv:

```bash
source ~/.venv_ncc_pinn/bin/activate
cd ~/NCC-PINN
```

**You'll see `(.venv_ncc_pinn)` in your prompt when activated.**

### 2. `run_and_terminate.sh` – Run Experiments with Auto-Shutdown

```bash
screen -S ncc_experiment
bash ~/NCC-PINN/AWS_scripts/run_and_terminate.sh
# Detach: Ctrl+A, then D
```

The script will:
1. Run `python run_experiments.py`
2. Wait 60 seconds (cancel with Ctrl+C if needed)
3. Shutdown the instance (STOP, not terminate)

### 3. `download_AWS_results.ps1` – Download Results

Run this on your Windows PC after restarting the instance:

```powershell
cd C:\Users\assaf\Desktop\Coding\Msc\Master\NCC-PINN
.\AWS_scripts\download_AWS_results.ps1
```

**Note**: You'll need to enter the new IP address (it changes after restart).

---

## Complete Workflow Example

### Morning: Start Experiments

```powershell
# On Windows - SSH into EC2
cd C:\Users\assaf\Desktop\Coding\Msc\Master
ssh -i .\NCC-PINN-ASSAF.pem ubuntu@13.60.229.209
```

```bash
# On EC2 - First time setup (if needed)
bash ~/NCC-PINN/AWS_scripts/prepare_AWS_run.sh

# Activate venv (REQUIRED every time you reconnect!)
source ~/.venv_ncc_pinn/bin/activate
cd ~/NCC-PINN

# Run experiments in screen (safe to disconnect)
screen -S ncc_experiment
bash ~/NCC-PINN/AWS_scripts/run_and_terminate.sh
# Press Ctrl+A, then D to detach
exit  # Disconnect from SSH - experiments continue!
```

**Note**: The venv activation step is **mandatory** every time you SSH in. You'll see `(.venv_ncc_pinn)` in your prompt when it's active.

### Later: Instance Stopped Automatically
- Go to AWS Console to verify instance is "Stopped"
- No action needed - you're saving money!

### When Ready: Download Results

1. **Restart instance**: EC2 Console → Select instance → Instance state → Start
2. **Note new IP**: Check "Public IPv4 address" in console
3. **Download**:
   ```powershell
   cd C:\Users\assaf\Desktop\Coding\Msc\Master\NCC-PINN
   .\AWS_scripts\download_AWS_results.ps1
   # Enter the new IP when prompted
   ```
4. **Stop instance**: EC2 Console → Instance state → Stop

---

## Screen Commands Reference

| Task | Command |
|------|---------|
| Start a new screen session | `screen -S ncc_experiment` |
| Detach (keeps running) | `Ctrl+A`, then `D` |
| List all sessions | `screen -ls` |
| Reattach to session | `screen -r ncc_experiment` |
| Stop running program | `Ctrl+C` (while attached) |
| Exit screen session | `exit` (while attached) |

---

## Cost Summary

| State | Compute | Storage | Total |
|-------|---------|---------|-------|
| Running (g4dn.xlarge) | $0.53/hour | ~$0.005/hour | **$0.54/hour** |
| Stopped | $0.00 | ~$0.005/hour | **~$4/month** |

**Auto-shutdown prevents forgetting to stop → saves $12+ per forgotten day!**
