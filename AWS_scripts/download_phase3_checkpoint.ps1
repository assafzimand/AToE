param(
    # Default EC2 public IP - you can override on each run
    [string]$Ec2Ip = "13.60.229.209",
    # Path to your SSH key (relative to repo root by default)
    [string]$PemPath = "$PSScriptRoot\..\..\NCC-PINN-ASSAF.pem",
    # Experiments root (where run_experiments.py writes)
    [string]$ExperimentsRoot = "/home/ubuntu/AToE/outputs/experiments",
    # The specific experiment folder to pull checkpoints from
    [string]$ExperimentName = "allen_cahn_M8_owner_imitator_20260712_214447",
    # Which checkpoint file to download from each run's checkpoints/ dir
    [string]$CheckpointName = "best_model_phase3.pt",
    # Local experiment folder to mirror the checkpoints into
    [string]$LocalTarget = "$env:USERPROFILE\Desktop\allen_cahn_M8_owner_imitator_20260712_214447"
)

Write-Host "=== Download $CheckpointName from $ExperimentName ===" -ForegroundColor Cyan
Write-Host ""

Write-Host "Current EC2 Public IP: $Ec2Ip"
$ipInput = Read-Host "Enter EC2 Public IP (or press Enter to keep current)"
if ($ipInput) { $Ec2Ip = $ipInput }
Write-Host ""

if (-not (Test-Path $PemPath)) {
    Write-Error "PEM file not found at '$PemPath'. Edit PemPath in this script or pass it as a parameter."
    exit 1
}

$remoteExpPath = "$ExperimentsRoot/$ExperimentName"

# Locate every copy of the checkpoint inside the experiment folder
# (one per run dir: OI1 / OI2 / OI3)
Write-Host "Locating '$CheckpointName' under $remoteExpPath ..." -ForegroundColor Cyan
$found = (& ssh -i $PemPath ubuntu@$Ec2Ip "find $remoteExpPath -name '$CheckpointName' -printf '%p %s\n'").Trim()
if (-not $found) {
    Write-Error "No '$CheckpointName' found under '$remoteExpPath' on EC2."
    exit 1
}

$entries = $found -split "`n" | Where-Object { $_ }
Write-Host "  Found $($entries.Count) checkpoint(s):" -ForegroundColor Cyan
foreach ($entry in $entries) {
    $p, $bytes = $entry -split ' '
    Write-Host ("    - {0}  ({1:N1} MB)" -f $p, ($bytes / 1MB)) -ForegroundColor Gray
}
Write-Host ""

$ok = 0
foreach ($entry in $entries) {
    $remoteFile = ($entry -split ' ')[0]

    # Mirror the path relative to the experiment folder into $LocalTarget,
    # e.g. <arch>/<timestamp>/checkpoints/best_model_phase3.pt
    $relPath = $remoteFile.Substring($remoteExpPath.Length).TrimStart('/')
    $localFile = Join-Path $LocalTarget ($relPath -replace '/', '\')
    $localDir = Split-Path $localFile -Parent
    New-Item -ItemType Directory -Force -Path $localDir | Out-Null

    Write-Host "Downloading: $relPath" -ForegroundColor Yellow
    & scp -i $PemPath "ubuntu@${Ec2Ip}:$remoteFile" $localFile
    if ($LASTEXITCODE -eq 0) {
        Write-Host "  -> $localFile" -ForegroundColor Green
        $ok++
    } else {
        Write-Host "  Failed (scp exit code $LASTEXITCODE)." -ForegroundColor Red
    }
}

Write-Host ""
if ($ok -eq $entries.Count) {
    Write-Host "Download complete ($ok/$($entries.Count))." -ForegroundColor Green
} else {
    Write-Error "Downloaded $ok of $($entries.Count) checkpoints."
}
