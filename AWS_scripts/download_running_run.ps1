param(
    # Default EC2 public IP - override per run
    [string]$Ec2Ip = "16.171.19.234",
    [string]$PemPath = "$PSScriptRoot\..\..\NCC-PINN-ASSAF.pem",
    # Remote outputs root on EC2 (the AToE repo clone)
    [string]$RemoteRoot = "/home/ubuntu/AToE/outputs",
    # Local destination: the Desktop
    [string]$LocalTarget = "$env:USERPROFILE\Desktop"
)

# Downloads the CURRENTLY-RUNNING (or most recent) run directory:
# the newest <arch>/<timestamp>/ under outputs/ (excluding experiments/),
# i.e. the run that has not yet been moved into outputs/experiments.
# Read-only on the EC2 side; safe to use mid-run. Partial/in-progress
# files (logs, incremental metrics) come down as point-in-time snapshots.

Write-Host "=== Download the currently-running AToE run ===" -ForegroundColor Cyan
Write-Host ""
Write-Host "Current EC2 Public IP: $Ec2Ip"
$ipInput = Read-Host "Enter EC2 Public IP (or press Enter to keep current)"
if ($ipInput) { $Ec2Ip = $ipInput }
Write-Host ""

$ckptInput = Read-Host "Download checkpoints (.pt files)? They can be large [y/N]"
$excludePatterns = @('*.pdf')
if ($ckptInput -notmatch '^[Yy]') {
    $excludePatterns += '*.pt'
    Write-Host "  Skipping .pdf and .pt files" -ForegroundColor Gray
} else {
    Write-Host "  Skipping .pdf files (checkpoints included)" -ForegroundColor Gray
}
Write-Host ""

if (-not (Test-Path $PemPath)) {
    Write-Error "PEM file not found at '$PemPath'."
    exit 1
}

# Newest run dir under outputs/ (arch/timestamp), excluding experiments/
Write-Host "Querying EC2 for the newest run under outputs/<arch>/ ..." -ForegroundColor Cyan
$latestRun = (& ssh -i $PemPath ubuntu@$Ec2Ip "cd $RemoteRoot && ls -1td */*/ 2>/dev/null | grep -v '^experiments/' | head -1").Trim().TrimEnd('/')
if (-not $latestRun) {
    Write-Error "No run directories found under $RemoteRoot (outside experiments/)."
    exit 1
}
Write-Host "Newest run: $latestRun" -ForegroundColor Cyan
Write-Host "Local destination: $LocalTarget\$latestRun" -ForegroundColor Cyan
Write-Host ""

# tar over ssh via cmd.exe (raw byte pipe; PowerShell pipes corrupt binary).
# Downloading the <arch>/<timestamp> path preserves both folder levels.
$excludeArgs = ($excludePatterns | ForEach-Object { "--exclude='$_'" }) -join ' '
$remoteCmd = "cd $RemoteRoot && tar cf - $excludeArgs '$latestRun'"
$cmdLine = "ssh -i `"$PemPath`" ubuntu@$Ec2Ip `"$remoteCmd`" | tar -xvf - -C `"$LocalTarget`""
$batchFile = Join-Path $env:TEMP "atoe_dl_running_$PID.cmd"
"@echo off`r`n$cmdLine" | Set-Content -Path $batchFile -Encoding ASCII
& cmd.exe /c $batchFile
$exitCode = $LASTEXITCODE
Remove-Item $batchFile -Force -ErrorAction SilentlyContinue

if ($exitCode -eq 0) {
    Write-Host ""
    Write-Host "Download complete: $LocalTarget\$latestRun" -ForegroundColor Green
} else {
    Write-Host ""
    Write-Host "tar/ssh exited non-zero ($exitCode). Mid-run this is often just" -ForegroundColor Yellow
    Write-Host "'file changed as we read it' on the live log - files usually landed fine." -ForegroundColor Yellow
}
