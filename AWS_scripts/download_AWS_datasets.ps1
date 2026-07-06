param(
    [string]$Ec2Ip = "13.61.24.69",
    [string]$PemPath = "$PSScriptRoot\..\..\NCC-PINN-ASSAF.pem",
    [string]$RemoteRoot = "/home/ubuntu/AToE/datasets",
    [string]$LocalTarget = "$PSScriptRoot\..\datasets"
)

Write-Host "=== Download AToE datasets from AWS EC2 ===" -ForegroundColor Cyan
Write-Host ""

Write-Host "Current EC2 Public IP: $Ec2Ip"
$ipInput = Read-Host "Enter EC2 Public IP (or press Enter to keep current)"
if ($ipInput) { $Ec2Ip = $ipInput }
Write-Host ""

if (-not (Test-Path $PemPath)) {
    Write-Error "PEM file not found at '$PemPath'. Edit PemPath in this script or pass it as a parameter."
    exit 1
}

Write-Host "Querying EC2 for available datasets ..." -ForegroundColor Cyan
try {
    $folders = (& ssh -i $PemPath ubuntu@$Ec2Ip "ls -1d $RemoteRoot/*/").Trim()
    $folderList = $folders -split "`n" | Where-Object { $_ }
} catch {
    Write-Error "Failed to query EC2 via ssh."
    exit 1
}

Write-Host "  Found datasets:" -ForegroundColor Cyan
foreach ($f in $folderList) {
    $name = ($f -replace '.*/', '').TrimEnd('/')
    Write-Host "    - $name" -ForegroundColor Gray
}
Write-Host ""

Write-Host "Remote path:" -ForegroundColor Cyan
Write-Host "  ubuntu@${Ec2Ip}:${RemoteRoot}"
Write-Host "Local destination:" -ForegroundColor Cyan
Write-Host "  $LocalTarget"
Write-Host ""

New-Item -ItemType Directory -Force -Path $LocalTarget | Out-Null

$scpCmd = "scp -i `"$PemPath`" -r ubuntu@${Ec2Ip}:`"$RemoteRoot`" `"$LocalTarget\..`""

Write-Host "Running:" -ForegroundColor Yellow
Write-Host "  $scpCmd"
Write-Host ""

try {
    Invoke-Expression $scpCmd
    Write-Host ""
    Write-Host "Download complete." -ForegroundColor Green
}
catch {
    Write-Error "scp failed: $_"
}
