param(
    # Default EC2 public IP - you can override on each run
    [string]$Ec2Ip = "13.60.229.209",
    # Path to your SSH key (relative to repo root by default)
    # Repo structure: Master\NCC-PINN-ASSAF.pem and Master\AToE\AWS_scripts\this_file
    # So from $PSScriptRoot (AToE\AWS_scripts) we need to go up two levels.
    [string]$PemPath = "$PSScriptRoot\..\..\NCC-PINN-ASSAF.pem",
    # Remote outputs root on EC2 (the AToE repo clone)
    [string]$RemoteRoot = "/home/ubuntu/AToE/outputs",
    # Experiments root (where run_experiments.py writes)
    [string]$ExperimentsRoot = "/home/ubuntu/AToE/outputs/experiments",
    # Local folder where results will be downloaded
    [string]$LocalTarget = "$PSScriptRoot\aws_outputs",
    # Default number of model folders for the failed-experiment fallback prompt
    [int]$defaultModels = 1
)

Write-Host "=== Download AToE outputs from AWS EC2 ===" -ForegroundColor Cyan
Write-Host ""

Write-Host "Current EC2 Public IP: $Ec2Ip"
$ipInput = Read-Host "Enter EC2 Public IP (or press Enter to keep current)"
if ($ipInput) { $Ec2Ip = $ipInput }
Write-Host ""

# PDFs are never downloaded (huge, and every plot also exists as .png).
# Checkpoints (.pt) are optional - they are by far the largest files.
$ckptInput = Read-Host "Download checkpoints (.pt files)? They can be multiple GB each [y/N]"
$excludePatterns = @('*.pdf')
if ($ckptInput -notmatch '^[Yy]') {
    $excludePatterns += '*.pt'
    Write-Host "  Skipping .pdf and .pt files" -ForegroundColor Gray
} else {
    Write-Host "  Skipping .pdf files (checkpoints included)" -ForegroundColor Gray
}
Write-Host ""

# Downloads a remote folder as a tar stream over ssh, honoring exclude patterns.
# scp -r cannot exclude files, and PowerShell pipes corrupt binary data, so the
# ssh | tar pipeline runs inside cmd.exe (raw byte pipe). tar.exe ships with Windows.
function Download-RemoteFolder {
    param(
        [string]$RemoteParent,  # remote dir containing the folder
        [string]$FolderName,    # folder to download, relative to RemoteParent
        [string]$Destination    # local target dir (must exist)
    )
    $excludeArgs = ($excludePatterns | ForEach-Object { "--exclude='$_'" }) -join ' '
    $remoteCmd = "cd $RemoteParent && tar cf - $excludeArgs '$FolderName'"
    $cmdLine = "ssh -i `"$PemPath`" ubuntu@$Ec2Ip `"$remoteCmd`" | tar -xvf - -C `"$Destination`""
    $batchFile = Join-Path $env:TEMP "atoe_download_$PID.cmd"
    "@echo off`r`n$cmdLine" | Set-Content -Path $batchFile -Encoding ASCII
    & cmd.exe /c $batchFile
    $exitCode = $LASTEXITCODE
    Remove-Item $batchFile -Force -ErrorAction SilentlyContinue
    return ($exitCode -eq 0)
}

if (-not (Test-Path $PemPath)) {
    Write-Error "PEM file not found at '$PemPath'. Edit PemPath in this script or pass it as a parameter."
    exit 1
}

# Check if any screen sessions are running (optional info)
Write-Host "Checking for active screen sessions on EC2..." -ForegroundColor Cyan
try {
    $screenSessions = (& ssh -i $PemPath ubuntu@$Ec2Ip "screen -ls 2>&1").Trim()
    if ($screenSessions -match "atoe_experiment|ncc_experiment") {
        Write-Host "  ⚠ Active screen session detected: experiments may still be running!" -ForegroundColor Yellow
        Write-Host "  Tip: SSH in and run 'screen -r atoe_experiment' to check progress" -ForegroundColor Yellow
    } else {
        Write-Host "  No active screen sessions found" -ForegroundColor Gray
    }
} catch {
    Write-Host "  (Could not check screen sessions)" -ForegroundColor Gray
}
Write-Host ""

# Find the most recently modified experiment under outputs/experiments on EC2
Write-Host "Querying EC2 for latest experiment under outputs/experiments/ ..." -ForegroundColor Cyan
try {
    # This assumes OpenSSH client is installed on Windows
    $lastFolder = (& ssh -i $PemPath ubuntu@$Ec2Ip "cd $ExperimentsRoot && ls -1t | head -1").Trim()
} catch {
    Write-Error "Failed to query EC2 via ssh. Make sure ssh is installed and the IP/key are correct."
    exit 1
}

if (-not $lastFolder) {
    Write-Error "Could not determine latest folder under '$ExperimentsRoot' on EC2."
    exit 1
}

$remotePath = "$ExperimentsRoot/$lastFolder"

Write-Host "Latest experiment folder detected:" -ForegroundColor Cyan
Write-Host "  $lastFolder"
Write-Host ""

# Check if the folder only contains experiments_plan.yaml (failed experiment)
Write-Host "Checking folder contents..." -ForegroundColor Cyan
try {
    $folderContents = (& ssh -i $PemPath ubuntu@$Ec2Ip "ls -1 $remotePath").Trim()
    $fileCount = ($folderContents -split "`n").Count
    $onlyPlanFile = ($fileCount -eq 1) -and ($folderContents -match "experiments_plan.yaml")
} catch {
    $onlyPlanFile = $false
}

if ($onlyPlanFile) {
    Write-Host "  ⚠ Experiment folder only contains experiments_plan.yaml!" -ForegroundColor Yellow
    Write-Host "  This usually means training failed before completion." -ForegroundColor Yellow
    Write-Host ""
    Write-Host "  Individual model outputs are saved directly in outputs/ folder." -ForegroundColor Cyan
    Write-Host ""
    
    # Ask how many models to download
    Write-Host "How many models were in this experiment? (Enter number to download from outputs/)" -ForegroundColor Cyan
    Write-Host ""
    
    $numModels = Read-Host "Number of models"
    
    # If empty input, use default
    if ([string]::IsNullOrWhiteSpace($numModels)) {
        $numModels = $defaultModels.ToString()
        Write-Host "  (Using default: $defaultModels models)" -ForegroundColor Yellow
    }
    
    if ($numModels -match "^\d+$" -and [int]$numModels -gt 0) {
        $numModels = [int]$numModels
        
        # Get the last N folders from outputs/ (excluding 'experiments' folder)
        Write-Host ""
        Write-Host "Finding last $numModels model folders from outputs/..." -ForegroundColor Cyan
        try {
            $modelFolders = (& ssh -i $PemPath ubuntu@$Ec2Ip "cd $RemoteRoot && ls -1td */ | grep -v 'experiments/' | head -$numModels").Trim()
            $modelFolderList = $modelFolders -split "`n" | Where-Object { $_ }
            
            if ($modelFolderList.Count -eq 0) {
                Write-Error "No model folders found in outputs/"
                exit 1
            }
            
            Write-Host "  Found folders:" -ForegroundColor Cyan
            foreach ($folder in $modelFolderList) {
                Write-Host "    - $folder" -ForegroundColor Gray
            }
            Write-Host ""
            
            # Ensure destination directory exists locally
            New-Item -ItemType Directory -Force -Path $LocalTarget | Out-Null
            
            # Download each model folder
            foreach ($folder in $modelFolderList) {
                $folder = $folder.TrimEnd('/')

                Write-Host "Downloading: $folder" -ForegroundColor Yellow
                if (Download-RemoteFolder -RemoteParent $RemoteRoot -FolderName $folder -Destination $LocalTarget) {
                    Write-Host "  Done." -ForegroundColor Green
                } else {
                    Write-Host "  Failed (tar/ssh exit code non-zero)." -ForegroundColor Red
                }
            }
            
            # Also download the experiments_plan.yaml for reference
            Write-Host ""
            Write-Host "Downloading experiments_plan.yaml..." -ForegroundColor Yellow
            $planScpCmd = "scp -i `"$PemPath`" ubuntu@${Ec2Ip}:`"$remotePath/experiments_plan.yaml`" `"$LocalTarget`""
            try {
                Invoke-Expression $planScpCmd
                Write-Host "  Done." -ForegroundColor Green
            } catch {
                Write-Host "  Failed (non-critical): $_" -ForegroundColor Gray
            }
            
            Write-Host ""
            Write-Host "Download complete." -ForegroundColor Green
            exit 0
            
        } catch {
            Write-Error "Failed to list model folders: $_"
            exit 1
        }
    } else {
        Write-Host "Invalid input. Proceeding with normal download..." -ForegroundColor Gray
    }
}

Write-Host "Remote path:" -ForegroundColor Cyan
Write-Host "  ubuntu@${Ec2Ip}:${remotePath}"
Write-Host "Local destination:" -ForegroundColor Cyan
Write-Host "  $LocalTarget"
Write-Host ""

# Ensure destination directory exists locally
New-Item -ItemType Directory -Force -Path $LocalTarget | Out-Null

Write-Host "Downloading (excluding: $($excludePatterns -join ', ')) ..." -ForegroundColor Yellow
Write-Host ""

if (Download-RemoteFolder -RemoteParent $ExperimentsRoot -FolderName $lastFolder -Destination $LocalTarget) {
    Write-Host ""
    Write-Host "Download complete." -ForegroundColor Green
} else {
    Write-Error "Download failed (tar/ssh exit code non-zero)."
}