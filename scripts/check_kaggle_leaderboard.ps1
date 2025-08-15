# PowerShell script to check Kaggle competition leaderboard
# This script runs the Python leaderboard checker and handles any errors

param(
    [string]$CompetitionName = "brain-to-text-25"
)

Write-Host "Running Kaggle Leaderboard Checker..." -ForegroundColor Green
Write-Host "Competition: $CompetitionName" -ForegroundColor Cyan

# Get the directory of this script
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$PythonScript = Join-Path $ScriptDir "check_kaggle_leaderboard.py"

# Check if Python script exists
if (-not (Test-Path $PythonScript)) {
    Write-Host "ERROR: Python script not found: $PythonScript" -ForegroundColor Red
    exit 1
}

# Check if Python is available
try {
    $pythonVersion = python --version 2>$null
    if (-not $pythonVersion) {
        Write-Host "ERROR: Python not found. Please ensure Python is installed and in PATH." -ForegroundColor Red
        exit 1
    }
    Write-Host "SUCCESS: Python found: $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "ERROR: Error checking Python: $_" -ForegroundColor Red
    exit 1
}

# Check if Kaggle CLI is available
try {
    $kaggleVersion = kaggle --version 2>$null
    if (-not $kaggleVersion) {
        Write-Host "ERROR: Kaggle CLI not found. Please install with 'pip install kaggle'" -ForegroundColor Red
        exit 1
    }
    Write-Host "SUCCESS: Kaggle CLI found" -ForegroundColor Green
} catch {
    Write-Host "ERROR: Error checking Kaggle CLI: $_" -ForegroundColor Red
    exit 1
}

# Run the Python script
try {
    Write-Host "Executing leaderboard check..." -ForegroundColor Yellow
    
    if ($CompetitionName -eq "brain-to-text-25") {
        python $PythonScript
    } else {
        python $PythonScript $CompetitionName
    }
    
    $exitCode = $LASTEXITCODE
    
    if ($exitCode -eq 0) {
        Write-Host "SUCCESS: Leaderboard check completed successfully!" -ForegroundColor Green
        
        # Show recent log files
        $LogDir = Join-Path (Split-Path -Parent $ScriptDir) "kaggle_logs"
        if (Test-Path $LogDir) {
            $RecentLogs = Get-ChildItem $LogDir -Filter "leaderboard_check_*.json" | Sort-Object LastWriteTime -Descending | Select-Object -First 3
            if ($RecentLogs) {
                Write-Host "`nRecent log files:" -ForegroundColor Cyan
                foreach ($log in $RecentLogs) {
                    Write-Host "  - $($log.Name) - $($log.LastWriteTime)" -ForegroundColor Gray
                }
            }
        }
    } else {
        Write-Host "ERROR: Leaderboard check failed with exit code: $exitCode" -ForegroundColor Red
        exit $exitCode
    }
    
} catch {
    Write-Host "ERROR: Error running leaderboard check: $_" -ForegroundColor Red
    exit 1
}
