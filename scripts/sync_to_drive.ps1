# Sync a results folder to Google Drive with rclone (PowerShell)
# Prereq: rclone installed and a remote named "drive" configured via `rclone config`.
# Usage examples:
#   pwsh scripts/sync_to_drive.ps1 -LocalPath "results/2025-08-15_RNN_3gram_lambda0.6_beam16" -RemotePath "MyRuns/brain2text/2025-08-15_RNN_3gram_lambda0.6_beam16"
#   pwsh scripts/sync_to_drive.ps1 -LocalPath "results" -RemotePath "MyRuns/brain2text"

param(
  [Parameter(Mandatory=$true)][string]$LocalPath,
  [Parameter(Mandatory=$true)][string]$RemotePath,
  [string]$RemoteName = "drive",
  [switch]$Sync # if set, use sync; otherwise copy
)

if (!(Get-Command rclone -ErrorAction SilentlyContinue)) {
  Write-Error "rclone is not installed or not in PATH. See https://rclone.org/install/"
  exit 1
}

$op = if ($Sync) { "sync" } else { "copy" }

$cmd = @(
  "rclone",
  $op,
  "$LocalPath",
  "$RemoteName`:$RemotePath",
  "--progress",
  "--transfers=8",
  "--checkers=8",
  "--drive-stop-on-upload-limit"
)

Write-Host "Running: $($cmd -join ' ')"
& $cmd[0] $cmd[1..($cmd.Length-1)]
if ($LASTEXITCODE -ne 0) {
  Write-Error "rclone $op failed with exit code $LASTEXITCODE"
  exit $LASTEXITCODE
}

Write-Host "Done."

