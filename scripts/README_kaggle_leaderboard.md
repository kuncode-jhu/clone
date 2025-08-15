# Kaggle Leaderboard Checker

This set of scripts checks the current leaderboard for a Kaggle competition and saves the results to the `kaggle_logs` directory with a timestamp.

## Files

- `check_kaggle_leaderboard.py` - Main Python script that fetches and processes leaderboard data
- `check_kaggle_leaderboard.ps1` - PowerShell wrapper script with error handling
- `check_leaderboard.bat` - Windows batch file for easy execution
- `README_kaggle_leaderboard.md` - This documentation file

## Prerequisites

1. **Python** - Ensure Python is installed and available in your PATH
2. **Kaggle CLI** - Install with `pip install kaggle`
3. **Kaggle API credentials** - Set up your Kaggle API token (see [Kaggle API documentation](https://www.kaggle.com/docs/api))

## Usage

### Method 1: Python Script (Direct)
```bash
# Check default competition (brain-to-text-25)
python check_kaggle_leaderboard.py

# Check specific competition
python check_kaggle_leaderboard.py competition-name
```

### Method 2: PowerShell Script
```powershell
# Check default competition
.\check_kaggle_leaderboard.ps1

# Check specific competition
.\check_kaggle_leaderboard.ps1 -CompetitionName "competition-name"
```

### Method 3: Batch File
```cmd
# Check default competition
check_leaderboard.bat

# Check specific competition
check_leaderboard.bat competition-name
```

## Output

The script will:

1. **Display current leader information** including:
   - Team name
   - Current score
   - Total number of participants
   - Timestamp of check

2. **Save detailed log** to `../kaggle_logs/leaderboard_check_YYYYMMDD_HHMMSS.json` containing:
   - Current leader details
   - Top 10 participants
   - Total participant count
   - Timestamp and competition name

## Log File Format

The JSON log files contain:
```json
{
  "timestamp": "2025-08-14T22:47:57.123456",
  "competition": "brain-to-text-25",
  "leader_info": {
    "teamName": "Team Name",
    "score": "0.12345",
    "submissions": "10"
  },
  "top_10": [
    // Array of top 10 participants
  ],
  "total_participants": 17,
  "check_time": "2025-08-14 22:47:57"
}
```

## Error Handling

The scripts include error handling for:
- Missing Python installation
- Missing Kaggle CLI
- Network/API errors
- Competition access issues
- File system errors when saving logs

## Scheduling

You can schedule these scripts to run automatically using:
- **Windows Task Scheduler** (for batch/PowerShell scripts)
- **Cron jobs** (if using WSL/Linux)
- **Python scheduling libraries** (like `schedule` or `APScheduler`)

Example for Windows Task Scheduler:
- Program: `powershell.exe`
- Arguments: `-ExecutionPolicy Bypass -File "C:\path\to\check_kaggle_leaderboard.ps1"`

## Troubleshooting

1. **"Python not found"** - Ensure Python is installed and in your system PATH
2. **"Kaggle CLI not found"** - Install with `pip install kaggle`
3. **API authentication errors** - Verify your `~/.kaggle/kaggle.json` file is properly configured
4. **Competition not found** - Check the competition name spelling and ensure you have access to it
