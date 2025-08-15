#!/usr/bin/env python3
"""
Kaggle Leaderboard Checker Script

This script checks the current leaderboard for the brain-to-text-25 Kaggle competition
and saves the results to the kaggle_logs directory with a timestamp.
"""

import subprocess
import json
import csv
import os
from datetime import datetime
import sys

def run_kaggle_command(cmd):
    """Run a Kaggle CLI command and return the output"""
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, check=True)
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        print(f"Error running command '{cmd}': {e}")
        print(f"Error output: {e.stderr}")
        return None

def get_competition_leaderboard(competition_name="brain-to-text-25"):
    """Get the current leaderboard for the specified competition"""
    print(f"Fetching leaderboard for competition: {competition_name}")
    
    # Get leaderboard in CSV format
    cmd = f"kaggle competitions leaderboard {competition_name} --show --csv"
    leaderboard_csv = run_kaggle_command(cmd)
    
    if leaderboard_csv is None:
        return None
    
    return leaderboard_csv

def parse_leaderboard_csv(csv_content):
    """Parse the CSV content and extract leader information"""
    if not csv_content:
        return None
    
    lines = csv_content.strip().split('\n')
    if len(lines) < 2:
        print("No leaderboard data found")
        return None
    
    # Parse CSV
    reader = csv.DictReader(lines)
    leaderboard_data = list(reader)
    
    if not leaderboard_data:
        print("No leaderboard entries found")
        return None
    
    # Get the leader (first entry)
    leader = leaderboard_data[0]
    
    # Get top 10 or all entries if less than 10
    top_entries = leaderboard_data[:min(10, len(leaderboard_data))]
    
    return {
        "leader": leader,
        "top_10": top_entries,
        "total_entries": len(leaderboard_data)
    }

def save_leaderboard_log(data, competition_name="brain-to-text-25"):
    """Save the leaderboard data to the kaggle_logs directory"""
    
    # Create timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Define log directory and filename
    log_dir = os.path.join("..", "kaggle_logs")
    log_filename = f"leaderboard_check_{timestamp}.json"
    log_filepath = os.path.join(log_dir, log_filename)
    
    # Ensure log directory exists
    os.makedirs(log_dir, exist_ok=True)
    
    # Prepare log entry
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "competition": competition_name,
        "leader_info": data["leader"],
        "top_10": data["top_10"],
        "total_participants": data["total_entries"],
        "check_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    # Save to JSON file
    try:
        with open(log_filepath, 'w') as f:
            json.dump(log_entry, f, indent=2)
        print(f"SUCCESS: Leaderboard data saved to: {log_filepath}")
        return log_filepath
    except Exception as e:
        print(f"ERROR: Error saving log file: {e}")
        return None

def print_leader_summary(data):
    """Print a summary of the current leader"""
    leader = data["leader"]
    
    print("\n" + "="*60)
    print("CURRENT KAGGLE COMPETITION LEADER")
    print("="*60)
    
    # Print leader details
    for key, value in leader.items():
        if key.lower() in ['teamname', 'team', 'name']:
            print(f"Team: {value}")
        elif key.lower() in ['score', 'publicscore', 'public_score']:
            print(f"Score: {value}")
        elif key.lower() in ['submissions', 'submissioncount']:
            print(f"Submissions: {value}")
        elif key.lower() in ['rank', 'position']:
            print(f"Rank: {value}")
    
    print(f"Total Participants: {data['total_entries']}")
    print(f"Check Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)

def main():
    """Main function to run the leaderboard checker"""
    print("Starting Kaggle Leaderboard Check...")
    
    # Get competition name from command line argument or use default
    competition_name = sys.argv[1] if len(sys.argv) > 1 else "brain-to-text-25"
    
    # Get leaderboard data
    leaderboard_csv = get_competition_leaderboard(competition_name)
    
    if leaderboard_csv is None:
        print("ERROR: Failed to fetch leaderboard data")
        return 1
    
    # Parse the leaderboard
    leaderboard_data = parse_leaderboard_csv(leaderboard_csv)
    
    if leaderboard_data is None:
        print("ERROR: Failed to parse leaderboard data")
        return 1
    
    # Print summary
    print_leader_summary(leaderboard_data)
    
    # Save to log file
    log_file = save_leaderboard_log(leaderboard_data, competition_name)
    
    if log_file:
        print(f"\nLog saved successfully!")
        return 0
    else:
        print(f"\nERROR: Failed to save log file")
        return 1

if __name__ == "__main__":
    exit(main())
