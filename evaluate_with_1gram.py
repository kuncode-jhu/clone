#!/usr/bin/env python3
"""
Script to evaluate the brain-to-text model with the 1-gram language model.

This script handles:
1. Starting the Redis server
2. Starting the 1-gram language model
3. Running the evaluation
4. Cleaning up

Usage:
    python evaluate_with_1gram.py --eval_type val --gpu_number 0
"""

import argparse
import subprocess
import time
import os
import signal
import sys
import redis
from pathlib import Path

def check_redis_running():
    """Check if Redis server is running"""
    try:
        r = redis.Redis(host='localhost', port=6379, db=0)
        r.ping()
        return True
    except redis.ConnectionError:
        return False

def start_redis_server():
    """Start Redis server if not already running"""
    if check_redis_running():
        print("Redis server is already running")
        return None
    
    print("Starting Redis server...")
    try:
        # Start Redis server in the background
        redis_process = subprocess.Popen(
            ['redis-server', '--daemonize', 'yes'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        # Wait a moment for Redis to start
        time.sleep(2)
        
        if check_redis_running():
            print("Redis server started successfully")
            return redis_process
        else:
            print("Failed to start Redis server")
            return None
            
    except FileNotFoundError:
        print("Redis server not found. Please install Redis:")
        print("  Ubuntu: sudo apt-get install redis-server")
        print("  macOS: brew install redis")
        return None

def check_conda_env(env_name):
    """Check if conda environment exists"""
    try:
        result = subprocess.run(
            ['conda', 'env', 'list'], 
            capture_output=True, 
            text=True, 
            check=True
        )
        return env_name in result.stdout
    except subprocess.CalledProcessError:
        return False

def start_language_model(gpu_number=0):
    """Start the 1-gram language model in a separate process"""
    print(f"Starting 1-gram language model on GPU {gpu_number}...")
    
    # Check if b2txt25_lm environment exists
    if not check_conda_env('b2txt25_lm'):
        print("Warning: b2txt25_lm conda environment not found")
        print("Using current environment instead")
        conda_cmd = []
    else:
        conda_cmd = ['conda', 'run', '-n', 'b2txt25_lm']
    
    # Language model command
    lm_cmd = conda_cmd + [
        'python', 'language_model/language-model-standalone.py',
        '--lm_path', 'language_model/pretrained_language_models/openwebtext_1gram_lm_sil',
        '--do_opt',
        '--nbest', '100',
        '--acoustic_scale', '0.325',
        '--blank_penalty', '90',
        '--alpha', '0.55',
        '--redis_ip', 'localhost',
        '--gpu_number', str(gpu_number)
    ]
    
    print(f"Running command: {' '.join(lm_cmd)}")
    
    # Start language model
    lm_process = subprocess.Popen(
        lm_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    
    # Wait for language model to initialize
    print("Waiting for language model to initialize...")
    start_time = time.time()
    timeout = 60  # 60 second timeout
    
    while time.time() - start_time < timeout:
        if lm_process.poll() is not None:
            # Process ended
            stdout, stderr = lm_process.communicate()
            print(f"Language model process ended unexpectedly:")
            print(f"STDOUT: {stdout}")
            print(f"STDERR: {stderr}")
            return None
        
        # Check if Redis has language model connection
        try:
            r = redis.Redis(host='localhost', port=6379, db=0)
            # Try to get stream info (language model creates streams)
            streams = r.execute_command('INFO', 'stream')
            if 'stream' in streams.lower():
                print("Language model connected successfully")
                return lm_process
        except:
            pass
        
        time.sleep(2)
    
    print("Timeout waiting for language model to start")
    lm_process.terminate()
    return None

def run_evaluation(eval_type='val', gpu_number=1):
    """Run the model evaluation"""
    print(f"Running evaluation on {eval_type} set with GPU {gpu_number}...")
    
    # Check if b2txt25 environment exists
    if not check_conda_env('b2txt25'):
        print("Warning: b2txt25 conda environment not found")
        print("Using current environment instead")
        conda_cmd = []
    else:
        conda_cmd = ['conda', 'run', '-n', 'b2txt25']
    
    # Evaluation command
    eval_cmd = conda_cmd + [
        'python', 'model_training/evaluate_model.py',
        '--model_path', 'data/t15_pretrained_rnn_baseline',
        '--data_dir', 'data/hdf5_data_final',
        '--eval_type', eval_type,
        '--gpu_number', str(gpu_number)
    ]
    
    print(f"Running command: {' '.join(eval_cmd)}")
    
    # Run evaluation
    try:
        result = subprocess.run(
            eval_cmd,
            check=True,
            text=True,
            capture_output=True
        )
        print("Evaluation completed successfully!")
        print("STDOUT:", result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Evaluation failed with error code {e.returncode}")
        print("STDOUT:", e.stdout)
        print("STDERR:", e.stderr)
        return False

def cleanup_processes(lm_process):
    """Clean up background processes"""
    print("Cleaning up processes...")
    
    if lm_process:
        print("Terminating language model...")
        lm_process.terminate()
        try:
            lm_process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            lm_process.kill()
    
    # Stop Redis server
    try:
        subprocess.run(['redis-cli', 'shutdown'], check=False)
        print("Redis server stopped")
    except:
        print("Could not stop Redis server (may not be running)")

def main():
    parser = argparse.ArgumentParser(description='Evaluate model with 1-gram language model')
    parser.add_argument('--eval_type', choices=['val', 'test'], default='val',
                      help='Evaluation type: val or test (default: val)')
    parser.add_argument('--lm_gpu', type=int, default=0,
                      help='GPU number for language model (default: 0)')
    parser.add_argument('--eval_gpu', type=int, default=1,
                      help='GPU number for evaluation (default: 1)')
    parser.add_argument('--skip_lm_start', action='store_true',
                      help='Skip starting language model (assume already running)')
    
    args = parser.parse_args()
    
    lm_process = None
    
    try:
        # Step 1: Start Redis server
        redis_process = start_redis_server()
        if not check_redis_running():
            print("Failed to start Redis server. Exiting.")
            return 1
        
        # Step 2: Start language model (unless skipped)
        if not args.skip_lm_start:
            lm_process = start_language_model(args.lm_gpu)
            if lm_process is None:
                print("Failed to start language model. Exiting.")
                return 1
        else:
            print("Skipping language model startup (assuming already running)")
        
        # Step 3: Run evaluation
        success = run_evaluation(args.eval_type, args.eval_gpu)
        
        if success:
            print(f"\nEvaluation completed! Check the data/t15_pretrained_rnn_baseline/ directory")
            print(f"for the output CSV file: baseline_rnn_{args.eval_type}_predicted_sentences_*.csv")
            return 0
        else:
            print("Evaluation failed.")
            return 1
            
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        return 1
    finally:
        cleanup_processes(lm_process)

if __name__ == '__main__':
    sys.exit(main())
