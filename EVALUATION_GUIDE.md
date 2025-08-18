# Model Evaluation with 1-Gram Language Model

This guide shows you how to evaluate the brain-to-text model using the 1-gram language model.

## Overview

The evaluation process involves:
1. **Neural Model**: Converts brain signals to phoneme probabilities (logits)
2. **Language Model**: Converts phoneme sequences to word sequences
3. **Evaluation**: Compares predicted sentences to ground truth

## Prerequisites

### Required Conda Environments
```bash
# Check if environments exist
conda env list

# You should see:
# - b2txt25     (for model evaluation)
# - b2txt25_lm  (for language model, if available)
```

### Required Software
```bash
# Install Redis (if not already installed)
# Ubuntu/Debian:
sudo apt-get update
sudo apt-get install redis-server

# macOS:
brew install redis

# Check Redis installation
redis-server --version
```

## Method 1: Automated Script (Recommended)

I've created a helper script that handles the entire process:

```bash
# Make the script executable
chmod +x evaluate_with_1gram.py

# Run evaluation on validation set
python evaluate_with_1gram.py --eval_type val --lm_gpu 0 --eval_gpu 1

# Run evaluation on test set
python evaluate_with_1gram.py --eval_type test --lm_gpu 0 --eval_gpu 1
```

### Script Parameters:
- `--eval_type`: Choose 'val' or 'test' dataset
- `--lm_gpu`: GPU for language model (default: 0)
- `--eval_gpu`: GPU for neural model evaluation (default: 1)
- `--skip_lm_start`: Skip starting language model if already running

## Method 2: Manual Step-by-Step

### Step 1: Start Redis Server
```bash
# Start Redis server
redis-server --daemonize yes

# Verify Redis is running
redis-cli ping
# Should return: PONG
```

### Step 2: Start Language Model Server
Open a **new terminal** and run:
```bash
# Activate language model environment (if available)
conda activate b2txt25_lm

# Start 1-gram language model
python language_model/language-model-standalone.py \
    --lm_path language_model/pretrained_language_models/openwebtext_1gram_lm_sil \
    --do_opt \
    --nbest 100 \
    --acoustic_scale 0.325 \
    --blank_penalty 90 \
    --alpha 0.55 \
    --redis_ip localhost \
    --gpu_number 0
```

**Wait for this message**: `"Successfully connected to the redis server"`

### Step 3: Run Evaluation
In your **original terminal**:
```bash
# Activate evaluation environment
conda activate b2txt25

# Run evaluation on validation set
python model_training/evaluate_model.py \
    --model_path data/t15_pretrained_rnn_baseline \
    --data_dir data/hdf5_data_final \
    --eval_type val \
    --gpu_number 1

# Or run on test set
python model_training/evaluate_model.py \
    --model_path data/t15_pretrained_rnn_baseline \
    --data_dir data/hdf5_data_final \
    --eval_type test \
    --gpu_number 1
```

### Step 4: Cleanup
```bash
# Stop Redis server
redis-cli shutdown

# Stop language model (Ctrl+C in the language model terminal)
```

## Output Files

The evaluation will create a CSV file in the model directory:
```
data/t15_pretrained_rnn_baseline/baseline_rnn_{eval_type}_predicted_sentences_YYYYMMDD_HHMMSS.csv
```

### CSV Format:
```csv
session,block,trial,pred_sentence
t15.2023.08.11,2,1,"hello world"
t15.2023.08.11,2,2,"this is a test"
...
```

## Understanding the Parameters

### Language Model Parameters:
- `--acoustic_scale 0.325`: Weight for acoustic model vs language model
- `--blank_penalty 90`: Penalty for blank tokens (silence)
- `--alpha 0.55`: Length penalty for sentence scoring
- `--nbest 100`: Number of candidate sentences to consider

### Model Evaluation:
- `--eval_type val`: Use validation set (has ground truth for WER calculation)
- `--eval_type test`: Use test set (for final submission)
- `--gpu_number`: Which GPU to use for neural network inference

## Expected Performance

With the 1-gram language model, you should expect:
- **Validation WER**: ~15-25% (varies by session)
- **Processing Time**: ~15-20 minutes for full validation set
- **Output**: Predicted sentences for each trial

## Troubleshooting

### Common Issues:

1. **Redis Connection Error**:
   ```bash
   # Check if Redis is running
   redis-cli ping
   
   # If not running, start it
   redis-server --daemonize yes
   ```

2. **CUDA Out of Memory**:
   - Use different GPUs for language model and evaluation
   - Reduce batch size in the evaluation script

3. **Language Model Not Connecting**:
   - Wait longer for initialization (can take 1-2 minutes)
   - Check GPU memory availability
   - Verify conda environment has required packages

4. **Missing Files**:
   - Ensure `data/t15_pretrained_rnn_baseline/` exists
   - Ensure `data/hdf5_data_final/` contains the evaluation data
   - Ensure `language_model/pretrained_language_models/openwebtext_1gram_lm_sil/` exists

### Debug Commands:
```bash
# Check GPU usage
nvidia-smi

# Check Redis streams (should show language model activity)
redis-cli
> INFO stream

# Check file existence
ls -la data/t15_pretrained_rnn_baseline/
ls -la data/hdf5_data_final/
ls -la language_model/pretrained_language_models/openwebtext_1gram_lm_sil/
```

## Next Steps

After evaluation, you can:
1. Analyze the results using the phoneme database we created earlier
2. Compare performance across different sessions/days
3. Try different language model parameters
4. Use the 3-gram or 5-gram language models for better performance

The CSV output can be loaded into the phoneme analysis database for detailed performance analysis with Kumo AI.
