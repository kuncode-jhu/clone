# Model Training & Evaluation
This directory contains code and resources for training the brain-to-text RNN model. This model is largely based on the architecture described in the paper "*An Accurate and Rapidly Calibrating Speech Neuroprosthesis*" by Card et al. (2024), but also contains modifications to improve performance, efficiency, and usability.

A pretrained baseline RNN model is included in the [Dryad Dataset](https://datadryad.org/dataset/doi:10.5061/dryad.dncjsxm85), as is the neural data required to train that model. The code for training the same model is included here.

All model training and evaluation code was tested on a computer running Ubuntu 22.04 with two RTX 4090's and 512 GB of RAM.

## Setup
1. Install the required `b2txt25` conda environment by following the instructions in the root `README.md` file. This will set up the necessary dependencies for running the model training and evaluation code.

2. Download the dataset from Dryad: [Dryad Dataset](https://datadryad.org/dataset/doi:10.5061/dryad.dncjsxm85). Place the downloaded data in the `data` directory. See the main [README.md](../README.md) file for more details on the included datasets and the proper `data` directory structure.

## Training
### Baseline RNN Model
We have included a custom PyTorch implementation of the RNN model used in the paper (the paper used a TensorFlow implementation). This implementation aims to replicate or improve upon the original model's performance while leveraging PyTorch's features, resulting in a more efficient training process with a slight increase in decoding accuracy. This model includes day-specific input layers (512x512 linear input layers with softsign activation), a 5-layer GRU with 768 hidden units per layer, and a linear output layer. The model is trained to predict phonemes from neural data using CTC loss and the AdamW optimizer. Data is augmented with noise and temporal jitter to improve robustness. All model hyperparameters are specified in the [`rnn_args.yaml`](rnn_args.yaml) file.

### Model training script
To train the baseline RNN model, use the `b2txt25` conda environment to run the `train_model.py` script from the `model_training` directory:
```bash
conda activate b2txt25
python train_model.py
```
The model will train for 120,000 mini-batches (~3.5 hours on an RTX 4090) and should achieve an aggregate phoneme error rate of 10.1% on the validation partition. We note that the number of training batches and specific model hyperparameters may not be optimal here, and this baseline model is only meant to serve as an example. See [`rnn_args.yaml`](rnn_args.yaml) for a list of all hyperparameters.

### DCoND-LIFT: diphones + LLM refinement
The repository also contains an implementation of the DCoND-LIFT approach described in *Brain-to-Text Decoding with Context-Aware Neural Representations and Large Language Models* (Li et al., 2024). The key additions are:

* **DCoND neural decoder** – a GRU backbone that predicts diphone distributions and marginalises them into phoneme probabilities via a learned blank/diphone head. Training jointly optimises the diphone CTC loss and the marginal phoneme CTC loss with a configurable trade-off parameter `α` that is annealed during the first part of training.
* **Dataset extensions** – when DCoND mode is enabled the dataset automatically constructs diphone supervision targets (including silence-to-phoneme transitions) from the provided phoneme sequences.
* **Evaluation pipeline updates** – `evaluate_model.py` transparently loads either the baseline GRU model or the DCoND variant based on the saved configuration and reports phoneme logits produced by the marginalisation step.

All relevant hyperparameters live in [`dcond_args.yaml`](dcond_args.yaml). To launch DCoND training run:

```bash
conda activate b2txt25
python train_model.py --config dcond_args.yaml
```

This configuration enables diphone supervision, sets the silence index to the CMU `SIL` class (ID 40), and linearly increases `α` from 0.1 to 0.6 over the first 20k optimisation steps. Validation checkpoints store both phoneme and diphone losses so you can monitor convergence.

After training, evaluate the model exactly as with the baseline decoder. The script will detect the stored configuration in `checkpoint/args.yaml` and instantiate the DCoND model automatically. The resulting logits (saved when `save_val_logits: true`) now correspond to the marginalised phoneme distribution required for downstream language model processing.

### Preparing LIFT data for large language models
Running `evaluate_model.py` with `--eval_type val` produces a `val_metrics.pkl` file (when `save_val_logits: true`). For DCoND runs this pickle contains, per trial, the marginal phoneme posteriors, decoded phoneme sequences, and ground-truth transcriptions. These outputs can be combined with the n-best sentences produced by the 5-gram + OPT pipeline (see below) to form the prompts used for GPT-3.5 in-context learning or fine-tuning, as described in Li et al. (2024).

A recommended workflow for reproducing DCoND-LIFT end-to-end is:

1. Train the DCoND model with `dcond_args.yaml`.
2. Evaluate the model on the validation or test split to export phoneme logits.
3. Run the 5-gram + OPT language model (next section) to generate candidate sentences for each trial.
4. Pair each candidate sentence with its corresponding phoneme hypotheses from step 2 to construct prompts for GPT-based refinement (examples of the required prompt format are given in Appendix A.8 of the paper and reproduced in the repository documentation).

## Evaluation
### Start redis server
To evaluate the model, first start a redis server on `localhost` in terminal with:
```bash
redis-server
```

### Start language model
Next, use the `b2txt25_lm` conda environment to start the ngram language model in a seperate terminal window. For example, the 1gram language model can be started using the command below. Note that the 1gram model has no gramatical structure built into it. Details on downloading pretrained 3gram and 5gram language models and running them can be found in the README.md in the `language_model` directory.
To run the 1gram language model from the root directory of this repository:
```bash
conda activate b2txt25_lm
python language_model/language-model-standalone.py --lm_path language_model/pretrained_language_models/openwebtext_1gram_lm_sil --do_opt --nbest 100 --acoustic_scale 0.325 --blank_penalty 90 --alpha 0.55 --redis_ip localhost --gpu_number 0
```
If the language model successfully starts and connects to Redis, you should see a message saying "Successfully connected to the redis server" in the Terminal.

### Evaluate
Finally, use the `b2txt25` conda environment to run the `evaluate_model.py` script to load the pretrained baseline RNN, use it for inference on the heldout val or test sets to get phoneme logits, pass them through the language model via redis to get word predictions, and then save the predicted sentences to a .csv file in the format required for competition submission. An example output file for the val split can be found at `rnn_baseline_submission_file_valsplit.csv`.
```bash
conda activate b2txt25
python evaluate_model.py --model_path ../data/t15_pretrained_rnn_baseline --data_dir ../data/hdf5_data_final --eval_type test --gpu_number 1
```
If the script runs successfully, it will save the predicted sentences to a text file named `baseline_rnn_{eval_type}_predicted_sentences_YYYYMMDD_HHMMSS.csv` in the pretrained model's directory (`/data/t15_pretrained_rnn_baseline`). The `eval_type` can be set to either `val` or `test`, depending on which dataset you want to evaluate.

### Shutdown redis
When you're done, you can shutdown the redis server from any terminal using `redis-cli shutdown`.
