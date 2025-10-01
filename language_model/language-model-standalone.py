import argparse
import logging
import os
import re
import time
from datetime import datetime
from functools import lru_cache
from typing import Optional

import numpy as np
import redis
import torch
import lm_decoder
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from transformers import BitsAndBytesConfig
except ImportError:  # pragma: no cover - optional dependency for quantisation
    BitsAndBytesConfig = None  # type: ignore[assignment]

# set up logging
logging.basicConfig(format='%(asctime)s %(levelname)s: %(message)s',level=logging.INFO)


def _resolve_torch_dtype(name: Optional[str]) -> torch.dtype:
    """Map a string descriptor to a torch dtype."""

    if name is None:
        return torch.float16 if torch.cuda.is_available() else torch.float32

    normalised = name.lower()
    mapping = {
        'float16': torch.float16,
        'fp16': torch.float16,
        'half': torch.float16,
        'bfloat16': torch.bfloat16,
        'bf16': torch.bfloat16,
        'float32': torch.float32,
        'fp32': torch.float32,
        'f32': torch.float32,
    }
    if normalised not in mapping:
        raise ValueError(
            f"Unsupported torch dtype '{name}'. Supported values: {sorted(mapping.keys())}."
        )
    return mapping[normalised]

# function for initializing the ngram decoder
def build_lm_decoder(
        model_path,
        max_active=7000,
        min_active=200,
        beam=17.,
        lattice_beam=8.0,
        acoustic_scale=1.5,
        ctc_blank_skip_threshold=1.0,
        length_penalty=0.0,
        nbest=1,
    ):

    decode_opts = lm_decoder.DecodeOptions(
        max_active,
        min_active,
        beam,
        lattice_beam,
        acoustic_scale,
        ctc_blank_skip_threshold,
        length_penalty,
        nbest
    )

    TLG_path = os.path.join(model_path, 'TLG.fst')
    words_path = os.path.join(model_path, 'words.txt')
    G_path = os.path.join(model_path, 'G.fst')
    rescore_G_path = os.path.join(model_path, 'G_no_prune.fst')
    if not os.path.exists(rescore_G_path):
        rescore_G_path = ""
        G_path = ""
    if not os.path.exists(TLG_path):
        raise ValueError('TLG file not found at {}'.format(TLG_path))
    if not os.path.exists(words_path):
        raise ValueError('words file not found at {}'.format(words_path))

    decode_resource = lm_decoder.DecodeResource(
        TLG_path,
        G_path,
        rescore_G_path,
        words_path,
        ""
    )
    decoder = lm_decoder.BrainSpeechDecoder(decode_resource, decode_opts)

    return decoder


# function for updating the ngram decoder parameters
def update_ngram_params(
        ngramDecoder,
        max_active=200,
        min_active=17.0,
        beam=13.0,
        lattice_beam=8.0,
        acoustic_scale=1.5,
        ctc_blank_skip_threshold=1.0,
        length_penalty=0.0,
        nbest=100,
    ):
    
    decode_opts = lm_decoder.DecodeOptions(
        max_active,
        min_active,
        beam,
        lattice_beam,
        acoustic_scale,
        ctc_blank_skip_threshold,
        length_penalty,
        nbest,
    )
    ngramDecoder.SetOpt(decode_opts)


# function for initializing the OPT model and tokenizer
def build_opt(
        model_name='facebook/opt-6.7b',
        cache_dir=None,
        device='cuda' if torch.cuda.is_available() else 'cpu',
    ):
    
    '''
    Load the OPT-6.7b model and tokenizer from Hugging Face.
    We will load the model with 16-bit precision for faster inference. This requires ~13 GB of VRAM.
    Put the model onto the GPU (if available).
    '''
    
    # load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        cache_dir=cache_dir,
        torch_dtype=torch.float16,
    )

    if device != 'cpu':
        # Move the model to the GPU
        model = model.to(device)

    # Set the model to evaluation mode
    model.eval()

    # ensure padding token
    tokenizer.padding_side = "right"
    tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


def build_llama(
        model_name='meta-llama/Llama-3.1-8B-Instruct',
        cache_dir=None,
        device: Optional[torch.device] = None,
        torch_dtype: Optional[str] = None,
        device_map: Optional[str] = 'auto',
        load_in_4bit: bool = False,
        load_in_8bit: bool = False,
    ):

    """Load a LLaMA-family causal language model for rescoring."""

    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model_kwargs = {
        'cache_dir': cache_dir,
    }

    if load_in_4bit or load_in_8bit:
        if BitsAndBytesConfig is None:
            raise ImportError(
                'bitsandbytes is required for 4-bit/8-bit loading but is not installed.'
            )
        quant_config = BitsAndBytesConfig(
            load_in_4bit=load_in_4bit,
            load_in_8bit=load_in_8bit,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4',
            bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        )
        model_kwargs['quantization_config'] = quant_config
        model_kwargs['device_map'] = device_map or 'auto'
    else:
        dtype = _resolve_torch_dtype(torch_dtype)
        model_kwargs['torch_dtype'] = dtype
        if device_map and device_map.lower() != 'none':
            model_kwargs['device_map'] = device_map

    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)

    if not (load_in_4bit or load_in_8bit):
        if device_map is None or (isinstance(device_map, str) and device_map.lower() == 'none'):
            if device is None:
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model = model.to(device)

    model.eval()
    return model, tokenizer


# function for rescoring hypotheses with a causal language model
@torch.inference_mode()
def rescore_with_causal_lm(
        model,
        tokenizer,
        device,
        hypotheses,
        length_penalty
    ):

    # set model to evaluation mode
    model.eval()

    inputs = tokenizer(hypotheses, return_tensors='pt', padding=True)
    if device is not None:
        inputs = {k: v.to(device) for k, v in inputs.items()}

    outputs = model(**inputs)
    # compute log-probabilities
    log_probs = torch.nn.functional.log_softmax(outputs.logits, dim=-1)
    log_probs = log_probs.cpu().numpy()

    input_ids = inputs['input_ids'].cpu().numpy()
    attention_mask = inputs['attention_mask'].cpu().numpy()
    batch_size, seq_len, _ = log_probs.shape

    scores = []
    for i in range(batch_size):
        n_tokens = int(attention_mask[i].sum())
        # sum log-probs of each token given the previous context
        score = sum(
            log_probs[i, t-1, input_ids[i, t]]
            for t in range(1, n_tokens)
        )
        scores.append(score - n_tokens * length_penalty)

    return scores


# function for decoding with a causal language model
def gpt2_lm_decode(
        model,
        tokenizer,
        device,
        nbest,
        acoustic_scale,
        length_penalty,
        alpha,
        returnConfidence=False,
        current_context_str=None,
        backend='opt',
    ):

    hypotheses = []
    acousticScores = []
    oldLMScores = []

    for out in nbest:

        # get the candidate sentence (hypothesis)
        hyp = out[0].strip()
        if len(hyp) == 0:
            continue

        # add context to the front of each sentence
        if current_context_str is not None and len(current_context_str.split()) > 0:
            hyp = current_context_str + ' ' + hyp
        
        hyp = hyp.replace('>', '')
        hyp = hyp.replace('  ', ' ')
        hyp = hyp.replace(' ,', ',')
        hyp = hyp.replace(' .', '.')
        hyp = hyp.replace(' ?', '?')
        hypotheses.append(hyp)
        acousticScores.append(out[1])
        oldLMScores.append(out[2])

    if len(hypotheses) == 0:
        logging.error('In g2p_lm_decode, len(hypotheses) == 0')
        return ("", []) if not returnConfidence else ("", [], 0.)
    
    # convert to numpy arrays
    acousticScores = np.array(acousticScores)
    oldLMScores = np.array(oldLMScores)

    # get new LM scores from LLM
    try:
        # first, try to rescore all at once
        newLMScores = np.array(rescore_with_causal_lm(model, tokenizer, device, hypotheses, length_penalty))

    except Exception as e:
        logging.error(f'Error during OPT rescore: {e}')

        try:
            # if that fails, try to rescore in batches (to avoid VRAM issues)
            newLMScores = []
            for i in range(0, len(hypotheses), int(np.ceil(len(hypotheses)/5))):
                newLMScores.extend(rescore_with_causal_lm(model, tokenizer, device, hypotheses[i:i+int(np.ceil(len(hypotheses)/5))], length_penalty))
            newLMScores = np.array(newLMScores)

        except Exception as e:
            logging.error(f'Error during OPT rescore: {e}')
            newLMScores = np.zeros(len(hypotheses))

    # remove context from start of each sentence
    if current_context_str is not None and len(current_context_str.split()) > 0:
        hypotheses = [h[(len(current_context_str)+1):] for h in hypotheses]

    # calculate total scores
    totalScores = (acoustic_scale * acousticScores) + ((1 - alpha) * oldLMScores) + (alpha * newLMScores)

    # get the best hypothesis
    maxIdx = np.argmax(totalScores)
    bestHyp = hypotheses[maxIdx]

    # create nbest output
    nbest_out = []
    min_len = np.min((len(nbest), len(newLMScores), len(totalScores)))
    for i in range(min_len):
        nbest_out.append(';'.join(map(str,[nbest[i][0], nbest[i][1], nbest[i][2], newLMScores[i], totalScores[i]])))

    # return
    if not returnConfidence:
        return bestHyp, nbest_out
    else:
        totalScores = totalScores - np.max(totalScores)
        probs = np.exp(totalScores)
        return bestHyp, nbest_out, probs[maxIdx] / np.sum(probs)


def connect_to_redis_server(redis_ip, redis_port):
    try:
        # logging.info("Attempting to connect to redis...")
        redis_conn = redis.Redis(host=redis_ip, port=redis_port)
        redis_conn.ping()
    except redis.exceptions.ConnectionError:    
        logging.warning("Can't connect to redis server (ConnectionError).")
        return
    else:
        logging.info("Connected to redis.")
        return redis_conn
    

def get_current_redis_time_ms(redis_conn):
    t = redis_conn.time()
    return int(t[0]*1000 + t[1]/1000)


# function to get string differences between two sentences
def get_string_differences(cue, decoder_output):
        decoder_output_words = decoder_output.split()
        cue_words = cue.split()

        @lru_cache(None)
        def reverse_w_backtrace(i, j):
            if i == 0:
                return j, ['I'] * j
            elif j == 0:
                return i, ['D'] * i
            elif i > 0 and j > 0 and decoder_output_words[i-1] == cue_words[j-1]:
                cost, path = reverse_w_backtrace(i-1, j-1)
                return cost, path + [i - 1]
            else:
                insertion_cost, insertion_path = reverse_w_backtrace(i, j-1)
                deletion_cost, deletion_path = reverse_w_backtrace(i-1, j)
                substitution_cost, substitution_path = reverse_w_backtrace(i-1, j-1)
                if insertion_cost <= deletion_cost and insertion_cost <= substitution_cost:
                    return insertion_cost + 1, insertion_path + ['I']
                elif deletion_cost <= insertion_cost and deletion_cost <= substitution_cost:
                    return deletion_cost + 1, deletion_path + ['D']
                else:
                    return substitution_cost + 1, substitution_path + ['R']

        cost, path = reverse_w_backtrace(len(decoder_output_words), len(cue_words))

        # remove insertions from path
        path = [p for p in path if p != 'I']

        # Get the indices in decoder_output of the words that are different from cue
        indices_to_highlight = []
        current_index = 0
        for label, word in zip(path, decoder_output_words):
            if label in ['R','D']:
                indices_to_highlight.append((current_index, current_index+len(word)))
            current_index += len(word) + 1

        return cost, path, indices_to_highlight


def remove_punctuation(sentence):
    # Remove punctuation
    sentence = re.sub(r'[^a-zA-Z\- \']', '', sentence)
    sentence = sentence.replace('- ', ' ').lower()
    sentence = sentence.replace('--', '').lower()
    sentence = sentence.replace(" '", "'").lower()

    sentence = sentence.strip()
    sentence = ' '.join(sentence.split())

    return sentence


# function to augment the nbest list by swapping words around, artificially increasing the number of candidates
def augment_nbest(nbest, top_candidates_to_augment=20, acoustic_scale=0.3, score_penalty_percent=0.01):

    sentences = []
    ac_scores = []
    lm_scores = []
    total_scores = []

    for i in range(len(nbest)):
        sentences.append(nbest[i][0].strip())
        ac_scores.append(nbest[i][1])
        lm_scores.append(nbest[i][2])
        total_scores.append(acoustic_scale*nbest[i][1] + nbest[i][2])

    # sort by total score
    sorted_indices = np.argsort(total_scores)[::-1]
    sentences = [sentences[i] for i in sorted_indices]
    ac_scores = [ac_scores[i] for i in sorted_indices]
    lm_scores = [lm_scores[i] for i in sorted_indices]
    total_scores = [total_scores[i] for i in sorted_indices]

    # new sentences and scores
    new_sentences = []
    new_ac_scores = []
    new_lm_scores = []
    new_total_scores = []

    # swap words around
    for i1 in range(np.min([len(sentences)-1, top_candidates_to_augment])):
        words1 = sentences[i1].split()

        for i2 in range(i1+1, np.min([len(sentences), top_candidates_to_augment])):
            words2 = sentences[i2].split()

            if len(words1) != len(words2):
                continue
            
            _, path1, _ = get_string_differences(sentences[i1], sentences[i2])
            _, path2, _ = get_string_differences(sentences[i2], sentences[i1])

            replace_indices1 = [i for i, p in enumerate(path2) if p == 'R']
            replace_indices2 = [i for i, p in enumerate(path1) if p == 'R']

            for r1, r2 in zip(replace_indices1, replace_indices2):
                
                new_words1 = words1.copy()
                new_words2 = words2.copy()

                new_words1[r1] = words2[r2]
                new_words2[r2] = words1[r1]

                new_sentence1 = ' '.join(new_words1)
                new_sentence2 = ' '.join(new_words2)

                if new_sentence1 not in sentences and new_sentence1 not in new_sentences:
                    new_sentences.append(new_sentence1)
                    new_ac_scores.append(np.mean([ac_scores[i1], ac_scores[i2]]) - score_penalty_percent * np.abs(np.mean([ac_scores[i1], ac_scores[i2]])))
                    new_lm_scores.append(np.mean([lm_scores[i1], lm_scores[i2]]) - score_penalty_percent * np.abs(np.mean([lm_scores[i1], lm_scores[i2]])))
                    new_total_scores.append(acoustic_scale*new_ac_scores[-1] + new_lm_scores[-1])

                if new_sentence2 not in sentences and new_sentence2 not in new_sentences:
                    new_sentences.append(new_sentence2)
                    new_ac_scores.append(np.mean([ac_scores[i1], ac_scores[i2]]) - score_penalty_percent * np.abs(np.mean([ac_scores[i1], ac_scores[i2]])))
                    new_lm_scores.append(np.mean([lm_scores[i1], lm_scores[i2]]) - score_penalty_percent * np.abs(np.mean([lm_scores[i1], lm_scores[i2]])))
                    new_total_scores.append(acoustic_scale*new_ac_scores[-1] + new_lm_scores[-1])

    # combine new sentences and scores with old
    for i in range(len(new_sentences)):
        sentences.append(new_sentences[i])
        ac_scores.append(new_ac_scores[i])
        lm_scores.append(new_lm_scores[i])
        total_scores.append(new_total_scores[i])

    # sort by total score
    sorted_indices = np.argsort(total_scores)[::-1]
    sentences = [sentences[i] for i in sorted_indices]
    ac_scores = [ac_scores[i] for i in sorted_indices]
    lm_scores = [lm_scores[i] for i in sorted_indices]
    total_scores = [total_scores[i] for i in sorted_indices]

    # return nbest
    nbest_out = []
    for i in range(len(sentences)):
        nbest_out.append([sentences[i], ac_scores[i], lm_scores[i]])

    return nbest_out


# main function
def main(args):

    lm_path = args.lm_path
    gpu_number = args.gpu_number

    max_active = args.max_active
    min_active = args.min_active
    beam = args.beam
    lattice_beam = args.lattice_beam
    acoustic_scale = args.acoustic_scale
    ctc_blank_skip_threshold = args.ctc_blank_skip_threshold
    length_penalty = args.length_penalty
    nbest = args.nbest
    top_candidates_to_augment = args.top_candidates_to_augment
    score_penalty_percent = args.score_penalty_percent
    blank_penalty = args.blank_penalty

    llm_backend = args.llm_backend.lower()
    if args.do_opt:
        llm_backend = 'opt'
    llm_cache_dir = args.llm_cache_dir or args.opt_cache_dir
    alpha = args.alpha
    rescore = args.rescore

    llama_model_name = args.llama_model_name
    llm_dtype = args.llm_dtype
    llm_device_map = args.llm_device_map
    llm_load_in_4bit = args.llm_load_in_4bit
    llm_load_in_8bit = args.llm_load_in_8bit

    if llm_load_in_4bit and llm_load_in_8bit:
        raise ValueError('Cannot enable both 4-bit and 8-bit quantization simultaneously.')

    use_llm = llm_backend != 'none'
    backend_display = llm_backend.upper() if use_llm else 'NONE'

    redis_ip = args.redis_ip
    redis_port = args.redis_port
    input_stream = args.input_stream
    partial_output_stream = args.partial_output_stream
    final_output_stream = args.final_output_stream

    # expand user on paths
    lm_path = os.path.expanduser(lm_path)
    if not os.path.exists(lm_path):
        raise ValueError(f'Language model path does not exist: {lm_path}')
    if llm_cache_dir is not None:
        llm_cache_dir = os.path.expanduser(llm_cache_dir)

    # create a nice dict of params to put into redis
    lm_args = {
        'lm_path': lm_path,
        'max_active': int(max_active),
        'min_active': int(min_active),
        'beam': float(beam),
        'lattice_beam': float(lattice_beam),
        'acoustic_scale': float(acoustic_scale),
        'ctc_blank_skip_threshold': float(ctc_blank_skip_threshold),
        'length_penalty': float(length_penalty),
        'nbest': int(nbest),
        'blank_penalty': float(blank_penalty),
        'alpha': float(alpha),
        'do_opt': int(use_llm),
        'rescore': int(rescore),
        'top_candidates_to_augment': int(top_candidates_to_augment),
        'score_penalty_percent': float(score_penalty_percent),
        'llm_backend': llm_backend,
    }

    # pick GPU
    device = torch.device(f"cuda:{gpu_number}" if torch.cuda.is_available() else "cpu")
    logging.info(f'Using device: {device}')

    llm_model = None
    llm_tokenizer = None
    llm_inference_device: Optional[torch.device] = device

    if use_llm:
        if llm_backend == 'opt':
            logging.info(f"Building {backend_display} model from {llm_cache_dir}...")
            start_time = time.time()
            llm_model, llm_tokenizer = build_opt(
                cache_dir=llm_cache_dir,
                device=device,
            )
            logging.info(f'{backend_display} model successfully built in {(time.time()-start_time):0.4f} seconds.')
            llm_inference_device = device
        elif llm_backend == 'llama':
            logging.info(f"Building {backend_display} model {llama_model_name}...")
            start_time = time.time()
            device_map_value = llm_device_map.lower() if llm_device_map is not None else None
            llama_device = device
            if device_map_value is not None and device_map_value.lower() != 'none':
                llama_device = None
            llm_model, llm_tokenizer = build_llama(
                model_name=llama_model_name,
                cache_dir=llm_cache_dir,
                device=llama_device,
                torch_dtype=llm_dtype,
                device_map=device_map_value,
                load_in_4bit=llm_load_in_4bit,
                load_in_8bit=llm_load_in_8bit,
            )
            logging.info(f'{backend_display} model successfully built in {(time.time()-start_time):0.4f} seconds.')
            llm_inference_device = llama_device
        else:
            raise ValueError(f"Unsupported llm_backend '{llm_backend}'.")
    else:
        llm_inference_device = None

    # initialize ngram decoder
    logging.info(f'Initializing language model decoder from {lm_path}...')
    start_time = time.time()
    ngramDecoder = build_lm_decoder(
        lm_path,
        max_active = 7000,
        min_active = 200,
        beam = 17.,
        lattice_beam = 8.,
        acoustic_scale = acoustic_scale,
        ctc_blank_skip_threshold = 1.0,
        length_penalty = 0.0,
        nbest = nbest,
    )
    logging.info(f'Language model successfully initialized in {(time.time()-start_time):0.4f} seconds.')

    # connect to redis server
    REDIS_STATE = -1
    logging.info(f'Attempting to connect to redis at {redis_ip}:{redis_port}...')
    r = connect_to_redis_server(redis_ip, redis_port)
    while r is None:
        r = connect_to_redis_server(redis_ip, redis_port)
        if r is None:
            logging.warning(f'At startup, could not connect to redis server at {redis_ip}:{redis_port}. Trying again in 3 seconds...')
            time.sleep(3)
    logging.info(f'Successfully connected to redis server at {redis_ip}:{redis_port}.')

    timeout_ms = 100
    oldStr = ''
    prev_loop_start_time = 0

    # main loop
    logging.info('Entering main loop...')
    while True:

        # make sure that the loop doesn't run too fast (max 1000 Hz)
        loop_time = time.time() - prev_loop_start_time  
        if loop_time < 0.001:
            time.sleep(0.001 - loop_time)
        prev_loop_start_time = time.time()

        # try catch is to make sure we're connected to redis, and reconnect if not
        try:
            r.ping()

        except redis.exceptions.ConnectionError:
            if REDIS_STATE != 0:
                logging.error(f'Could not connect to the redis server at at {redis_ip}:{redis_port}! I will keep trying...')
            REDIS_STATE = 0
            time.sleep(1)
            continue

        else:
            if REDIS_STATE != 1:
                logging.info('Successfully connected to the redis server.')
                logits_last_entry_seen = get_current_redis_time_ms(r)
                reset_last_entry_seen = get_current_redis_time_ms(r)
                finalize_last_entry_seen = get_current_redis_time_ms(r)
                update_params_last_entry_seen = get_current_redis_time_ms(r)
            REDIS_STATE = 1

            # if the 'remote_lm_args' stream is empty, add the current args
            # (this makes sure it's re-added once redis is flushed at the start of a new block)
            if r.xlen('remote_lm_args') == 0:
                r.xadd('remote_lm_args', lm_args)

            # check if we need to reset
            lm_reset_stream = r.xread(
                {'remote_lm_reset': reset_last_entry_seen},
                count=1,
                block=None,
            )
            if len(lm_reset_stream) > 0:
                for entry_id, entry_data in lm_reset_stream[0][1]:
                    reset_last_entry_seen = entry_id

                # Reset the language model and tell redis, then move on to the next loop
                oldStr = ''
                ngramDecoder.Reset()

                r.xadd('remote_lm_done_resetting', {'done': 1})
                logging.info('Reset the language model.')
                continue

            # check if we need to finalize
            lm_finalize_stream = r.xread(
                {'remote_lm_finalize': finalize_last_entry_seen},
                count=1,
                block=None,
            )
            if len(lm_finalize_stream) > 0:
                for entry_id, entry_data in lm_finalize_stream[0][1]:
                    finalize_last_entry_seen = entry_id

                if r.get('contextual_decoding_current_context') is not None:
                    current_context_str = r.get('contextual_decoding_current_context').decode().strip()
                    if len(current_context_str.split()) > 0:
                        logging.info(f'For LLM rescore, adding context str to the beginning of each candidate sentence:')
                        logging.info(f'\t"{current_context_str}"')
                else:
                    current_context_str = ''

                # Finalize decoding, add the output to the output stream, and then move on to the next loop
                ngramDecoder.FinishDecoding()

                oldStr = ''

                # Optionally rescore with unpruned LM
                if rescore:
                    startT = time.time()
                    ngramDecoder.Rescore()
                    logging.info('Rescore time: %.3f' % (time.time() - startT))


                # Collect candidate sentences and optionally augment them.
                nbest_out = []
                for d in ngramDecoder.result():
                    nbest_out.append([d.sentence, d.ac_score, d.lm_score])

                if nbest > 1 and nbest_out:
                    nbest_out_len = len(nbest_out)
                    nbest_out = augment_nbest(
                        nbest = nbest_out,
                        top_candidates_to_augment = top_candidates_to_augment,
                        acoustic_scale = acoustic_scale,
                        score_penalty_percent = score_penalty_percent,
                    )
                    logging.info(f'Augmented nbest from {nbest_out_len} to {len(nbest_out)} candidates.')

                # Optionally rescore with a LLM
                if use_llm and llm_model is not None and llm_tokenizer is not None and nbest_out:
                    startT = time.time()

                    decoded_final, nbest_redis, confidences = gpt2_lm_decode(
                        llm_model,
                        llm_tokenizer,
                        llm_inference_device,
                        nbest_out,
                        acoustic_scale,
                        alpha = alpha,
                        length_penalty = length_penalty,
                        current_context_str = current_context_str,
                        returnConfidence = True,
                        backend=llm_backend,
                    )
                    logging.info(f'{backend_display} time: {(time.time() - startT):.3f}')

                elif len(ngramDecoder.result()) > 0:
                    # Otherwise just output the best sentence
                    decoded_final = ngramDecoder.result()[0].sentence

                    # create nbest_redis with 0 values for LLM score
                    nbest_redis = []
                    for i in range(len(nbest_out)):
                        sentence = nbest_out[i][0].strip()
                        ac_score = nbest_out[i][1]
                        lm_score = nbest_out[i][2]
                        llm_score = 0.0
                        total_score = acoustic_scale * ac_score + lm_score
                        nbest_redis.append(';'.join(map(str,[sentence, ac_score, lm_score, llm_score, total_score])))

                else:
                    logging.error('No output from language model.')
                    decoded_final = ''
                    nbest_redis = ''
                
                logging.info(f'Final:  {decoded_final}')
                if nbest > 1:
                    r.xadd(final_output_stream, {'lm_response_final': decoded_final, 'scoring': ';'.join(nbest_redis), 'context_str': current_context_str})
                else:   
                    r.xadd(final_output_stream, {'lm_response_final': decoded_final})

                logging.info('Finalized the language model.\n')
                r.xadd('remote_lm_done_finalizing', {'done': 1})
                continue

            # check if we need to update the decoder params
            update_params_stream = r.xread(
                {'remote_lm_update_params': update_params_last_entry_seen},
                count=1,
                block=None,
            )
            if len(update_params_stream) > 0:
                for entry_id, entry_data in update_params_stream[0][1]:
                    update_params_last_entry_seen = entry_id

                    max_active = int(entry_data.get(b'max_active', max_active))
                    min_active = int(entry_data.get(b'min_active', min_active))
                    beam = float(entry_data.get(b'beam', beam))
                    lattice_beam = float(entry_data.get(b'lattice_beam', lattice_beam))
                    acoustic_scale = float(entry_data.get(b'acoustic_scale', acoustic_scale))
                    ctc_blank_skip_threshold = float(entry_data.get(b'ctc_blank_skip_threshold', ctc_blank_skip_threshold))
                    length_penalty = float(entry_data.get(b'length_penalty', length_penalty))
                    nbest = int(entry_data.get(b'nbest', nbest))
                    blank_penalty = float(entry_data.get(b'blank_penalty', blank_penalty))
                    alpha = float(entry_data.get(b'alpha', alpha))
                    incoming_do_opt = entry_data.get(b'do_opt', None)
                    if incoming_do_opt is not None:
                        use_llm = bool(int(incoming_do_opt))
                    incoming_backend = entry_data.get(b'llm_backend', None)
                    if incoming_backend is not None:
                        llm_backend = (incoming_backend.decode() if isinstance(incoming_backend, bytes) else str(incoming_backend)).lower()
                        backend_display = llm_backend.upper() if llm_backend != 'none' else 'NONE'
                        use_llm = llm_backend != 'none'
                    rescore = int(entry_data.get(b'rescore', rescore))
                    top_candidates_to_augment = int(entry_data.get(b'top_candidates_to_augment', top_candidates_to_augment))
                    score_penalty_percent = float(entry_data.get(b'score_penalty_percent', score_penalty_percent))

                    # make sure that the update remote lm args are put into redis nicely
                    lm_args = {
                        'lm_path': lm_path,
                        'max_active': int(max_active),
                        'min_active': int(min_active),
                        'beam': float(beam),
                        'lattice_beam': float(lattice_beam),
                        'acoustic_scale': float(acoustic_scale),
                        'ctc_blank_skip_threshold': float(ctc_blank_skip_threshold),
                        'length_penalty': float(length_penalty),
                        'nbest': int(nbest),
                        'blank_penalty': float(blank_penalty),
                        'alpha': float(alpha),
                        'do_opt': int(use_llm),
                        'rescore': int(rescore),
                        'top_candidates_to_augment': int(top_candidates_to_augment),
                        'score_penalty_percent': float(score_penalty_percent),
                        'llm_backend': llm_backend,
                    }
                    r.xadd('remote_lm_args', lm_args)

                    # update ngram parameters
                    update_ngram_params(
                        ngramDecoder,
                        max_active = max_active,
                        min_active = min_active,
                        beam = beam,
                        lattice_beam = lattice_beam,
                        acoustic_scale = acoustic_scale,
                        ctc_blank_skip_threshold = ctc_blank_skip_threshold,
                        length_penalty = length_penalty,
                        nbest = nbest,
                    )
                    logging.info(
                        f'Updated language model params:' +
                        f'\n\tmax_active = {max_active}' +
                        f'\n\tmin_active = {min_active}' +
                        f'\n\tbeam = {beam}' +
                        f'\n\tlattice_beam = {lattice_beam}' +
                        f'\n\tacoustic_scale = {acoustic_scale}' +
                        f'\n\tctc_blank_skip_threshold = {ctc_blank_skip_threshold}' +
                        f'\n\tlength_penalty = {length_penalty}' +
                        f'\n\tnbest = {nbest}' +
                        f'\n\tblank_penalty = {blank_penalty}' +
                        f'\n\talpha = {alpha}' +
                        f'\n\tllm_backend = {llm_backend}' +
                        f'\n\tuse_llm = {use_llm}' +
                        f'\n\trescore = {rescore}' +
                        f'\n\ttop_candidates_to_augment = {top_candidates_to_augment}' +
                        f'\n\tscore_penalty_percent = {score_penalty_percent}'
                    )
                    r.xadd('remote_lm_done_updating_params', {'done': 1})

                continue


            # ------------------------------------------------------------------------------------------------------------------------
            # ------------ The loop can only get down to here if we're not finalizing, resetting, or updating params -----------------
            # ------------------------------------------------------------------------------------------------------------------------

            # try to read logits from redis stream
            try:
                read_result = r.xread(
                    {input_stream: logits_last_entry_seen},
                    count = 1,
                    block = timeout_ms
                )
            except redis.exceptions.ConnectionError:
                if REDIS_STATE != 0:
                    logging.error(f'Could not connect to the redis server at at {redis_ip}:{redis_port}! I will keep trying...')
                REDIS_STATE = 0
                time.sleep(1)
                continue

            if (len(read_result) >= 1): 
                # --------------- Read input stream --------------------------------
                for entry_id, entry_data in read_result[0][1]:
                    logits_last_entry_seen = entry_id
                    logits = np.frombuffer(entry_data[b'logits'], dtype=np.float32)

                # reshape logits to (T, 41)
                logits = logits.reshape(-1, 41)

                # --------------- Run language model -------------------------------
                lm_decoder.DecodeNumpy(ngramDecoder,
                                        logits,
                                        np.zeros_like(logits),
                                        np.log(blank_penalty))

                # display partial decoded sentence if it exists
                if len(ngramDecoder.result()) > 0:
                    decoded_partial = ngramDecoder.result()[0].sentence
                    newStr = f'Partial: {decoded_partial}'
                    if oldStr != newStr:
                        logging.info(newStr)
                        oldStr = newStr
                else:
                    logging.info('Partial: [NONE]')
                    decoded_partial = ''
                # print(ngramDecoder.result())
                r.xadd(partial_output_stream, {'lm_response_partial': decoded_partial})

            else:
                # timeout if no data received for X ms
                # logging.warning(F'No logits came in for {timeout_ms} ms.')
                continue


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--lm_path', type=str, help='Path to language model folder')
    parser.add_argument('--gpu_number', type=int, default=0, help='GPU number to use')

    parser.add_argument('--max_active', type=int, default=7000, help='max_active param for LM')
    parser.add_argument('--min_active', type=int, default=200, help='min_active param for LM')
    parser.add_argument('--beam', type=float, default=17.0, help='beam param for LM')
    parser.add_argument('--lattice_beam', type=float, default=8.0, help='lattice_beam param for LM')
    parser.add_argument('--ctc_blank_skip_threshold', type=float, default=1., help='ctc_blank_skip_threshold param for LM')
    parser.add_argument('--length_penalty', type=float, default=0.0, help='length_penalty param for LM')
    parser.add_argument('--acoustic_scale', type=float, default=0.3, help='Acoustic scale for LM')
    parser.add_argument('--nbest', type=int, default=100, help='# of candidate sentences for LM decoding')
    parser.add_argument('--top_candidates_to_augment', type=int, default=20, help='# of top candidates to augment')
    parser.add_argument('--score_penalty_percent', type=float, default=0.01, help='Score penalty percent for augmented candidates')
    parser.add_argument('--blank_penalty', type=float, default=9.0, help='Blank penalty for LM')

    parser.add_argument('--rescore', action='store_true', help='Use an unpruned ngram model for rescoring?')
    parser.add_argument('--do_opt', action='store_true', help='Shortcut for --llm_backend=opt')
    parser.add_argument('--opt_cache_dir', type=str, default=None, help='Path to OPT cache (deprecated; use --llm_cache_dir)')
    parser.add_argument('--llm_backend', type=str, default='none', choices=['none', 'opt', 'llama'], help='Causal LLM backend used for rescoring')
    parser.add_argument('--llm_cache_dir', type=str, default=None, help='Cache directory for LLM weights (defaults to --opt_cache_dir if unset)')
    parser.add_argument('--llama_model_name', type=str, default='meta-llama/Llama-3.1-8B-Instruct', help='Hugging Face model id to load when --llm_backend=llama')
    parser.add_argument('--llm_dtype', type=str, default='bfloat16', help='Torch dtype for full-precision LLaMA loading (e.g., float16, bfloat16)')
    parser.add_argument('--llm_device_map', type=str, default='auto', help='Device map used when loading LLaMA. Set to "none" to keep the model on the selected GPU')
    parser.add_argument('--llm_load_in_4bit', action='store_true', help='Load the LLaMA backend using 4-bit quantization (requires bitsandbytes)')
    parser.add_argument('--llm_load_in_8bit', action='store_true', help='Load the LLaMA backend using 8-bit quantization (requires bitsandbytes)')
    parser.add_argument('--alpha', type=float, default=0.5, help='alpha value [0-1]: Higher = more weight on OPT rescore. Lower = more weight on ngram rescore')

    parser.add_argument('--redis_ip', type=str, default='192.168.150.2', help='IP of the redis stream (string)')
    parser.add_argument('--redis_port', type=int, default=6379, help='Port of the redis stream (int)')
    parser.add_argument('--input_stream', type=str, default="remote_lm_input", help='Input stream containing logits')
    parser.add_argument('--partial_output_stream', type=str, default="remote_lm_output_partial", help='Output stream containing partial decoded sentences')
    parser.add_argument('--final_output_stream', type=str, default="remote_lm_output_final", help='Output stream containing final decoded sentences')

    args = parser.parse_args()

    main(args)
