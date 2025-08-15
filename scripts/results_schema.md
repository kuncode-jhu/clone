Results artifact layout and schema

Directory layout per run
- results/
  - {run_id}/
    - per_utt.parquet (or per_utt.csv)
    - summary.json
    - decode_cfg.json
    - plots/ (optional: PNGs, PDFs)

Naming
- run_id is a concise, unique label encoding your setting, e.g. 2025-08-15_RNN_3gram_lambda0.6_beam16

per_utt.parquet schema
- run_id: string
- lm_order: int (0 for none, 1, 3, 5)
- lm_weight: float
- beam_size: int
- utt_id: string
- ref: string (reference transcription; or space-separated phonemes/characters)
- hyp: string (decoded hypothesis)
- wer: float (0..1)
- sub: int (Levenshtein substitutions)
- ins: int (insertions)
- del: int (deletions)
- oov_rate: float (optional)
- am_logprob: float (optional; acoustic/model score)
- lm_logprob: float (optional; LM score)
- length: int (token count of ref)
- decode_time_ms: float
- created_at: string (ISO 8601)

summary.json schema (one file per run)
{
  "run_id": "2025-08-15_RNN_3gram_lambda0.6_beam16",
  "lm_order": 3,
  "lm_weight": 0.6,
  "beam_size": 16,
  "n_utts": 1234,
  "wer_macro": 0.172,
  "wer_micro": 0.168,
  "sub": 123,
  "ins": 45,
  "del": 67,
  "decode_time_ms_total": 123456.0,
  "decode_time_ms_p50": 87.3,
  "decode_time_ms_p90": 143.2,
  "created_at": "2025-08-15T04:59:00Z",
  "commit": "<git_sha>",
  "seed": 1337,
  "notes": "optional freeform"
}

decode_cfg.json schema (capture exactly how you decoded)
{
  "model": {
    "checkpoint": "../checkpoints/best.pt",
    "arch": "RNN",
    "seed": 1337
  },
  "decoder": {
    "beam_size": 16,
    "lm_order": 3,
    "lm_weight": 0.6,
    "insertion_penalty": 0.0
  },
  "lm": {
    "arpa": "../lm/3gram.arpa",
    "pruning": "entropy=1e-8",
    "smoothing": "kneser-ney"
  },
  "data": {
    "eval_manifest": "../data/eval.tsv"
  }
}

Conventions
- Use Parquet for per_utt when possible (fast to query with DuckDB/Polars). If pyarrow/fastparquet isn’t available, fall back to CSV with the same columns.
- Always include run_id in every row so you can concatenate multiple runs easily.
- Prefer UTC ISO timestamps for created_at.

Suggested analysis queries (DuckDB)
- SELECT lm_order, lm_weight, AVG(wer) AS wer FROM per_utt GROUP BY 1,2 ORDER BY 1,2;
- SELECT lm_order, lm_weight, SUM(sub) AS sub, SUM(ins) AS ins, SUM(del) AS del FROM per_utt GROUP BY 1,2;

Drive sync tip (rclone)
- rclone copy results/ drive:MyRuns/brain2text --progress
- Or per run: rclone copy results/2025-08-15_RNN_3gram_lambda0.6_beam16 drive:MyRuns/brain2text/2025-08-15_RNN_3gram_lambda0.6_beam16 --progress

