import argparse
import json
import os
from datetime import datetime, timezone
from typing import List, Optional

try:
    import pandas as pd
except ImportError:
    pd = None

# Utility to persist per-utterance rows and a run summary in a consistent layout.
# Creates results/{run_id}/ with per_utt.parquet (or CSV), summary.json, decode_cfg.json.


def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def save_per_utt(df_rows: List[dict], out_path: str) -> None:
    if not df_rows:
        raise ValueError("No per-utterance rows provided")
    if pd is None:
        # Fallback to CSV
        import csv
        with open(out_path.replace('.parquet', '.csv'), 'w', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=sorted(df_rows[0].keys()))
            w.writeheader()
            for r in df_rows:
                w.writerow(r)
        return
    import pyarrow  # noqa: F401
    df = pd.DataFrame(df_rows)
    df.to_parquet(out_path, index=False)


def save_json(obj: dict, out_path: str) -> None:
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def main():
    p = argparse.ArgumentParser(description="Save standardized decode results")
    p.add_argument('--run-id', required=True)
    p.add_argument('--out-root', default='results')
    p.add_argument('--lm-order', type=int, required=True)
    p.add_argument('--lm-weight', type=float, required=True)
    p.add_argument('--beam-size', type=int, required=True)
    p.add_argument('--decode-cfg', type=str, help='Path to a JSON file to copy into decode_cfg.json')
    p.add_argument('--per-utt-json', type=str, help='Path to JSONL or JSON list of per-utt rows')
    p.add_argument('--summary-json', type=str, help='Path to JSON summary (optional, will compute minimal if omitted)')
    args = p.parse_args()

    run_dir = os.path.join(args.out_root, args.run_id)
    ensure_dir(run_dir)
    ensure_dir(os.path.join(run_dir, 'plots'))

    # Load per-utt rows
    rows: List[dict] = []
    if args.per-utt-json:
        with open(args.per-utt-json, 'r', encoding='utf-8') as f:
            txt = f.read().strip()
            if txt.startswith('['):
                rows = json.loads(txt)
            else:
                # JSONL
                rows = [json.loads(line) for line in txt.splitlines() if line.strip()]

    # Enrich rows with run metadata
    for r in rows:
        r.setdefault('run_id', args.run_id)
        r.setdefault('lm_order', args.lm_order)
        r.setdefault('lm_weight', args.lm_weight)
        r.setdefault('beam_size', args.beam_size)
        r.setdefault('created_at', now_iso())

    # Save per-utt
    if rows:
        per_utt_path = os.path.join(run_dir, 'per_utt.parquet')
        save_per_utt(rows, per_utt_path)

    # Copy decode_cfg
    if args.decode-cfg:
        with open(args.decode_cfg, 'r', encoding='utf-8') as f:
            cfg = json.load(f)
        save_json(cfg, os.path.join(run_dir, 'decode_cfg.json'))

    # Summary
    if args.summary_json:
        with open(args.summary_json, 'r', encoding='utf-8') as f:
            summ = json.load(f)
    else:
        # Minimal summary from rows
        n = len(rows)
        wer_macro: Optional[float] = None
        if rows and all('wer' in r for r in rows):
            wer_macro = sum(r['wer'] for r in rows) / n
        summ = {
            'run_id': args.run_id,
            'lm_order': args.lm_order,
            'lm_weight': args.lm_weight,
            'beam_size': args.beam_size,
            'n_utts': n,
            'wer_macro': wer_macro,
            'created_at': now_iso(),
        }
    save_json(summ, os.path.join(run_dir, 'summary.json'))

    print(f"Saved run artifacts to {run_dir}")


if __name__ == '__main__':
    main()

