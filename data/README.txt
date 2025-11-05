Synthetic training and evaluation datasets generated from the enriched
transaction taxonomy. Files in this directory:

- `train.csv`: labelled transactions for training.
- `test.csv`: held-out examples for evaluation using the same schema.

Regenerate with:

- `python src/synthetic_data.py --train-size 100000 --test-size 100000`

Notes:

- The generator adds realistic variety and controlled noise (e.g., casing/
  spacing glitches, occasional refunds, date anomalies, and rare non-INR
  currencies) to mimic real-world data while preserving the canonical schema.
- The script parses `config/taxonomy.yaml` directly, so taxonomy updates are
  reflected automatically.
