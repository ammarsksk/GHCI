Synthetic training and evaluation datasets generated from the enriched
transaction taxonomy. Files in this directory:

- `train.csv`: 3,344 labelled transactions spanning every taxonomy category.
- `test.csv`: 836 held-out examples for evaluation using the same schema.

Use `python src/synthetic_data.py` to regenerate the datasets. The script
parses `config/taxonomy.yaml` directly, so updates to the taxonomy will be
reflected automatically.
