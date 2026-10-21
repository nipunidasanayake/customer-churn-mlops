# Model artifacts

Model artifacts are generated locally and are not included in this public
source repository. They are serialized with joblib and must only be loaded
when produced by a trusted training run.

After placing an authorized dataset at `data/telco_customer_churn.csv`, install
the project requirements and run `python src/train.py`. This creates
`churn_model.joblib` and `model_metadata.json` in this directory. These files
are ignored by Git.
