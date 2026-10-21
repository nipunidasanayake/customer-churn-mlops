# Training data

The Telco customer churn CSV is intentionally not included. The Kaggle dataset
listing identifies the data files as © Original Authors and does not provide a
redistribution license URL. Confirm the applicable terms before using or
redistributing the dataset.

To run training after obtaining an authorized copy, place it at:

`data/telco_customer_churn.csv`

The training script expects `customerID`, `TotalCharges`, and `Churn` plus the
19 model feature columns documented by the API schema. The CSV is ignored by
Git so that it is not accidentally committed.

Source listing: <https://www.kaggle.com/datasets/blastchar/telco-customer-churn>
