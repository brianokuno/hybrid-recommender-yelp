# Hybrid Recommender System using Spark RDDs and XGBoost

This project implements a scalable, model-based recommendation system using Spark RDDs and XGBoost. It was developed as part of the USC DSCI 553 course and predicts user ratings for businesses on Yelp by engineering features from multiple sources.

---

## Key Features

- Fully RDD-Based: All data processing is done with PySpark RDDs (no DataFrames or Pandas pre-model)
- Model-Based Learning: Uses XGBoost regression to predict user ratings
- Feature Engineering from 5 Sources:
  - `user.json`: user metadata (review count, compliment stats, yelping since, etc.)
  - `business.json`: stars, review count, category richness
  - `review_train.json`: historical (user, business) star averages
  - `tip.json`: tip likes per user-business pair
  - `photo.json`: image label diversity per business
- Efficient Design: No class wrappers, no intermediate CSVs, minimal I/O
- Evaluation: RMSE, error distribution bins, runtime metrics

---

## Results

- Validation RMSE: `0.9790`
- Error Distribution:
  ```
  >=0 and <1     101859
  >=1 and <2      33343
  >=2 and <3       6078
  >=3 and <4        764
  >=4                 0
  ```

---

## How to Run

Make sure you have Spark 3.1.2 and Python 3.6+ installed. Then run:

```bash
spark-submit competition.py <input_folder> <val_csv> <output_csv>
```

Example:

```bash
spark-submit competition.py input/ input/yelp_val.csv output/test_output.csv
```

---

## Repository Contents

- `competition.py`: Spark/XGBoost pipeline (feature extraction + model training)
- `test_output.csv`: Sample prediction file
- *(Data files are excluded due to size; see below)*

---

## Data Sources

This project uses a filtered subset of the Yelp dataset provided for academic use. The following files were used but are not included in this repo:

- `yelp_train.csv`, `yelp_val.csv`
- `user.json`, `business.json`, `review_train.json`, `tip.json`, `photo.json`

---

## Author

**Brian Okuno**  
[okunoanalytics.com](https://okunoanalytics.com)  
This project was completed as part of USC’s Applied Data Science Master’s program.
