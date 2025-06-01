# Recommendation System using Model-Based XGBoost Regression
#
# This program implements a model-based recommendation system using XGBoost regression
# trained on user-business interaction features extracted from the Yelp dataset.
#
# Key features are engineered from five sources: user profiles, business metadata, 
# review history, tip activity, and photo labels. Each user-business pair is represented
# by a vector of aggregated statistics including average review rating, user activity,
# business popularity, tip likes, and photo diversity.
#
# Unlike the reference solution, this implementation flattens the data processing logic 
# into simple functional transformations without using class-based wrappers. It omits 
# intermediate CSV storage and operates in-memory, improving runtime performance and 
# reducing I/O overhead.
#
# The model uses XGBoost with manually tuned hyperparameters to capture non-linear 
# interactions between features. To reduce dimensionality and avoid overfitting, we 
# exclude low-signal review features such as 'funny' and 'cool' votes, and focus on 
# more robust features such as user history, elite status, and visual diversity.
#
# We further improve robustness by computing average compliments only if at least one
# compliment is present, and we aggregate review scores directly using map-reduce 
# primitives rather than raw data accumulation.
#
# The final system outputs rating predictions for a validation set and prints both
# RMSE and the error distribution across prediction intervals for model evaluation.

# Validation RMSE: 0.9793

# Error Distribution:
# >=0 and <1    101960
# >=1 and <2     33183
# >=2 and <3      6121
# >=3 and <4       782
# >=4                0


import csv
import json
import math
import os
import sys
import time
from datetime import datetime

import numpy as np
import pandas as pd
from pyspark import SparkConf, SparkContext
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBRegressor

# Columns to drop before training
DROP_COLUMNS = [
    "user_id", "business_id", "rating",
    "review_avg_stars"
]

# Parameters for XGBoost model
MODEL_PARAMS = {
    "lambda": 12.0,
    "alpha": 0.5,
    "colsample_bytree": 0.6,
    "subsample": 0.85,
    "learning_rate": 0.018,
    "max_depth": 13,
    "random_state": 42,
    "min_child_weight": 80,
    "n_estimators": 320
}


OUTPUT_COLUMNS = ["user_id", "business_id", "prediction"]


# Reads CSV file into an RDD and removes header row
def read_csv_rdd(sc, path):
    rdd = sc.textFile(path)
    header = rdd.first()
    return rdd.filter(lambda row: row != header).map(lambda row: row.split(","))


# Reads JSON file into an RDD of Python dictionaries
def read_json_rdd(sc, path):
    return sc.textFile(path).map(json.loads)


# Processes user.json and extracts key user features into a dictionary
def process_user_json(rdd):
    COMPLIMENT_KEYS = [
        "compliment_hot", "compliment_more", "compliment_profile", "compliment_cute",
        "compliment_list", "compliment_note", "compliment_plain", "compliment_cool",
        "compliment_funny", "compliment_writer", "compliment_photos"
    ]

    def parse(row):
        elite = row.get("elite", "")
        friends = row.get("friends", "")
        compliment_sum = sum(row.get(k, 0) for k in COMPLIMENT_KEYS)
        compliment_count = sum(1 for k in COMPLIMENT_KEYS if row.get(k, 0) > 0)
        avg_compliment = compliment_sum / compliment_count if compliment_count > 0 else 0
        yelp_years = (datetime.now() - datetime.strptime(row["yelping_since"], "%Y-%m-%d")).days / 365.25
        return row["user_id"], (
            row.get("review_count", 0), row.get("useful", 0), row.get("funny", 0), row.get("cool", 0),
            row.get("fans", 0), row.get("average_stars", 3.5),
            len(elite.split(",")) if elite != "None" else 0,
            len(friends.split(",")) if friends != "None" else 0,
            avg_compliment, yelp_years
        )

    return rdd.map(parse).collectAsMap()


# Processes business.json and returns key features per business
def process_business_json(rdd):
    def parse(row):
        return row["business_id"], (
            float(row.get("stars", 3.5)),
            float(row.get("review_count", 0)),
            int(row.get("is_open", 1)),
            len(row.get("attributes", {}) or {}),
            len((row.get("categories") or "").split(","))
        )
    return rdd.map(parse).collectAsMap()


# Aggregates and averages review stars by (user_id, business_id) pair
def process_review_json(rdd):
    def to_key_value(row):
        return ((row["user_id"], row["business_id"]), float(row["stars"]))

    return rdd.map(to_key_value).groupByKey().mapValues(lambda vals: sum(vals) / len(vals)).collectAsMap()


# Sums 'likes' from tip.json by (user_id, business_id)
def process_tip_json(rdd):
    return rdd.map(lambda r: ((r["user_id"], r["business_id"]), r.get("likes", 0))) \
              .reduceByKey(lambda x, y: x + y) \
              .collectAsMap()


# Counts unique and total photo labels per business
def process_photo_json(rdd):
    def map_row(r):
        return (r["business_id"], r["label"])

    return rdd.map(map_row).groupByKey().mapValues(lambda labels: (len(set(labels)), len(labels))).collectAsMap()


# Builds a single feature row for training or prediction
def make_feature_row(row, user_map, biz_map, review_map, tip_map, photo_map):
    uid, bid, rating = row if len(row) == 3 else (row[0], row[1], None)
    u = user_map.get(uid, (0,) * 10)
    b = biz_map.get(bid, (0,) * 5)
    r = (review_map.get((uid, bid), 0),)
    t = (tip_map.get((uid, bid), 0),)
    p = photo_map.get(bid, (0, 0))
    return (uid, bid, *r, *u, *b, *t, *p, float(rating)) if rating is not None else (uid, bid, *r, *u, *b, *t, *p)


# Loads all input files and returns feature DataFrames and ID pairs
def generate_dataframes(sc, input_folder, test_file):
    user_map = process_user_json(read_json_rdd(sc, os.path.join(input_folder, "user.json")))
    biz_map = process_business_json(read_json_rdd(sc, os.path.join(input_folder, "business.json")))
    review_map = process_review_json(read_json_rdd(sc, os.path.join(input_folder, "review_train.json")))
    tip_map = process_tip_json(read_json_rdd(sc, os.path.join(input_folder, "tip.json")))
    photo_map = process_photo_json(read_json_rdd(sc, os.path.join(input_folder, "photo.json")))

    train_rows = read_csv_rdd(sc, os.path.join(input_folder, "yelp_train.csv"))
    test_rows = read_csv_rdd(sc, test_file)

    train_data = train_rows.map(lambda x: make_feature_row((x[0], x[1], x[2]), user_map, biz_map, review_map, tip_map, photo_map)).collect()
    test_data = test_rows.map(lambda x: make_feature_row((x[0], x[1]), user_map, biz_map, review_map, tip_map, photo_map)).collect()
    test_pairs = test_rows.map(lambda x: (x[0], x[1])).collect()

    column_names = [
        "user_id", "business_id", "review_avg_stars",
        "usr_review_count", "usr_useful", "usr_funny", "usr_cool", "usr_fans",
        "usr_avg_stars", "num_elite", "num_friends", "usr_avg_comp", "membership_years",
        "bus_avg_stars", "bus_review_count", "bus_is_open", "num_attrs", "num_categories",
        "likes", "num_cat", "num_img", "rating"
    ]

    train_df = pd.DataFrame(train_data, columns=column_names)
    test_df = pd.DataFrame(test_data, columns=column_names[:-1])
    return train_df, test_df, test_pairs


# Scales data, trains XGBoost model, returns predictions
def train_and_predict(train_df, test_df):
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(train_df.drop(columns=DROP_COLUMNS))
    y_train = train_df["rating"].values
    X_test = scaler.transform(test_df.drop(columns=[col for col in DROP_COLUMNS if col in test_df.columns]))

    model = XGBRegressor(**MODEL_PARAMS)
    model.fit(X_train, y_train)
    return model.predict(X_test)


# Saves predictions to CSV with columns: user_id, business_id, prediction
def save_predictions(pairs, preds, path):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(OUTPUT_COLUMNS)
        writer.writerows([(uid, bid, pred) for (uid, bid), pred in zip(pairs, preds)])


# Calculates RMSE and error distribution against validation labels
def evaluate_predictions(preds, pairs, test_file_path):
    actual_df = pd.read_csv(test_file_path)
    pred_df = pd.DataFrame(pairs, columns=["user_id", "business_id"])
    pred_df["prediction"] = preds

    merged = pd.merge(actual_df, pred_df, on=["user_id", "business_id"])
    merged["error"] = abs(merged["stars"] - merged["prediction"])

    rmse = math.sqrt(mean_squared_error(merged["stars"], merged["prediction"]))
    print(f"Validation RMSE: {rmse:.4f}")

    bins = [-np.inf, 1, 2, 3, 4, np.inf]
    labels = [">=0 and <1", ">=1 and <2", ">=2 and <3", ">=3 and <4", ">=4"]
    merged["Error Range"] = pd.cut(merged["error"], bins=bins, labels=labels, right=False)

    print("\nError Distribution:")
    print(merged["Error Range"].value_counts().sort_index())


# Top-level pipeline runner
def run_pipeline(folder_path, test_file_path, output_file_path):
    conf = SparkConf().setAppName("YelpModelBased")
    sc = SparkContext(conf=conf)
    sc.setLogLevel("ERROR")
    start_time = time.time()

    try:
        train_df, test_df, test_pairs = generate_dataframes(sc, folder_path, test_file_path)
        preds = train_and_predict(train_df, test_df)
        save_predictions(test_pairs, preds, output_file_path)
        evaluate_predictions(preds, test_pairs, test_file_path)
    finally:
        sc.stop()
        print(f"Total Runtime: {time.time() - start_time:.2f} seconds")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: spark-submit competition.py <input_folder> <val_csv> <output_csv>")
        sys.exit(1)

    run_pipeline(sys.argv[1], sys.argv[2], sys.argv[3])
