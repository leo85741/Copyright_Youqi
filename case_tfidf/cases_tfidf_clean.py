# -*- coding: utf-8 -*-
"""
Compute TF, DF, and TF-IDF for case URLs from MongoDB.

Usage:
    python cases_tfidf_clean.py

Optional environment variables:
    MONGO_URI          MongoDB connection URI
    DB_NAME            Database name
    COLLECTION_NAME    Collection name
    OUTPUT_CSV         Output CSV path

Example:
    export MONGO_URI="mongodb://USER:PASSWORD@HOST:27017/?authSource=copyright"
    python cases_tfidf_clean.py
"""

import os

import numpy as np
import pandas as pd
from pymongo import MongoClient


# =========================
# MongoDB settings
# =========================
MONGO_URI = os.getenv(
    "MONGO_URI",
    "...",
)
DB_NAME = os.getenv("DB_NAME", "copyright")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "RST_Preprocessed_SBS")
OUTPUT_CSV = os.getenv("OUTPUT_CSV", "cases_TF_DF_TFIDF_clean.csv")


def main() -> None:
    client = MongoClient(MONGO_URI)
    collection = client[DB_NAME][COLLECTION_NAME]

    # Total document count
    n_docs = collection.count_documents({})

    if n_docs == 0:
        raise ValueError("Collection has 0 documents. Cannot compute IDF.")

    # =========================
    # PART 1: Correct TF + DF, grouped only by link
    # =========================
    pipeline_tf_df = [
        {"$unwind": "$urls_dic"},
        {"$match": {"urls_dic.category": "Cases"}},
        {
            "$group": {
                "_id": {
                    "doc_id": "$_id",
                    "link": "$urls_dic.link",
                },
                "tf_in_doc": {"$sum": 1},
            }
        },
        {
            "$group": {
                "_id": "$_id.link",
                "TF": {"$sum": "$tf_in_doc"},
                "DF": {"$sum": 1},
            }
        },
    ]

    tf_df = pd.DataFrame(list(collection.aggregate(pipeline_tf_df)))
    tf_df = tf_df.rename(columns={"_id": "link"})

    # =========================
    # PART 2: Select best raw_text, without affecting DF
    # =========================
    pipeline_text = [
        {"$unwind": "$urls_dic"},
        {"$match": {"urls_dic.category": "Cases"}},
        {
            "$addFields": {
                "word_count": {
                    "$size": {
                        "$filter": {
                            "input": {"$split": ["$urls_dic.raw_text", " "]},
                            "as": "w",
                            "cond": {"$ne": ["$$w", ""]},
                        }
                    }
                }
            }
        },
        {
            "$group": {
                "_id": {
                    "link": "$urls_dic.link",
                    "raw_text": "$urls_dic.raw_text",
                },
                "freq": {"$sum": 1},
                "word_count": {"$first": "$word_count"},
                "category": {"$first": "$urls_dic.category"},
                "source_signal": {"$first": "$urls_dic.source_signal"},
                "confidence": {"$first": "$urls_dic.confidence"},
                "rule_id": {"$first": "$urls_dic.rule_id"},
            }
        },
        {
            "$addFields": {
                "valid": {"$gt": ["$word_count", 5]},
            }
        },
        {
            "$sort": {
                "_id.link": 1,
                "valid": -1,
                "freq": -1,
            }
        },
        {
            "$group": {
                "_id": "$_id.link",
                "raw_text": {"$first": "$_id.raw_text"},
                "category": {"$first": "$category"},
                "source_signal": {"$first": "$source_signal"},
                "confidence": {"$first": "$confidence"},
                "rule_id": {"$first": "$rule_id"},
            }
        },
    ]

    text_df = pd.DataFrame(list(collection.aggregate(pipeline_text)))
    text_df = text_df.rename(columns={"_id": "link"})

    # =========================
    # PART 3: Merge + TF-IDF
    # =========================
    df = pd.merge(tf_df, text_df, on="link", how="left")

    df["IDF"] = np.log(n_docs / df["DF"])
    df["TF_IDF"] = df["TF"] * df["IDF"]

    df = df.sort_values("TF_IDF", ascending=False)

    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

    print(df.head())
    print(f"結果已輸出到 {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
