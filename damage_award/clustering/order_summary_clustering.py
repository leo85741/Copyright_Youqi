#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Order/Summary Clustering Pipeline

This script connects to MongoDB, extracts the "Order/Summary" section from
a section-labeled field, embeds texts with Legal-BERT, clusters with KMeans,
visualizes with UMAP, and extracts representative samples and cluster keywords.

Outputs (saved to --outdir):
- elbow_plot.png
- umap_clusters.png
- representatives.csv
- cluster_summary.csv
- cluster_keywords.csv

Usage:
    python order_summary_clustering.py \
        --mongo-uri "..." \
        --db copyright \
        --collection RST_Preprocessed_SBS \
        --k-min 2 --k-max 10 --k 6 --auto-k \
        --outdir ./outputs

Notes:
- --auto-k uses KneeLocator on inertia to pick K (falls back to --k if not found).
- Requires: sentence-transformers, kneed, umap-learn, spacy (en_core_web_sm), scikit-learn, pymongo, matplotlib, tqdm, pandas, numpy.
"""

import os
import re
import json
import math
import argparse
import warnings
from typing import List, Dict, Any

import numpy as np
import pandas as pd
from tqdm import tqdm
from pymongo import MongoClient

from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import text

from sentence_transformers import SentenceTransformer, models
from kneed import KneeLocator
import umap.umap_ as umap

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for script
import matplotlib.pyplot as plt

import spacy

# ------------------------------
# Helpers
# ------------------------------

def extract_order_summary(section_list: Any) -> str:
    """Extract the first 'Order/Summary' content from a section list."""
    if isinstance(section_list, list):
        for section in section_list:
            if isinstance(section, dict) and section.get("label") == "Order/Summary":
                return section.get("content", "") or ""
    return ""

def ensure_outdir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def lemmatize_text(nlp, text_str: str) -> str:
    doc = nlp(text_str or "")
    tokens = [
        token.lemma_.lower()
        for token in doc
        if token.is_alpha and not token.is_stop and len(token) > 2
    ]
    return " ".join(tokens)

def auto_pick_k(inertias: List[float], k_values: List[int]) -> int:
    """Use KneeLocator to pick K; return -1 if not found."""
    try:
        knee = KneeLocator(k_values, inertias, curve="convex", direction="decreasing")
        return knee.knee if knee and knee.knee is not None else -1
    except Exception:
        return -1

# ------------------------------
# Main pipeline
# ------------------------------

def run_pipeline(args):
    ensure_outdir(args.outdir)

    # ---- MongoDB ----
    client = MongoClient(args.mongo_uri)
    db = client[args.db]
    collection = db[args.collection]

    cursor = collection.find()
    docs = list(cursor)
    df = pd.DataFrame(docs)

    # ---- Extract Order/Summary ----
    if "LLMOUT_SectionLab" not in df.columns:
        raise KeyError("Input collection must contain 'LLMOUT_SectionLab' field.")

    df["Order/Summary"] = df["LLMOUT_SectionLab"].apply(extract_order_summary)
    # Keep non-empty
    df = df[df["Order/Summary"].astype(str).str.strip() != ""].reset_index(drop=True)

    if len(df) == 0:
        raise ValueError("No non-empty 'Order/Summary' records found after extraction.")

    # ---- Encode ----
    # Build Legal-BERT sentence transformer
    bert = models.Transformer('nlpaueb/legal-bert-base-uncased')
    pooling = models.Pooling(bert.get_word_embedding_dimension())
    st_model = SentenceTransformer(modules=[bert, pooling])

    texts = df["Order/Summary"].astype(str).tolist()
    embeddings = st_model.encode(texts, show_progress_bar=True)

    # ---- Elbow method ----
    k_values = list(range(args.k_min, args.k_max + 1))
    inertias = []
    for k in k_values:
        km = KMeans(n_clusters=k, random_state=42, n_init="auto")
        km.fit(embeddings)
        inertias.append(km.inertia_)

    # Plot elbow
    plt.figure(figsize=(6, 4))
    plt.plot(k_values, inertias, marker='o')
    plt.xlabel('k (Number of clusters)')
    plt.ylabel('Inertia (SSE)')
    plt.title('Elbow Method')
    picked_k = args.k
    if args.auto_k:
        knee_k = auto_pick_k(inertias, k_values)
        if knee_k != -1:
            picked_k = knee_k
    plt.axvline(x=picked_k, linestyle='--', label=f'Chosen k = {picked_k}')
    plt.legend()
    plt.grid(True)
    elbow_path = os.path.join(args.outdir, "elbow_plot.png")
    plt.tight_layout()
    plt.savefig(elbow_path, dpi=160)
    plt.close()

    # ---- KMeans clustering ----
    kmeans = KMeans(n_clusters=picked_k, random_state=42, n_init="auto")
    labels = kmeans.fit_predict(embeddings)
    df["cluster"] = labels

    cluster_centers = kmeans.cluster_centers_
    # similarity to own cluster center
    sims = []
    for emb, c in zip(embeddings, labels):
        sims.append(float(cosine_similarity([emb], [cluster_centers[c]])[0][0]))
    df["similarity_to_center"] = sims

    # representatives
    rep_indices = df.groupby("cluster")["similarity_to_center"].idxmax()
    representatives = df.loc[rep_indices].reset_index(drop=True)
    reps_csv = os.path.join(args.outdir, "representatives.csv")
    representatives.to_csv(reps_csv, index=False)

    # ---- UMAP 2D ----
    reducer = umap.UMAP(n_neighbors=args.umap_neighbors,
                        min_dist=args.umap_min_dist,
                        random_state=42)
    umap_2d = reducer.fit_transform(embeddings)
    df["x"] = umap_2d[:, 0]
    df["y"] = umap_2d[:, 1]

    # Plot clusters
    plt.figure(figsize=(10, 7))
    for cluster_id in sorted(df["cluster"].unique()):
        cluster_data = df[df["cluster"] == cluster_id]
        plt.scatter(cluster_data["x"], cluster_data["y"], alpha=0.6, label=f"Cluster {cluster_id}")
    # Mark reps
    plt.scatter(df.loc[rep_indices, "x"], df.loc[rep_indices, "y"], marker='x', s=100, label='Representative')
    plt.title("UMAP + KMeans Cluster Visualization")
    plt.legend()
    plt.grid(True)
    umap_path = os.path.join(args.outdir, "umap_clusters.png")
    plt.tight_layout()
    plt.savefig(umap_path, dpi=160)
    plt.close()

    # ---- Cluster summary ----
    cluster_summary = []
    for cluster_id in sorted(df["cluster"].unique()):
        cluster_data = df[df["cluster"] == cluster_id]
        rep_idx = cluster_data["similarity_to_center"].idxmax()
        representative = df.loc[rep_idx]
        # use 'content' if exists, else fallback to 'Order/Summary'
        rep_text_source = "content" if "content" in df.columns and isinstance(representative.get("content", ""), str) else "Order/Summary"
        rep_snippet = (representative.get(rep_text_source) or "")[:300] + "..."
        cluster_summary.append({
            "cluster": int(cluster_id),
            "size": int(len(cluster_data)),
            "avg_similarity_to_center": float(cluster_data["similarity_to_center"].mean()),
            "rep_similarity": float(representative["similarity_to_center"]),
            "rep_text_source": rep_text_source,
            "rep_snippet": rep_snippet
        })
    cluster_summary_df = pd.DataFrame(cluster_summary)
    cluster_summary_csv = os.path.join(args.outdir, "cluster_summary.csv")
    cluster_summary_df.to_csv(cluster_summary_csv, index=False)

    # ---- Keywords per cluster (TF-IDF over lemmatized text) ----
    # Load spaCy English model
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError as e:
        raise RuntimeError(
            "spaCy model 'en_core_web_sm' not found. Install via:\n"
            "python -m spacy download en_core_web_sm"
        ) from e

    custom_stopwords = {
        "dkt", "doc", "ecf", "default", "judgment", "defendants",
        "new trial", "jury", "verdict", "summary", "plaintiffs"
    }
    full_stops = set(text.ENGLISH_STOP_WORDS).union(custom_stopwords)

    # pick source field for keywording: prefer 'content', else 'Order/Summary'
    if "content" in df.columns and df["content"].astype(str).str.strip().any():
        base_texts = df["content"].astype(str)
    else:
        base_texts = df["Order/Summary"].astype(str)

    df["lemmatized_content"] = [lemmatize_text(nlp, s) for s in tqdm(base_texts, desc="Lemmatizing")]

    vectorizer = TfidfVectorizer(
        stop_words=list(full_stops),
        max_features=5000,
        ngram_range=(1, 2),
        max_df=0.5,
        min_df=2
    )
    tfidf_matrix = vectorizer.fit_transform(df["lemmatized_content"])
    feature_names = vectorizer.get_feature_names_out()

    cluster_keywords = []
    for cluster_id in sorted(df["cluster"].unique()):
        cluster_indices = df[df["cluster"] == cluster_id].index
        if len(cluster_indices) == 0:
            continue
        cluster_tfidf = tfidf_matrix[cluster_indices].mean(axis=0)
        row = cluster_tfidf.A1
        top_n_idx = row.argsort()[::-1][:30]
        top_keywords = [feature_names[i] for i in top_n_idx]
        cluster_keywords.append({
            "cluster": int(cluster_id),
            "top_keywords": top_keywords
        })

    cluster_keywords_df = pd.DataFrame(cluster_keywords)
    cluster_keywords_csv = os.path.join(args.outdir, "cluster_keywords.csv")
    cluster_keywords_df.to_csv(cluster_keywords_csv, index=False)

    # Save a compact manifest
    manifest = {
        "picked_k": int(picked_k),
        "elbow_plot": elbow_path,
        "umap_plot": umap_path,
        "representatives_csv": reps_csv,
        "cluster_summary_csv": cluster_summary_csv,
        "cluster_keywords_csv": cluster_keywords_csv,
        "records_used": int(len(df))
    }
    with open(os.path.join(args.outdir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(json.dumps(manifest, ensure_ascii=False, indent=2))


def build_argparser():
    parser = argparse.ArgumentParser(description="Order/Summary Clustering Pipeline")
    parser.add_argument("--mongo-uri", default="...",
                        help="MongoDB connection URI")
    parser.add_argument("--db", default="copyright", help="Database name")
    parser.add_argument("--collection", default="RST_Preprocessed_SBS", help="Collection name")
    parser.add_argument("--k-min", type=int, default=2, help="Min k to scan (elbow)")
    parser.add_argument("--k-max", type=int, default=10, help="Max k to scan (elbow)")
    parser.add_argument("--k", type=int, default=6, help="Fallback/forced k to use")
    parser.add_argument("--auto-k", action="store_true", help="Auto-pick k via KneeLocator")
    parser.add_argument("--umap-neighbors", type=int, default=15, help="UMAP n_neighbors")
    parser.add_argument("--umap-min-dist", type=float, default=0.1, help="UMAP min_dist")
    parser.add_argument("--outdir", default="./outputs", help="Directory to save outputs")
    return parser

if __name__ == "__main__":
    parser = build_argparser()
    args = parser.parse_args()
    run_pipeline(args)
