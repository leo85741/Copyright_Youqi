# Copyright Court Decision Analysis

This repository contains multiple tasks related to copyright court decision analysis, including statutory damages extraction, few-shot clustering, TF-IDF calculation for cited cases, and LexisNexis HTML case document parsing.

## Repository Structure

| Folder / File | Description |
|---|---|
| `damage_award/damage_award_extration/extract_statutory_damages.py` | Extracts statutory damages amounts from copyright-related court decision paragraphs using few-shot prompting and an LLM. |
| `damage_award/clustering/order_summary_clustering.py` | Performs clustering on Order/Summary sections using Legal-BERT embeddings, K-Means, UMAP visualization, and TF-IDF keyword extraction. |
| `case_tfidf/cases_tfidf_clean.py` | Calculates TF, DF, IDF, and TF-IDF scores for cited cases extracted from court decision documents. |
| `HTML_parsing/lexis_html_parser.py` | Parses HTML case documents exported from LexisNexis and converts the extracted case information into JSON format. |

## Tasks

### 1. Statutory Damages Amount Extraction

This task extracts statutory damages amounts from court decision documents stored in MongoDB. It selects copyright-related paragraphs, applies few-shot prompting with an LLM, normalizes the predicted amounts, and compares them with human annotations.

Script path:

```text
damage_award/damage_award_extration/extract_statutory_damages.py
```

### 2. Few-Shot Clustering

This task clusters Order/Summary paragraphs from court decisions. Legal-BERT is used to generate sentence embeddings, K-Means is applied for clustering, and representative samples are selected for constructing few-shot prompts.

Script path:

```text
damage_award/clustering/order_summary_clustering.py
```

### 3. TF-IDF Calculation for Cited Cases

This task calculates citation importance scores for cited cases in court decisions. It computes TF, DF, IDF, and TF-IDF values based on case links extracted from the `urls_dic` field.

Script path:

```text
case_tfidf/cases_tfidf_clean.py
```

### 4. LexisNexis Case Document Parser

This task parses HTML case documents exported from LexisNexis and converts the extracted case information into a JSON output file.

The parser extracts case metadata and content fields such as case number, case title, case date, court information, reporters, prior history, core terms, case summary, counsel, opinion author, and full opinion text.

In addition, the program automatically generates a LexisNexis-style citation based on the extracted `case_title`, `reporters`, `case_court`, and `case_date`.

Script path:

```text
HTML_parsing/lexis_html_parser.py
```

## Notes

Each task is located in its corresponding folder or script path. Detailed methodology, output files, and field descriptions can be documented in separate `README.md` files within each task folder.
