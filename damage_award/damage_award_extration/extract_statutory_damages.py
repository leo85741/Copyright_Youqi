#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Extract and evaluate statutory damages amounts from court text stored in MongoDB,
including capturing the LLM's raw output and an evidence (reasoning) quote.

- Connects to a MongoDB collection.
- Builds inputs from `content_clean_statute_new` where statute == "copyright".
- Uses a few-shot LLM prompt to extract:
    (1) the total statutory damages amount,
    (2) a quoted evidence sentence, and
    (3) preserves the model's raw output for auditability.
- Normalizes predictions and ground-truth labels from `annotation_GT_LW`.
- Computes evaluation metrics and (optionally) saves a CSV.

Usage:
    export OPENAI_API_KEY="..."
    python extract_statutory_damages.py \
        --mongo-uri "..." \
        --db copyright \
        --collection RST_Preprocessed_SBS_ON2 \
        --model gpt-5 \
        --temperature 1 \
        --limit 0 \

Notes:
- The OpenAI API key is read from environment variable OPENAI_API_KEY.
- Avoid hardcoding secrets in source files.
"""

import argparse
import os
import re
import sys
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd
from tqdm import tqdm
from pymongo import MongoClient

# LangChain
from langchain.prompts import PromptTemplate, FewShotPromptTemplate
from langchain.chat_models import ChatOpenAI


# ------------------------------
# Few-shot examples WITH Evidence (reasoning quote)
# ------------------------------
EXAMPLES: List[Dict[str, str]] = [
    {
        "content": (
            "For the foregoing reasons, the Court will grant Plaintiffs' motion for default judgment. (Doc. No. 11.) "
            "An order consistent with this memorandum follows. , this 14th day of August 2014, : 1. Plaintiffs' motion for "
            "default judgment (Doc. No. 11) is ; 2. Defendants Kujo Long LLC, Stephen G. Kujovsky and Lonnie M. Long and their "
            "agents, servants, employees and all persons acting under their permission or authority shall be permanently "
            "enjoined and restrained from infringing, in any manner, the copyrighted musical compositions licensed by Broadcast "
            "Music, Inc.; 3. Defendants are to pay statutory damages in the amount of $2,000 for each of the twelve violations, "
            "for a total of $24,000, plus interest, pursuant to 17 U.S.C. § 504(c) and 28 U.S.C. § 1961; 4. Defendants are to pay "
            "attorneys' fees of $3,330.00 and costs of $619.30; and 5. is hereby entered in favor of Plaintiffs and the Clerk of Court "
            "is directed to close the case. /s/ Yvette Kane Yvette Kane, District Judge United States District Court Middle District of Pennsylvania"
        ),
        "output": "Evidence: \"...for a total of $24,000...\"\n24000",
    },
    {
        "content": (
            "CONCLUSION Accordingly, IT IS HEREBY RECOMMENDED that: 1. Plaintiffs' August 17, 2012 motion for default judgment "
            "(Doc. No. 10) be granted; 2. The district court enter a default judgment against defendants Antigua Cantina & Grill, "
            "LCC and Felipe Olvera, Jr.; 3. Defendants, their agents, servants, employees, and all persons acting under their "
            "permission and authority, be enjoined and restrained from infringing, in any manner, the copyrighted musical "
            "compositions licensed by BMI, pursuant to 17 U.S.C. § 502 ; 4. Defendants be directed to pay plaintiffs a total award "
            "of $24,425, comprised of an award of $18,000 in statutory damages pursuant to 17 U.S.C. § 504(c) , and $6,425 in "
            "attorneys' fees and costs pursuant to 17 U.S.C. § 505 ; and 5. Defendants be ordered to pay interest on the full amount "
            "of this judgment from the date of entry of judgment. These findings and recommendations are submitted to the United States "
            "District Judge assigned to the case, pursuant to the provisions of 28 U.S.C. § 636(b)(1) . Within fourteen days after being "
            "served with these findings and recommendations, any party may file written objections with the court and serve a copy on all "
            "parties. Such a document should be captioned \"Objections to Magistrate Judge's Findings and Recommendations.\" Any reply to the "
            "objections shall be served and filed within seven days after service of the objections. The parties are advised that failure to "
            "file objections within the specified time may waive the right to appeal the District Court's order. Martinez v. Ylst, 951 F.2d 1153 "
            "(9th Cir. 1991) . DATED: February 4, 2013. /s/ Dale A. Drozd DALE A. DROZD UNITED STATES MAGISTRATE JUDGE"
        ),
        "output": "Evidence: \"...an award of $18,000 in statutory damages...\"\n18000",
    },
    {
        "content": (
            "For the foregoing reasons, Plaintiffs' Motion for Summary Judgment is GRANTED. Accordingly, the Court ENJOINS Defendants "
            "from further [*20] infringing upon Plaintiffs' copyrights and awards Plaintiffs $ 26,784 in STATUTORY DAMAGES and $ 8,192.40 in "
            "ATTORNEYS FEES and COSTS. The total award is $ 34,976.40. This disposes of all of Plaintiffs' claims before the Court on this matter."
        ),
        "output": "Evidence: \"...awards Plaintiffs $26,784 in STATUTORY DAMAGES...\"\n26784",
    },
    {
        "content": (
            "Here, Plaintiffs are seeking the statutory minimum of $750.00 for each infringement. Since there were 10 infringements, the total comes "
            "to $7,500.00. The Court finds this amount reasonable. Courts routinely award the statutory minimum, or higher, as part of a default [*4] "
            "judgment in copyright infringement cases. See, e.g., D.C. Comics Inc. v. Mini Gift Shop, 912 F.2d 29, 35, 37 (2d Cir. 1990) (awarding greater "
            "than minimum statutory damages award as part of default judgment). Moreover, because Plaintiffs are only seeking the statutory minimum, there "
            "is no need for an evidentiary hearing. See, e.g., id. at 34, 37 (no hearing held where greater than minimum amount of statutory damages awarded)."
        ),
        "output": "Evidence: \"...the total comes to $7,500.00.\"\n7500",
    },
    {
        "content": (
            "Finally, Malibu Media seeks attorneys' fees of $1,182 and costs of $495. Under 17 U.S.C. § 505 , a district court may award costs and attorneys' "
            "fees to a prevailing party in a copyright infringement suit. After reviewing the declaration of Malibu Media's counsel setting forth the time spent "
            "on this matter, the Court finds that the fees and costs sought are reasonable. For the foregoing reasons, the Court will grant Malibu Media's motion "
            "for default judgment in part and deny it in part. An appropriate Order follows. BY THE COURT: /s/ Gene E.K. Pratter GENE E.K. PRATTER United States District Judge , this 24th day of October, 2014, upon consideration of Plaintiff's Motion for Default Judgment (Docket No. 10), it is hereby that the Motion (Docket No. 10) is in part and . The Court orders the following relief: 1. Default Judgment is against Defendant; 2. Statutory damages in the amount of $18,000.00 are awarded to the Plaintiff for the Defendant's 24 infringements alleged in the amended complaint pursuant to 17 U.S.C. § 504(c)(1) ($750.00 for each infringement); 3. Costs and Attorneys' fees pursuant to 17 U.S.C. § 505 in the amount of $1,677.00 are awarded to the Plaintiff. 4. Defendant shall pay to Plaintiff post-judgment interest on the Court's award of $19,677 at the current legal rate allowed and accruing under 28 U.S.C. § 1961 as of the date of this Default Judgment until the date of its satisfaction; 5. A permanent injunction is entered against the Defendant as follows: Defendant is permanently enjoined from directly, contributorily or indirectly infringing Plaintiff's rights under federal or state law of Plaintiff's copyrighted works (the \"Works\"), including, without limitation, by using the internet, BitTorrent or any other online media distribution system to reproduce ( e.g ., download) or distribute the Works, or to make the Works 1 available for distribution to the public, except pursuant to a lawful license or with express authority of Plaintiff. Defendant is ordered to destroy all copies of Plaintiff's Works that Defendant has downloaded onto any computer hard drive or server without Plaintiff's authorization, and shall destroy all copies of the Works transferred onto any physical medium or device in Defendant's possession, custody, or control. 6. The Clerk of Court shall mark this matter for all purposes, including statistics. BY THE COURT: /s/ Gene E.K. Pratter GENE E.K. PRATTER United States District Judge"
        ),
        "output": "Evidence: \"Statutory damages in the amount of $18,000.00 are awarded...\"\n18000",
    },
    {
        "content": (
            "IT IS HEREBY ORDERED THAT the Motion is GRANTED and judgment shall be entered as follows: 1. Defendant is hereby restrained and enjoined from: A. "
            "Infringing Plaintiff's Properties, either directly or contributorily, in any manner, including generally, but not limited to manufacturing, importing, "
            "distributing, advertising, selling and/or offering for sale any merchandise which features any of Plaintiff's Properties, and, specifically: i) "
            "Importing, manufacturing, distributing, advertising, selling and/or offering for sale the Counterfeit Products or any other unauthorized products which "
            "picture, reproduce, copy or use the likenesses of or bear a substantial similarity to any of Plaintiff's Properties; ii) Importing, manufacturing, "
            "distributing, advertising, selling and/or offering for sale in connection thereto any unauthorized promotional materials, labels, packaging or containers "
            "which picture, reproduce, copy or use the likenesses of or bear a confusing similarity to any of Plaintiff's Properties; iii) Engaging in any conduct that "
            "tends falsely to represent that, or is likely to confuse, mislead or deceive purchasers, Defendant's customers and/or members of the public believe, the "
            "actions of Defendant, the products sold by Defendant, or Defendant herself are connected with Plaintiff, are sponsored, approved or licensed by Plaintiff, "
            "or are affiliated with Plaintiff; iv) Affixing, applying, annexing or using in connection with the importation, manufacture, distribution, advertising, sale "
            "and/or offer for sale or other use of any goods or services, a false description or representation, including words or other symbols, tending to falsely "
            "describe or represent such goods as those being those of Plaintiff. 2. Plaintiff is awarded a monetary judgment constituting: A. Statutory damages provided "
            "by the Copyright Act, 17 U.S.C. § 504 , for the willful infringements of Plaintiff's copyright in the amount of $15,000; B. Attorneys' fees totaling $1,500; and "
            "C. Interest on the principal amount of the judgment to Plaintiff at a statutory rate pursuant to 28 U.S.C. § 1961(a) . IT IS SO ORDERED. Dated: June 6, 2013 /s/ "
            "Jesus G. Bernal THE HONORABLE JESUS G. BERNAL United States District Judge"
        ),
        "output": "Evidence: \"...statutory damages ... in the amount of $15,000;\"\n15000",
    },
]
EXAMPLE_PROMPT = PromptTemplate(
    input_variables=["content", "output"],
    template=(
        "Paragraph:\n{content}\nExpected output:\n{output}\n"
    ),
)

FEW_SHOT_PROMPT = FewShotPromptTemplate(
    examples=EXAMPLES,
    example_prompt=EXAMPLE_PROMPT,
    prefix=(
        "Task: Extract only the **total amount of Statutory Damages** mentioned in the paragraph.\n\n"
        "Rules:\n"
        "• On the FIRST line, output: Evidence: \"<quote the exact sentence(s) from the paragraph that justify the amount>\".\n"
        "• If the paragraph expresses the amount via arithmetic (e.g., \"$250 per work × 2 works\"), "
        "  **compute the final total** and then output it on the second line.\n"
        "• If any operand needed for the calculation is not explicitly present in the paragraph, output N/A on the second line.\n"
        "• On the SECOND line, output only the final number (remove '$' and commas), or N/A.\n"
        "• Do NOT include any other explanations.\n\n"
        "Here are some examples:\n"
    ),
    suffix="Paragraph:\n{content}\n",
    input_variables=["content"],
)

# ------------------------------
# Helpers for building inputs
# ------------------------------
JSONLike = Union[Dict[str, Any], List[Any], str, None]


def _flatten_content(obj: JSONLike) -> Optional[str]:
    """Flatten nested dict/list/string content into a single string (joined by newlines)."""
    if obj is None:
        return None
    if isinstance(obj, str):
        return obj.strip()
    if isinstance(obj, list):
        parts: List[str] = []
        for v in obj:
            t = _flatten_content(v)
            if t:
                parts.append(t)
        return "\n".join(parts) if parts else None
    if isinstance(obj, dict):
        def key_sort(k: Any):
            try:
                return (0, int(k))
            except Exception:
                return (1, str(k))
        parts: List[str] = []
        for k in sorted(obj.keys(), key=key_sort):
            t = _flatten_content(obj[k])
            if t:
                parts.append(t)
        return "\n".join(parts) if parts else None
    return None


def build_copyright_input(doc: Dict[str, Any]) -> Optional[str]:
    """Build concatenated text from doc['content_clean_statute_new'] where statute == 'copyright'."""
    field = doc.get("content_clean_statute_new")
    if field is None:
        return None

    if isinstance(field, dict):
        def key_sort(k: Any):
            try:
                return (0, int(k))
            except Exception:
                return (1, str(k))
        items = [field[k] for k in sorted(field.keys(), key=key_sort)]
    elif isinstance(field, list):
        items = field
    else:
        items = []

    want: List[str] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        if item.get("statute") == "copyright":
            text = _flatten_content(item.get("content"))
            if text:
                want.append(text)

    return "\n".join(want) if want else None


# ------------------------------
# LLM extraction helpers (RAW + amount + evidence)
# ------------------------------

# def _fallback_evidence_from_text(text: str) -> str:
#     """If model doesn't return Evidence, try to grab a sentence about statutory damages as a fallback."""
#     if not text:
#         return ""
#     sentences = re.split(r"(?<=[\.\?!])\s+", text)
#     for sent in sentences:
#         if "statutory damages" in sent.lower():
#             return sent.strip()
#     for sent in sentences:
#         if "504(c" in sent.lower():
#             return sent.strip()
#     return ""


def extract_award_and_reason(content: str, llm: ChatOpenAI) -> Tuple[str, str, str]:
    """Return (raw_output, amount_str, evidence_str).
    Model expected output format:
        Line 1: Evidence: "<quote>"
        Line 2: <amount or N/A>
    """
    if not content:
        return ("", "N/A", "")

    try:
        _input = FEW_SHOT_PROMPT.format(content=content)
        resp = llm.invoke(_input)
        raw_output = getattr(resp, "content", str(resp)).strip()
    except Exception as e:
        print(f"LLM抽取失敗: {e}")
        return ("", "N/A", "")

    lines = [ln.strip() for ln in raw_output.splitlines() if ln.strip()]

    # 1) Evidence
    evidence = "N/A"
    if lines and lines[0].lower().startswith("evidence:"):
        ev = lines[0].split(":", 1)[1].strip()
        evidence = ev.strip().strip('"').strip("'")
    elif lines:
        evidence = lines[0]
        evidence = "N/A"

    # 2) Amount
    amount = "N/A"
    if len(lines) >= 2:
        second = lines[1].strip()
        if second.upper() != "N/A":
            m = re.search(r"(?:USD\s*)?\$?\s*([0-9][0-9,]*(?:\.\d+)?)", second)
            if m:
                try:
                    amount = str(int(float(re.sub(r"[,$]", "", m.group(1)))))
                except Exception:
                    amount = "N/A"

    return (raw_output, amount, evidence)


# ------------------------------
# Normalization & labels (for evaluation)
# ------------------------------

def _norm_amount(x: Any) -> str:
    """Normalize amounts to integer string; return 'N/A' if not parseable."""
    if x is None:
        return "N/A"
    s = str(x).strip()
    if s.upper() == "N/A" or s == "":
        return "N/A"
    m = re.search(r"(?:USD\s*)?\$?\s*([0-9][0-9,]*(?:\.\d+)?)", s, flags=re.I)
    if not m:
        return "N/A"
    try:
        val = int(float(m.group(1).replace(",", "")))
        return str(val)
    except Exception:
        return "N/A"


def get_true_statutory(row: pd.Series) -> str:
    data = row.get("annotation_GT_LW", None)
    if isinstance(data, dict):
        for k in data.keys():
            if str(k).strip().lower() == "statutory damages award":
                return _norm_amount(data[k])
        for k in data.keys():
            kl = str(k).lower()
            if "statutory" in kl and "damage" in kl and "award" in kl:
                return _norm_amount(data[k])
        return "N/A"
    else:
        return _norm_amount(data)


# ------------------------------
# Main
# ------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract statutory damages, reasoning, and evaluate accuracy.")
    parser.add_argument("--mongo-uri", required=True, help="MongoDB connection URI")
    parser.add_argument("--db", required=True, help="Database name")
    parser.add_argument("--collection", required=True, help="Collection name")
    parser.add_argument("--model", default="gpt-5", help="OpenAI model name (default: gpt-5)")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature (default: 1.0)")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of docs (0 for all)")
    parser.add_argument("--output-csv", default="", help="Optional path to save evaluated dataframe as CSV")
    return parser.parse_args()


essential_cols = [
    "_id",
    "content_for_pred",
    "predicted_raw",
    "predicted_award",
    "predicted_reason",
    "true_award",
    "predicted_award_single",
    "award_match",
]

def main() -> int:
    args = parse_args()

    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        print("ERROR: Please set OPENAI_API_KEY in your environment.")
        return 2

    # Init LLM
    llm = ChatOpenAI(
        model=args.model,
        temperature=args.temperature,
        openai_api_key=api_key,
    )

    # Mongo
    client = MongoClient(args.mongo_uri)
    db = client[args.db]
    collection = db[args.collection]

    # Fetch docs
    # SAMPLED = 1
    query = {"SAMPLED": 1}
    cursor = collection.find(query)

    if args.limit and args.limit > 0:
        cursor = cursor.limit(args.limit)

    rst_docs = list(cursor)
    if not rst_docs:
        print("No documents found.")
        return 0

    df = pd.DataFrame(rst_docs)

    print("開始處理：使用 content_clean_statute_new 中 statute=copyright 的 content 串接...")
    tqdm.pandas()

    # Build model inputs
    df["content_for_pred"] = df.progress_apply(build_copyright_input, axis=1)

    # Run extraction
    def _apply_extract(x: Optional[str]) -> Tuple[str, str, str]:
        return extract_award_and_reason(x, llm) if x else ("", "N/A", "")

    triples = df["content_for_pred"].progress_apply(_apply_extract)
    df["predicted_raw"], df["predicted_award"], df["predicted_reason"] = zip(*triples)

    # Labels & normalization
    df["true_award"] = df.apply(get_true_statutory, axis=1)
    df["predicted_award_single"] = df["predicted_award"].map(_norm_amount)

    # Filter out rows with no content_for_pred
    df_eval = df[df["content_for_pred"].notna()].copy()

    # Valid samples mask
    mask_valid = (
        ((df_eval["true_award"] == "N/A") & (df_eval["predicted_award_single"] == "N/A"))
        | ((df_eval["true_award"] != "N/A") & (df_eval["predicted_award_single"] != "N/A"))
    )

    df_eval["award_match"] = (df_eval["true_award"] == df_eval["predicted_award_single"]) & mask_valid

    valid_cnt = int(mask_valid.sum())
    acc = float(df_eval.loc[mask_valid, "award_match"].mean()) if valid_cnt > 0 else 0.0

    print(f"可評估樣本數：{valid_cnt} / {len(df_eval)}")
    print(f"有效樣本的準確率：{acc:.2%}")

    overall_acc = float((df_eval["true_award"] == df_eval["predicted_award_single"]).mean()) if len(df_eval) else 0.0
    print(f"整體準確率：{overall_acc:.2%}")

    # Save CSV
    raw_path = "result_raw.csv"
    clean_path = "result_clean.csv"

    cols = [c for c in essential_cols if c in df_eval.columns]
    df_eval[cols].to_csv(raw_path, index=False)
    print(f"Saved results to {raw_path}")

    # === NEW: also save a clean CSV with only _id, amount, reasoning ===
    required = {"_id", "predicted_award_single", "predicted_reason"}
    if required.issubset(set(df_eval.columns)):
        clean_df = (
            df_eval[["_id", "predicted_award_single", "predicted_reason"]]
            .rename(columns={
                "predicted_award_single": "amount",
                "predicted_reason": "reasoning",
            })
        )
        clean_df.to_csv(clean_path, index=False)
        print(f"Saved clean results to {clean_path}")
    else:
        missing = required - set(df_eval.columns)
        print(f"Warning: missing columns for clean output: {missing}; skipping result_clean.")


    return 0


if __name__ == "__main__":
    sys.exit(main())
