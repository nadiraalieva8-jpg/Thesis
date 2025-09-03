#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SDG Chi-Square Feature Selection Pipeline
-----------------------------------------

What this script does
- Loads your dataset (Excel or CSV) containing ~313 paragraphs.
- Detects the text column and the label columns for Manual and OSDG.
- Tokenizes text with CountVectorizer (bag-of-words; you can tweak params).
- For each label source (Manual, OSDG) and each SDG in your data:
    * Builds a binary target y_k: paragraph has SDG k (1) vs not (0).
    * Runs chi-square test for each term vs y_k (sklearn.feature_selection.chi2).
    * Produces ranked terms with chi2 and p-values; saves per-SDG CSV files.
- Computes overlap between top-K keyword lists from Manual vs OSDG for each SDG.
- Generates simple bar charts of top terms for a chosen SDG (Manual & OSDG).

How to run
----------
1) Install requirements (only once):
   pip install pandas scikit-learn matplotlib openpyxl

2) Edit the CONFIG section below or pass CLI args:
   python sdg_chi2_pipeline.py --file /path/to/your.xlsx --top_k 50 --min_df 2 --sdg_plot SDG13

3) Outputs will be saved under ./outputs/

Notes
-----
- Multi-label rows are supported (e.g., "SDG13, SDG7").
- Labels are normalized to the canonical form: SDG01..SDG17 when possible.
- If your label columns are named differently, the script tries to auto-detect
  them (case-insensitive search among common variants). You can also force names
  via --manual_col and --osdg_col.

Author: (you can add your name)
"""

import argparse
import re
import os
from typing import List, Dict, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import chi2

# -------------------------------
# CONFIG (can be overridden by CLI)
# -------------------------------
DEFAULT_TOP_K = 50
DEFAULT_MIN_DF = 2
DEFAULT_SDG_FOR_PLOTS = "SDG13"  # Which SDG to visualize if present
OUTPUT_DIR = "outputs"

# Small Russian stopword set (optional, expand if you need to)
RU_STOP = {
    "и","в","во","не","что","он","на","я","с","со","как","а","то","все","она","так",
    "его","но","да","ты","к","у","же","вы","за","бы","по","только","ее","мне","было",
    "вот","от","меня","еще","нет","о","из","ему","теперь","когда","даже","ну","вдруг",
    "ли","если","уже","или","ни","быть","был","него","до","вас","нибудь","опять","уж",
    "вам","ведь","там","потом","себя","ничего","ей","может","они","тут","где","есть","над",
    "надо","ней","для","мы","тебя","их","чем","была","сам","чтоб","без","будто","чего",
    "раз","тоже","себе","под","будет","ж","только","про","них","какая","какой","куда","кто"
}

# Common label column name candidates we will look for (case-insensitive)
MANUAL_CANDIDATES = [
    "label_manual","manual_label","manual labels","manual",
    "my label","mnul label","mnual label","manul label","manual sdg","manual_sdg"
]
OSDG_CANDIDATES = [
    "label_osdg","osdg_label","osdg labels","osdg","osdg ai","osdg_sdg"
]
TEXT_CANDIDATES = [
    "paragraph","text","content","snippet","para","paragraphs"
]

# -------------------------------
# Utilities
# -------------------------------
def ensure_output_dir(path: str = OUTPUT_DIR):
    os.makedirs(path, exist_ok=True)

def detect_column(df: pd.DataFrame, candidates: List[str]) -> str:
    """Detect a column by matching candidates case-insensitively and allowing fuzzy spacing/underscores."""
    cols = list(df.columns)
    lowered = {c.lower().strip(): c for c in cols}
    for cand in candidates:
        key = cand.lower().strip()
        if key in lowered:
            return lowered[key]
    # try relaxed match: ignore spaces/underscores
    norm = {re.sub(r"[\s_]+","", c.lower().strip()): c for c in cols}
    for cand in candidates:
        key = re.sub(r"[\s_]+","", cand.lower().strip())
        if key in norm:
            return norm[key]
    # if nothing found, return the first column as fallback with a warning
    print(f"[WARN] Could not auto-detect from candidates {candidates}. Falling back to first column.")
    return cols[0]

def normalize_sdg_token(token: str) -> str:
    """
    Normalize an SDG label into canonical form 'SDG01'...'SDG17' when possible.
    Accepts forms like 'SDG 13', '13', 'sdg13 Climate Action', etc.
    """
    if token is None:
        return None
    t = str(token).strip()
    if not t:
        return None
    # Extract a number if present
    m = re.search(r"\b(?:sdg)?\s*([0-9]{1,2})\b", t, flags=re.IGNORECASE)
    if m:
        n = int(m.group(1))
        if 1 <= n <= 17:
            return f"SDG{n:02d}"
    # If no number, try to map from SDG name (optional; minimal list here)
    # Add more mappings if your labels are textual
    name_map = {
        "no poverty":"SDG01","zero hunger":"SDG02","good health and well-being":"SDG03",
        "quality education":"SDG04","gender equality":"SDG05","clean water and sanitation":"SDG06",
        "affordable and clean energy":"SDG07","decent work and economic growth":"SDG08",
        "industry, innovation and infrastructure":"SDG09","reduced inequalities":"SDG10",
        "sustainable cities and communities":"SDG11","responsible consumption and production":"SDG12",
        "climate action":"SDG13","life below water":"SDG14","life on land":"SDG15",
        "peace, justice and strong institutions":"SDG16","partnerships for the goals":"SDG17"
    }
    key = t.lower().strip()
    return name_map.get(key, t)  # if we can't normalize, return original

def parse_multi_labels(cell) -> List[str]:
    """
    Parse a multi-label cell into a list of canonical SDG codes.
    Supports separators: ',', ';', '|', '/', '&'
    Example inputs:
      'SDG13, SDG7'
      '13;4'
      'SDG 5 | SDG 10'
    """
    if pd.isna(cell):
        return []
    s = str(cell)
    # split by common delimiters
    parts = re.split(r"[,\|;/&]+", s)
    labels = []
    for p in parts:
        code = normalize_sdg_token(p)
        if code and code not in labels:
            labels.append(code)
    return labels

def build_vectorizer(min_df: int = DEFAULT_MIN_DF) -> CountVectorizer:
    """
    Create a CountVectorizer. You can adjust parameters as needed.
    Note: stop_words='english' removes common English stopwords.
    If your data has Russian, you can add RU_STOP via a custom list.
    """
    # Merge English and our small RU stopword list
    # sklearn only has built-in 'english'; we extend manually by preprocessor trick
    # We'll simply rely on 'english' and a token_pattern that removes 1-char tokens.
    return CountVectorizer(
        lowercase=True,
        stop_words='english',
        token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z\-]{1,}\b",  # tokens with >=2 letters
        ngram_range=(1,1),
        min_df=min_df
    )

def add_ru_stopwords_to_counts(vectorizer: CountVectorizer, X: np.ndarray, feature_names: List[str]) -> Tuple[np.ndarray, List[str]]:
    """
    Optional utility: remove RU_STOP tokens post-hoc from the matrix X by masking columns.
    Only used if you suspect Russian stopwords in your corpus.
    """
    mask = np.array([fn not in RU_STOP for fn in feature_names], dtype=bool)
    X_new = X[:, mask]
    feat_new = [f for f, m in zip(feature_names, mask) if m]
    return X_new, feat_new

def binarize_y_for_sdg(label_lists: List[List[str]], sdg_code: str) -> np.ndarray:
    """Return binary vector y: 1 if sdg_code in labels for that row, else 0."""
    return np.array([1 if sdg_code in labels else 0 for labels in label_lists], dtype=int)

def chi2_for_all_terms(X, y) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run chi-square between features in X and binary target y.
    Returns chi2 scores and p-values (arrays with length = #features).
    """
    # chi2 expects non-negative values (counts). CountVectorizer ensures that.
    chi2_scores, pvals = chi2(X, y)
    return chi2_scores, pvals

def top_terms_df(chi2_scores: np.ndarray, pvals: np.ndarray, feature_names: List[str], top_k: int) -> pd.DataFrame:
    order = np.argsort(-chi2_scores)  # descending
    order = order[:top_k]
    df = pd.DataFrame({
        "term": np.array(feature_names)[order],
        "chi2": chi2_scores[order],
        "p_value": pvals[order]
    })
    return df

def compute_overlap(manual_terms: List[str], osdg_terms: List[str]) -> Dict[str, float]:
    """Compute intersection size, union size, Jaccard, and % overlap relative to smaller list."""
    set_m = set(manual_terms)
    set_o = set(osdg_terms)
    inter = set_m & set_o
    union = set_m | set_o
    jaccard = len(inter)/len(union) if union else 0.0
    pct_overlap_smaller = len(inter)/min(len(set_m), len(set_o)) * 100 if min(len(set_m), len(set_o))>0 else 0.0
    return {
        "intersection_count": len(inter),
        "union_count": len(union),
        "jaccard": jaccard,
        "percent_overlap_smaller": pct_overlap_smaller
    }

def contingency_and_expected(count_in_k_with_term:int, count_in_k_without_term:int,
                             count_not_k_with_term:int, count_not_k_without_term:int) -> Dict[str, float]:
    """
    Build a 2x2 table and compute expected counts under independence.
    Returns observed dict and expected dict (for illustration on one term/SDG).
    """
    A = count_in_k_with_term
    B = count_in_k_without_term
    C = count_not_k_with_term
    D = count_not_k_without_term
    N = A+B+C+D
    # Row/col totals
    row1 = A + B
    row2 = C + D
    col1 = A + C
    col2 = B + D
    # Expected under independence
    E_A = row1 * col1 / N if N else 0.0
    E_B = row1 * col2 / N if N else 0.0
    E_C = row2 * col1 / N if N else 0.0
    E_D = row2 * col2 / N if N else 0.0
    return {
        "observed_A_inK_withTerm": A,
        "observed_B_inK_withoutTerm": B,
        "observed_C_notK_withTerm": C,
        "observed_D_notK_withoutTerm": D,
        "expected_A": E_A, "expected_B": E_B, "expected_C": E_C, "expected_D": E_D,
        "total_N": N
    }

def plot_top_terms_bar(df_terms: pd.DataFrame, title: str, outpath: str, max_terms: int = 20):
    """
    Make a simple horizontal bar chart for the top terms.
    NOTE (per your plotting rules): single plot, no subplots, no explicit colors.
    """
    df = df_terms.head(max_terms)
    plt.figure(figsize=(10, 6))
    plt.barh(df["term"][::-1], df["chi2"][::-1])
    plt.title(title)
    plt.xlabel("Chi-square score")
    plt.ylabel("Term")
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()

def extract_feature_presence_matrix(X) -> np.ndarray:
    """Convert counts to binary presence (0/1) if you need it for contingency illustration."""
    return (X > 0).astype(int)

def main():
    parser = argparse.ArgumentParser(description="SDG Chi-Square Feature Selection Pipeline")
    parser.add_argument("--file", type=str, required=True, help="Path to your dataset (.xlsx or .csv)")
    parser.add_argument("--sheet", type=str, default=None, help="Excel sheet name (if Excel)")
    parser.add_argument("--text_col", type=str, default=None, help="Override: name of the text column")
    parser.add_argument("--manual_col", type=str, default=None, help="Override: name of the Manual label column")
    parser.add_argument("--osdg_col", type=str, default=None, help="Override: name of the OSDG label column")
    parser.add_argument("--top_k", type=int, default=DEFAULT_TOP_K, help="Top K terms per SDG to output")
    parser.add_argument("--min_df", type=int, default=DEFAULT_MIN_DF, help="Vectorizer min_df")
    parser.add_argument("--sdg_plot", type=str, default=DEFAULT_SDG_FOR_PLOTS, help="Which SDG to plot if present (e.g., SDG13)")
    parser.add_argument("--max_terms_plot", type=int, default=20, help="How many top terms to show in plots")
    args = parser.parse_args()

    ensure_output_dir(OUTPUT_DIR)

    # Load data
    if args.file.lower().endswith((".xlsx",".xls")):
        df = pd.read_excel(args.file, sheet_name=args.sheet)
    elif args.file.lower().endswith(".csv"):
        df = pd.read_csv(args.file)
    else:
        raise ValueError("Unsupported file type. Use .xlsx, .xls, or .csv")

    # Detect columns
    text_col = args.text_col or detect_column(df, TEXT_CANDIDATES)
    manual_col = args.manual_col or detect_column(df, MANUAL_CANDIDATES)
    osdg_col = args.osdg_col or detect_column(df, OSDG_CANDIDATES)

    print(f"[INFO] Using columns -> TEXT: '{text_col}', MANUAL: '{manual_col}', OSDG: '{osdg_col}'")

    # Basic cleaning: drop rows with missing text
    df = df.copy()
    df = df[~df[text_col].isna()].reset_index(drop=True)

    # Parse labels into lists of canonical SDGs
    df["labels_manual"] = df[manual_col].apply(parse_multi_labels)
    df["labels_osdg"] = df[osdg_col].apply(parse_multi_labels)

    # Build vectorizer and matrix
    vectorizer = build_vectorizer(min_df=args.min_df)
    X_counts = vectorizer.fit_transform(df[text_col].astype(str))
    feature_names = vectorizer.get_feature_names_out().tolist()

    # Optionally remove Russian stopwords post-hoc (uncomment to enable)
    # X_counts, feature_names = add_ru_stopwords_to_counts(vectorizer, X_counts, feature_names)

    # Prepare label sets present in the data
    def unique_sdgs(label_series: pd.Series) -> List[str]:
        s = set()
        for lst in label_series:
            for lab in lst:
                s.add(lab)
        # keep only normalized SDGxx forms if possible; otherwise include as-is
        # We sort numerically when possible
        def sdg_key(x):
            m = re.match(r"SDG(\d{2})$", x)
            return int(m.group(1)) if m else 999
        return sorted(list(s), key=sdg_key)

    sdgs_manual = unique_sdgs(df["labels_manual"])
    sdgs_osdg = unique_sdgs(df["labels_osdg"])
    print(f"[INFO] SDGs (Manual) found: {sdgs_manual}")
    print(f"[INFO] SDGs (OSDG) found:   {sdgs_osdg}")

    # Compute chi2 per SDG for MANUAL and OSDG
    results = {"manual": {}, "osdg": {}}

    for source, col in [("manual","labels_manual"), ("osdg","labels_osdg")]:
        present_sdgs = sdgs_manual if source=="manual" else sdgs_osdg
        for sdg in present_sdgs:
            y = binarize_y_for_sdg(df[col].tolist(), sdg)
            if y.sum() == 0 or y.sum() == len(y):
                print(f"[WARN] Skipping {source.upper()} {sdg}: degenerate y (all 0 or all 1).")
                continue
            chi2_scores, pvals = chi2_for_all_terms(X_counts, y)
            df_top = top_terms_df(chi2_scores, pvals, feature_names, top_k=args.top_k)
            results[source][sdg] = df_top

            # Save per-SDG CSV
            out_csv = os.path.join(OUTPUT_DIR, f"top_terms_{source}_{sdg}.csv")
            df_top.to_csv(out_csv, index=False, encoding="utf-8")
            print(f"[SAVE] {out_csv} ({len(df_top)} rows)")

    # Compute overlap of MANUAL vs OSDG top-K for SDGs that exist in both
    overlap_rows = []
    common_sdgs = sorted(set(results["manual"].keys()) & set(results["osdg"].keys()))
    for sdg in common_sdgs:
        terms_m = results["manual"][sdg]["term"].tolist()
        terms_o = results["osdg"][sdg]["term"].tolist()
        ov = compute_overlap(terms_m, terms_o)
        ov_row = {"sdg": sdg, **ov}
        overlap_rows.append(ov_row)

        # Save intersection list for inspection
        inter = list(set(terms_m) & set(terms_o))
        pd.DataFrame({"term": sorted(inter)}).to_csv(
            os.path.join(OUTPUT_DIR, f"overlap_terms_{sdg}.csv"),
            index=False, encoding="utf-8"
        )
    if overlap_rows:
        df_overlap = pd.DataFrame(overlap_rows)
        df_overlap.to_csv(os.path.join(OUTPUT_DIR, "overlap_summary.csv"), index=False, encoding="utf-8")
        print(f"[SAVE] {os.path.join(OUTPUT_DIR, 'overlap_summary.csv')}")
    else:
        print("[INFO] No common SDGs to compute overlap.")

    # Simple demonstration of contingency + expected for one example term & SDG
    # Choose the SDG specified by args.sdg_plot if present, else skip illustration
    if args.sdg_plot in (set(results["manual"].keys()) | set(results["osdg"].keys())):
        # pick a frequent term from the global vocab to illustrate
        # We'll choose the top term from manual if available else osdg
        chosen_source = "manual" if args.sdg_plot in results["manual"] else "osdg"
        demo_terms = results[chosen_source][args.sdg_plot]["term"].tolist()
        if demo_terms:
            demo_term = demo_terms[0]
            print(f"[DEMO] Contingency illustration for term '{demo_term}' and {args.sdg_plot} ({chosen_source.upper()}).")

            # Build presence matrix for the demo term
            term_idx = feature_names.index(demo_term)
            X_bin = extract_feature_presence_matrix(X_counts)
            term_presence = X_bin[:, term_idx].A.ravel() if hasattr(X_bin, "A") else np.array(X_bin)[:, term_idx]

            # y for the chosen SDG, using chosen source
            y_chosen = binarize_y_for_sdg(df[f"labels_{chosen_source}"].tolist(), args.sdg_plot)

            # 2x2 counts
            A = int(((y_chosen == 1) & (term_presence == 1)).sum())
            B = int(((y_chosen == 1) & (term_presence == 0)).sum())
            C = int(((y_chosen == 0) & (term_presence == 1)).sum())
            D = int(((y_chosen == 0) & (term_presence == 0)).sum())

            table = contingency_and_expected(A, B, C, D)
            pd.DataFrame([table]).to_csv(os.path.join(OUTPUT_DIR, f"contingency_{args.sdg_plot}_{demo_term}.csv"),
                                         index=False, encoding="utf-8")
            print(f"[SAVE] contingency table for '{demo_term}' in {args.sdg_plot} -> {os.path.join(OUTPUT_DIR, f'contingency_{args.sdg_plot}_{demo_term}.csv')}")

    # Plots for a chosen SDG (Manual and OSDG if available)
    if args.sdg_plot in results["manual"]:
        plot_top_terms_bar(results["manual"][args.sdg_plot], 
                           title=f"Top terms (Manual) for {args.sdg_plot}",
                           outpath=os.path.join(OUTPUT_DIR, f"plot_top_terms_manual_{args.sdg_plot}.png"),
                           max_terms=args.max_terms_plot)
        print(f"[SAVE] plot_top_terms_manual_{args.sdg_plot}.png")

    if args.sdg_plot in results["osdg"]:
        plot_top_terms_bar(results["osdg"][args.sdg_plot], 
                           title=f"Top terms (OSDG) for {args.sdg_plot}",
                           outpath=os.path.join(OUTPUT_DIR, f"plot_top_terms_osdg_{args.sdg_plot}.png"),
                           max_terms=args.max_terms_plot)
        print(f"[SAVE] plot_top_terms_osdg_{args.sdg_plot}.png")

    print("\n[DONE] All outputs saved under ./outputs")
    print("       Files include: top_terms_*.csv, overlap_summary.csv, overlap_terms_*.csv, plots, and a demo contingency table.\n")

if __name__ == "__main__":
    main()
