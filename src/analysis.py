import json
import os
import csv
import math
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# --- Configuration ---
# Output paths
TABLES_OUTPUT_DIR = Path("./paper/tables/")
FIGURES_OUTPUT_DIR = Path("./paper/figures/")

os.makedirs(TABLES_OUTPUT_DIR, exist_ok=True)
os.makedirs(FIGURES_OUTPUT_DIR, exist_ok=True)

# Base directories for data files
PREDICTION_BASE_DIR = Path("./data/output_results")
GROUND_TRUTH_BASE_DIR = Path("./data/input_libraries")

# List of libraries to process
developmentLibraryNames = [
    "antibiotic_prescribing_and_telehealth",
    "long_covid",
    "natural_history_primary_care",
    "non_drug_interventions",
    "salt_substitution",
]

evaluationLibraryNames = [
    "balneotherapy_for_chronic_venous",
    "calcium_vitamin_d_for_bones",
    "methylxanthine_for_apnea",
    "phosphodiestrase_5_inhibitors",
    "topical_and_oral_steroids_for_om",
    "non_rct_covid_schools",
    "non_rct_diabetes_tb",
    "non_rct_falls_prevention",
    "non_rct_fluoride_fluorosis",
    "non_rct_sanitation_diarrhoea",
]

# Map libraries to their BibTeX citation keys
CITATION_MAP = {
    "antibiotic_prescribing_and_telehealth": "antibiotic_prescribing_and_telehealth",
    "long_covid": "long_covid",
    "natural_history_primary_care": "natural_history_primary_care",
    "non_drug_interventions": "non_drug_interventions",
    "salt_substitution": "salt_substitution",
    "balneotherapy_for_chronic_venous": "balneotherapy_for_chronic_venous",
    "calcium_vitamin_d_for_bones": "calcium_vitamin_d_for_bones",
    "methylxanthine_for_apnea": "methylxanthine_for_apnea",
    "phosphodiestrase_5_inhibitors": "phosphodiestrase_5_inhibitors",
    "topical_and_oral_steroids_for_om": "topical_and_oral_steroids_for_om",
    "non_rct_covid_schools": "non_rct_covid_schools",
    "non_rct_diabetes_tb": "non_rct_diabetes_tb",
    "non_rct_falls_prevention": "non_rct_falls_prevention",
    "non_rct_fluoride_fluorosis": "non_rct_fluoride_fluorosis",
    "non_rct_sanitation_diarrhoea": "non_rct_sanitation_diarrhoea",
}

# Map libraries to their study Author and Year for the CSV output
STUDY_METADATA_MAP = {
    # Evaluation Libraries
    "balneotherapy_for_chronic_venous": {"author": "de Moraes Silva", "year": 2023},
    "calcium_vitamin_d_for_bones": {"author": "Méndez-Sánchez", "year": 2023},
    "methylxanthine_for_apnea": {"author": "Marques", "year": 2023},
    "phosphodiestrase_5_inhibitors": {"author": "Maltez", "year": 2023},
    "topical_and_oral_steroids_for_om": {"author": "Mulvaney", "year": 2023},
    "non_rct_covid_schools": {"author": "Littlecott", "year": 2024},
    "non_rct_diabetes_tb": {"author": "Franco", "year": 2024},
    "non_rct_falls_prevention": {"author": "Lewis", "year": 2024},
    "non_rct_fluoride_fluorosis": {"author": "Wong", "year": 2024},
    "non_rct_sanitation_diarrhoea": {"author": "Bauza", "year": 2023},
}

# Optional map for custom display names in final outputs (tables and figures)
DISPLAY_NAME_MAP = {
    "non_rct_covid_schools": "COVID-19 Measures in Schools",
    "non_rct_diabetes_tb": "Diabetes as a TB Risk Factor",
    "non_rct_falls_prevention": "Falls Interventions",
    "non_rct_fluoride_fluorosis": "Topical Fluoride causing Dental Fluorosis",
    "non_rct_sanitation_diarrhoea": "Sanitation for Prevention Diarrhoea",
}

# Prediction filename pattern components
PREDICTION_SUFFIX = ".json"
GROUND_TRUTH_SUFFIX = "references.json"

# --- End Configuration ---

def wilson_ci(x, n, z=1.96):
    """
    Calculates the Wilson Score Interval for a proportion.
    Robust for extreme proportions (e.g., 100% recall) where Wald intervals fail.
    """
    if n == 0:
        return 0.0, 0.0
    p = x / n
    denominator = 1 + z**2 / n
    centre_adjusted_p = p + z**2 / (2 * n)
    adjusted_std = z * math.sqrt((p * (1 - p) / n) + (z**2 / (4 * n**2)))

    lower = (centre_adjusted_p - adjusted_std) / denominator
    upper = (centre_adjusted_p + adjusted_std) / denominator
    return max(0.0, lower), min(1.0, upper)

# Addresses Reviewer 1 Comment 4 and Reviewer 2 Comment 5 (clustering):
# Records are clustered within reviews, so a record-level (micro) confidence
# interval that treats every record as independent understates uncertainty.
# This cluster bootstrap resamples whole REVIEWS (with replacement) and
# recomputes the macro-average (mean of per-review proportions), yielding a
# 95% interval that reflects between-review heterogeneity rather than assuming
# the ~58,000 records are independent observations.
def cluster_bootstrap_ci(per_review_values, n_boot=10000, seed=42, z_ci=(2.5, 97.5)):
    """
    Percentile cluster-bootstrap CI for the macro-average of a per-review metric.
    Resamples reviews (the clustering unit) with replacement.
    Returns (lower, upper). Degenerate (e.g. all reviews == 1.00) collapses to a point.
    """
    values = np.asarray(per_review_values, dtype=float)
    k = len(values)
    if k == 0:
        return 0.0, 0.0
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot)
    for b in range(n_boot):
        sample = rng.choice(values, size=k, replace=True)
        means[b] = sample.mean()
    lower, upper = np.percentile(means, z_ci)
    return max(0.0, float(lower)), min(1.0, float(upper))

def get_sort_key(library_name):
    """Sort based on the exact index of the library in configuration arrays to ensure Non-RCT grouping at the end."""
    if library_name in developmentLibraryNames:
        return developmentLibraryNames.index(library_name)
    if library_name in evaluationLibraryNames:
        return evaluationLibraryNames.index(library_name)
    return 999

def format_library_name(name):
    """Formats library names, utilizing the custom DISPLAY_NAME_MAP if available."""
    # Check custom name map first
    if name in DISPLAY_NAME_MAP:
        return DISPLAY_NAME_MAP[name]

    # Fallback default formatting
    if name.startswith("non_rct_"):
        name = name[8:]
    return name.replace("_", " ").title().replace(" And ", " and ").replace(" For ", " for ")

def calculate_stats(prediction_file: Path, ground_truth_file: Path, threshold: int, library_name: str):
    """
    Calculates recall, specificity, precision, F1, and accuracy.
    """
    try:
        with open(prediction_file, 'r') as f:
            predictions = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        print(f"  [!] Error loading prediction file: {prediction_file}")
        return None

    try:
        with open(ground_truth_file, 'r') as f:
            ground_truth = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        print(f"  [!] Error loading ground truth file: {ground_truth_file}")
        return None

    tp = 0; fp = 0; tn = 0; fn = 0

    for ref_id, prediction_data in predictions.items():
        if prediction_data == "not screened": continue
        if ref_id not in ground_truth: continue

        try:
            actual_include_val = ground_truth[ref_id].get("include")
            if actual_include_val is None: continue

            if isinstance(actual_include_val, str):
                actual_include = actual_include_val.lower() == 'true'
            else:
                 actual_include = bool(actual_include_val)
        except Exception:
            continue

        predicted_score = None
        if isinstance(prediction_data, dict) and prediction_data:
            try: predicted_score = int(max(prediction_data, key=prediction_data.get))
            except: continue
        elif isinstance(prediction_data, (int, float)):
            predicted_score = int(prediction_data)
        elif isinstance(prediction_data, str):
            try: predicted_score = int(float(prediction_data))
            except: continue
        else:
            continue

        if predicted_score is None: continue

        predicted_include = (predicted_score >= threshold)

        if actual_include and predicted_include: tp += 1
        elif not actual_include and predicted_include: fp += 1
        elif not actual_include and not predicted_include: tn += 1
        elif actual_include and not predicted_include: fn += 1

    actual_positives = tp + fn
    actual_negatives = tn + fp
    total_classified = tp + tn + fp + fn
    predicted_positives = tp + fp

    recall = tp / actual_positives if actual_positives > 0 else 0.0
    specificity = tn / actual_negatives if actual_negatives > 0 else 0.0
    precision = tp / predicted_positives if predicted_positives > 0 else 0.0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / total_classified if total_classified > 0 else 0.0

    # Calculate Wilson 95% Confidence Intervals
    recall_ci = wilson_ci(tp, actual_positives)
    spec_ci = wilson_ci(tn, actual_negatives)

    return {
        'recall': recall, 'recall_ci': recall_ci,
        'specificity': specificity, 'spec_ci': spec_ci,
        'precision': precision, 'f1_score': f1_score,
        'false_negatives': fn, 'false_positives': fp,
        'true_negatives': tn, 'true_positives': tp,
        'total_refs': total_classified,
        'actual_positives': actual_positives,
        'actual_negatives': actual_negatives
    }

# Addresses Reviewer 2 Comment 7: quantify how many true negatives / false positives
# come from references published AFTER the source review was conducted. Such records
# could not have been screened by the original review authors, yet MechaScreener screens
# them and (under our ground-truth rule) treats them as "exclude". We use each review's
# publication year (from STUDY_METADATA_MAP) as a conservative cutoff: a record whose
# publication year is strictly greater than the review's publication year was certainly
# published after the review's search. This reports the percentage of all TN and all FP
# in the evaluation set that are attributable to these post-publication records.
def compute_post_publication_impact(threshold=2):
    """Prints the share of evaluation-set TN and FP contributed by post-publication records."""
    total_tn = total_fp = 0
    post_tn = post_fp = 0

    for lib in evaluationLibraryNames:
        review_year = STUDY_METADATA_MAP.get(lib, {}).get("year")
        if review_year is None:
            continue

        gt_file = GROUND_TRUTH_BASE_DIR / f"{lib}/{GROUND_TRUTH_SUFFIX}"
        pred_file = PREDICTION_BASE_DIR / f"{lib}{PREDICTION_SUFFIX}"
        try:
            with open(gt_file, 'r') as f:
                ground_truth = json.load(f)
            with open(pred_file, 'r') as f:
                predictions = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            continue

        for ref_id, ref_data in ground_truth.items():
            prediction_data = predictions.get(ref_id)
            if prediction_data is None or prediction_data == "not screened":
                continue
            if not isinstance(ref_data, dict):
                continue

            # Only excluded (ground-truth negative) records can be TN or FP
            include_val = str(ref_data.get("include", "")).lower() == "true"
            if include_val:
                continue

            # Resolve predicted score
            score = None
            if isinstance(prediction_data, dict) and prediction_data:
                try: score = int(max(prediction_data, key=prediction_data.get))
                except: continue
            elif isinstance(prediction_data, (int, float)):
                score = int(prediction_data)
            elif isinstance(prediction_data, str):
                try: score = int(float(prediction_data))
                except: continue
            if score is None:
                continue

            is_true_negative = score < threshold
            if is_true_negative:
                total_tn += 1
            else:
                total_fp += 1

            # Determine publication year
            try:
                pub_year = int(str(ref_data.get("year"))[:4])
            except (TypeError, ValueError):
                continue

            if pub_year > review_year:
                if is_true_negative:
                    post_tn += 1
                else:
                    post_fp += 1

    tn_pct = 100 * post_tn / total_tn if total_tn else 0.0
    fp_pct = 100 * post_fp / total_fp if total_fp else 0.0
    print("  -> Post-publication record impact (Reviewer 2 Comment 7):")
    print(f"     TN: {post_tn:,}/{total_tn:,} ({tn_pct:.1f}% of all true negatives)")
    print(f"     FP: {post_fp:,}/{total_fp:,} ({fp_pct:.1f}% of all false positives)")
    return {"post_tn": post_tn, "total_tn": total_tn, "tn_pct": tn_pct,
            "post_fp": post_fp, "total_fp": total_fp, "fp_pct": fp_pct}

def generate_missing_abstracts_table(output_dir):
    """Generates a table splitting RCT and Non-RCT, counting total refs and missing abstracts."""
    rct_rows = []
    non_rct_rows = []

    for lib in evaluationLibraryNames:
        gt_file = GROUND_TRUTH_BASE_DIR / f"{lib}/{GROUND_TRUTH_SUFFIX}"
        total = 0
        missing = 0
        try:
            with open(gt_file, 'r') as f:
                data = json.load(f)
            total = len(data)
            for ref_id, ref_data in data.items():
                if isinstance(ref_data, dict):
                    abstract = ref_data.get("abstract")
                else:
                    abstract = None

                if not abstract or str(abstract).strip() == "":
                    missing += 1
        except Exception:
            pass

        name = format_library_name(lib)

        # Insert citations for evaluation libraries in the missing abstracts table
        if lib in CITATION_MAP:
            name += f"~\\cite{{{CITATION_MAP[lib]}}}"

        row_str = f"{name} & {total:,} & {missing:,} \\\\"

        if lib.startswith("non_rct_"):
            non_rct_rows.append(row_str)
        else:
            rct_rows.append(row_str)

    rct_body = "\n".join(rct_rows)
    non_rct_body = "\n".join(non_rct_rows)

    latex_content = fr"""
\begin{{tabularx}}{{\textwidth}}{{@{{}} X r r @{{}}}}
\toprule
\textbf{{Library Name}} & \textbf{{Total References}} & \textbf{{Missing Abstracts}} \\
\midrule
\multicolumn{{3}}{{c}}{{\textbf{{RCT Studies}}}} \\
\midrule
{rct_body}
\midrule
\multicolumn{{3}}{{c}}{{\textbf{{Non-RCT Studies}}}} \\
\midrule
{non_rct_body}
\bottomrule
\end{{tabularx}}
"""
    output_path = output_dir / "missing_abstracts_table.tex"
    with open(output_path, "w") as f:
        f.write(latex_content.strip())
    print(f"  -> LaTeX missing abstracts table saved to: {output_path}")

# Addresses Reviewer 2 Comment 6: reports the distribution of the 1-5 ordinal
# relevance scores for each evaluation library so readers can see how the score
# is used in practice (rather than assuming it is a calibrated probability).
def generate_score_distribution_table(library_list, output_dir, dataset_name):
    """Generates a LaTeX table of the count of each ordinal score (1-5) per library."""
    rows = []
    for lib in library_list:
        pred_file = PREDICTION_BASE_DIR / f"{lib}{PREDICTION_SUFFIX}"
        counts = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
        try:
            with open(pred_file, 'r') as f:
                predictions = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            continue

        for ref_id, prediction_data in predictions.items():
            if prediction_data == "not screened":
                continue
            score = None
            if isinstance(prediction_data, dict) and prediction_data:
                try: score = int(max(prediction_data, key=prediction_data.get))
                except: continue
            elif isinstance(prediction_data, (int, float)):
                score = int(prediction_data)
            elif isinstance(prediction_data, str):
                try: score = int(float(prediction_data))
                except: continue
            if score in counts:
                counts[score] += 1

        name = format_library_name(lib)
        row = f"{name} & {counts[1]:,} & {counts[2]:,} & {counts[3]:,} & {counts[4]:,} & {counts[5]:,} \\\\"
        rows.append(row)

    table_body = "\n".join(rows)
    latex_content = fr"""
\begin{{tabularx}}{{\textwidth}}{{@{{}} X r r r r r @{{}}}}
\toprule
\textbf{{Library Name}} & \textbf{{Score 1}} & \textbf{{Score 2}} & \textbf{{Score 3}} & \textbf{{Score 4}} & \textbf{{Score 5}} \\
\midrule
{table_body}
\bottomrule
\end{{tabularx}}
"""
    output_path = output_dir / f"score_distribution_{dataset_name.lower()}.tex"
    with open(output_path, "w") as f:
        f.write(latex_content.strip())
    print(f"  -> LaTeX score distribution table saved to: {output_path}")

def generate_latex_table(results_list, output_dir, dataset_name):
    """Generates and saves a LaTeX table based on the results."""

    # Sort results to group RCT vs Non-RCT logically by their order in arrays
    results_list.sort(key=lambda x: get_sort_key(x['library_name']))

    rows = []
    for res in results_list:
        raw_lib_name = res['library_name']
        lib_name = format_library_name(raw_lib_name)

        # Insert citations for development libraries in the results table
        if dataset_name == "Development" and raw_lib_name in CITATION_MAP:
            lib_name += f"~\\cite{{{CITATION_MAP[raw_lib_name]}}}"

        total = res['total_refs']

        # Format: TP/TotalPositives (XX%)
        tp = res['true_positives']
        total_pos = res['actual_positives']
        inc_pct = round((tp / total_pos) * 100) if total_pos > 0 else 0
        inc_col = f"{tp}/{total_pos} ({inc_pct}\\%)"

        # Format: TN/TotalNegatives (XX%)
        tn = res['true_negatives']
        total_neg = res['actual_negatives']
        exc_pct = round((tn / total_neg) * 100) if total_neg > 0 else 0
        exc_col = f"{tn:,}/{total_neg:,} ({exc_pct}\\%)"

        # Construct row
        rows.append(f"{lib_name} & {total:,} & {inc_col} & {exc_col} \\\\")

    # Join rows with newlines
    table_body = "\n".join(rows)

    latex_content = fr"""
\begin{{tabularx}}{{\textwidth}}{{@{{}} X r r r @{{}}}}
\toprule
\textbf{{Library Name}} &
\makecell[b]{{\textbf{{No. of}}\\ \textbf{{Refs}}}} &
\makecell[b]{{\textbf{{Include Studies}}\\ \textbf{{Correctly Included}}}} &
\makecell[b]{{\textbf{{Exclude Studies}}\\ \textbf{{Correctly Excluded}}}} \\
\midrule
{table_body}
\bottomrule
\end{{tabularx}}
"""
    output_path = output_dir / f"results_table_{dataset_name.lower()}.tex"
    with open(output_path, "w") as f: f.write(latex_content.strip())

def generate_statistical_summary_table(results_list, output_dir, dataset_name):
    results_list.sort(key=lambda x: get_sort_key(x['library_name']))
    rows = []

    pooled_tp = pooled_fp = pooled_tn = pooled_fn = 0
    macro_recall, macro_spec = [], []

    for res in results_list:
        lib_name = format_library_name(res['library_name'])
        r, r_ci = res['recall'], res['recall_ci']
        s, s_ci = res['specificity'], res['spec_ci']

        rows.append(f"{lib_name} & {r:.2f} ({r_ci[0]:.2f}-{r_ci[1]:.2f}) & {s:.2f} ({s_ci[0]:.2f}-{s_ci[1]:.2f}) \\\\")

        pooled_tp += res['true_positives']
        pooled_fn += res['false_negatives']
        pooled_fp += res['false_positives']
        pooled_tn += res['true_negatives']
        macro_recall.append(r)
        macro_spec.append(s)

    mean_r, mean_s = np.mean(macro_recall), np.mean(macro_spec)

    micro_total_pos = pooled_tp + pooled_fn
    micro_total_neg = pooled_tn + pooled_fp
    micro_total = micro_total_pos + micro_total_neg

    micro_r = pooled_tp / micro_total_pos if micro_total_pos > 0 else 0
    micro_s = pooled_tn / micro_total_neg if micro_total_neg > 0 else 0

    micro_r_ci = wilson_ci(pooled_tp, micro_total_pos)
    micro_s_ci = wilson_ci(pooled_tn, micro_total_neg)

    # Addresses Reviewer 1 Comment 4 and Reviewer 2 Comment 5:
    # Cluster-aware (between-review) 95% CIs for the macro-average, plus a
    # review-level "zero missed includes" estimand for recall. The latter
    # answers "what is the probability of zero misses in a new review?" by
    # treating each review (not each record) as the unit of analysis.
    n_reviews = len(macro_recall)
    macro_r_ci = cluster_bootstrap_ci(macro_recall)
    macro_s_ci = cluster_bootstrap_ci(macro_spec)
    reviews_zero_miss = sum(1 for r in macro_recall if r >= 1.0)
    zero_miss_ci = wilson_ci(reviews_zero_miss, n_reviews)

    table_body = "\n".join(rows)

    latex_content = fr"""
\begin{{tabularx}}{{\textwidth}}{{@{{}} X c c c @{{}}}}
\toprule
\textbf{{Library Name}} & \textbf{{Recall (95\% CI)}} & \textbf{{Specificity (95\% CI)}} \\
\midrule
{table_body}
\midrule
\textbf{{Macro-Average (per-review mean)}} & \textbf{{{mean_r:.2f} ({macro_r_ci[0]:.2f}-{macro_r_ci[1]:.2f})}} & \textbf{{{mean_s:.2f} ({macro_s_ci[0]:.2f}-{macro_s_ci[1]:.2f})}} \\
\textbf{{Pooled (Micro-Average, per-record)}} & \textbf{{{micro_r:.2f} ({micro_r_ci[0]:.2f}-{micro_r_ci[1]:.2f})}} & \textbf{{{micro_s:.2f} ({micro_s_ci[0]:.2f}-{micro_s_ci[1]:.2f})}} \\
\textbf{{Reviews with zero missed includes}} & \textbf{{{reviews_zero_miss}/{n_reviews} ({zero_miss_ci[0]:.2f}-{zero_miss_ci[1]:.2f})}} & \textbf{{--}} \\
\bottomrule
\end{{tabularx}}
"""
    output_path = output_dir / f"statistical_summary_{dataset_name.lower()}.tex"
    with open(output_path, "w") as f: f.write(latex_content.strip())
    print(f"  -> LaTeX statistical summary table saved to: {output_path}")

def generate_results_figure(results_list, mean_recall, mean_specificity, output_dir, dataset_name):
    """Generates a bar chart showing Recall and Specificity for each library + the overall mean."""
    # Sort results
    sorted_results = sorted(results_list, key=lambda x: get_sort_key(x['library_name']))

    labels = [format_library_name(res['library_name']) for res in sorted_results]
    labels.append('Overall Mean')

    recalls = [res['recall'] for res in sorted_results] + [mean_recall]
    specificities = [res['specificity'] for res in sorted_results] + [mean_specificity]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 7))

    # B&W/Print friendly hatched color styling
    rects1 = ax.bar(x - width/2, recalls, width, label='Recall', color='#f0f0f0', edgecolor='black', hatch='//')
    rects2 = ax.bar(x + width/2, specificities, width, label='Specificity', color='#a0a0a0', edgecolor='black', hatch='\\\\')

    ax.set_ylabel('Score')
    ax.set_title(f'Recall and Specificity by Library ({dataset_name} Dataset)')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right')

    # Make "Overall Mean" bold on x-axis labels
    for tick_label in ax.get_xticklabels():
        if tick_label.get_text() == 'Overall Mean':
            tick_label.set_fontweight('bold')

    # Format graph grid layout
    ax.legend()
    ax.set_ylim([0, 1.35])
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, linestyle='--', alpha=0.7, color='gray')

    # Append 4-decimal place rotated data labels precisely above bars
    def autolabel(rects):
        for i, rect in enumerate(rects):
            height = rect.get_height()

            # Check if this is the last bar ("Overall Mean")
            weight = 'bold' if i == len(rects) - 1 else 'normal'

            ax.annotate(f'{height:.2f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 4),  # 4 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', rotation=90, fontsize=9,
                        fontweight=weight) # Apply the dynamic fontweight here

    autolabel(rects1)
    autolabel(rects2)

    fig.tight_layout()

    output_path = output_dir / f"{dataset_name.lower()}_results_figure.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  -> Figure saved to: {output_path}")

def generate_csv_results(results_list, output_dir, dataset_name):
    if dataset_name.lower() == 'development':
        # Don't generate csv for development results
        return
    """Generates a CSV file for the evaluation metrics based on exact formatting criteria."""
    output_path = output_dir / f"{dataset_name.lower()}_metabayesdta_results.csv"

    with open(output_path, mode='w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)

        # Write case-sensitive columns headers exactly as requested
        writer.writerow(['author', 'year', 'TP', 'FN', 'FP', 'TN'])

        for res in results_list:
            lib_name = res['library_name']

            # Lookup unique author and year from map; default uniquely if not found
            metadata = STUDY_METADATA_MAP.get(lib_name, {"author": f"{format_library_name(lib_name)} Group", "year": 2023})

            author = metadata['author']
            year = metadata['year']
            # Add 1/2 to each cell to prevent failure to converge when FN is 0
            # https://doi.org/10.1002/sim.4780040405
            tp = res['true_positives']
            fn = res['false_negatives']
            fp = res['false_positives']
            tn = res['true_negatives']

            writer.writerow([author, year, tp, fn, fp, tn])

    print(f"  -> CSV table saved to: {output_path}")

# --- Main execution loop ---
print("Starting analysis for all libraries...\n")
generate_missing_abstracts_table(TABLES_OUTPUT_DIR)
compute_post_publication_impact()

# Grouping datasets to loop through them easily
datasets_to_process = {
    "Development": developmentLibraryNames,
    "Evaluation": evaluationLibraryNames
}

for dataset_name, library_list in datasets_to_process.items():
    print(f"=== Processing {dataset_name} Set ===")
    all_results, latex_results = [], []

    for library_name in library_list:
        current_prediction_file = PREDICTION_BASE_DIR / f"{library_name}{PREDICTION_SUFFIX}"
        current_ground_truth_file = GROUND_TRUTH_BASE_DIR / f"{library_name}/{GROUND_TRUTH_SUFFIX}"
        metrics = calculate_stats(current_prediction_file, current_ground_truth_file, 2, library_name)

        if metrics:
            all_results.append(metrics)

            # Add library name to metrics for the table generator
            metrics_with_name = metrics.copy()
            metrics_with_name['library_name'] = library_name
            latex_results.append(metrics_with_name)

            print(f"  [✓] Processed: {library_name}")
        else:
            print(f"  [✗] Skipped: {library_name}")

    # --- Generate LaTeX/CSV Tables and figures for this dataset ---
    if latex_results:
        # Sort results logically for consistent outputs
        latex_results.sort(key=lambda x: get_sort_key(x['library_name']))

        generate_latex_table(latex_results, TABLES_OUTPUT_DIR, dataset_name)
        generate_statistical_summary_table(latex_results, TABLES_OUTPUT_DIR, dataset_name)
        generate_csv_results(latex_results, TABLES_OUTPUT_DIR, dataset_name)

        mean_recall = np.mean([res['recall'] for res in all_results])
        mean_specificity = np.mean([res['specificity'] for res in all_results])

        if dataset_name == "Evaluation":
            generate_results_figure(latex_results, mean_recall, mean_specificity, FIGURES_OUTPUT_DIR, dataset_name)
            # Addresses Reviewer 2 Comment 6: ordinal score distribution for the supplement
            generate_score_distribution_table(library_list, TABLES_OUTPUT_DIR, dataset_name)

    print("\n" + "="*50 + "\n")

print("Analysis finished.")