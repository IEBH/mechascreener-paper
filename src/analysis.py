import json
import os
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

# Prediction filename pattern components
PREDICTION_SUFFIX = ".json"
GROUND_TRUTH_SUFFIX = "references.json"

# --- End Configuration ---

def get_sort_key(library_name):
    """Sort based on the exact index of the library in configuration arrays to ensure Non-RCT grouping at the end."""
    if library_name in developmentLibraryNames:
        return developmentLibraryNames.index(library_name)
    if library_name in evaluationLibraryNames:
        return evaluationLibraryNames.index(library_name)
    return 999

def format_library_name(name):
    """Formats 'snake_case_name' to 'Snake Case Name' and drops 'non_rct_' prefix."""
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
        if isinstance(prediction_data, dict):
            if not prediction_data: continue
            try:
                predicted_score = int(max(prediction_data, key=prediction_data.get))
            except Exception: continue
        elif isinstance(prediction_data, (int, float)):
            predicted_score = int(prediction_data)
        elif isinstance(prediction_data, str):
            try:
                predicted_score = int(float(prediction_data))
            except ValueError: continue
        else:
            continue

        if predicted_score is None: continue

        predicted_include = (predicted_score >= threshold)

        if actual_include and predicted_include: tp += 1
        elif not actual_include and predicted_include: fp += 1
        elif not actual_include and not predicted_include: tn += 1
        elif actual_include and not predicted_include:
            fn += 1

    # Check for division by zero
    actual_positives = tp + fn
    recall = tp / actual_positives if actual_positives > 0 else 0.0

    actual_negatives = tn + fp
    specificity = tn / actual_negatives if actual_negatives > 0 else 0.0

    predicted_positives = tp + fp
    precision = tp / predicted_positives if predicted_positives > 0 else 0.0

    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    total_classified = tp + tn + fp + fn
    accuracy = (tp + tn) / total_classified if total_classified > 0 else 0.0

    return {
        'recall': recall,
        'false_negatives': fn,
        'false_positives': fp,
        'true_negatives': tn,
        'true_positives': tp,
        'specificity': specificity,
        'precision': precision,
        'f1_score': f1_score,
        'accuracy': accuracy,
        'total_refs': total_classified,
        'actual_positives': actual_positives,
        'actual_negatives': actual_negatives
    }

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
% X column expands to fill space, r=right align, c=center align
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

    # Save file using the dataset name (Development / Evaluation)
    filename = f"results_table_{dataset_name.lower()}.tex"
    output_path = output_dir / filename
    with open(output_path, "w") as f:
        f.write(latex_content.strip())

    print(f"  -> LaTeX breakdown table saved to: {output_path}")

def generate_mean_summary_table(results_list, mean_recall, mean_specificity, output_dir, dataset_name):
    """Generates a summary table with recall and specificity for each library, plus the overall mean."""

    # Sort results
    sorted_results = sorted(results_list, key=lambda x: get_sort_key(x['library_name']))

    rows = []
    for res in sorted_results:
        lib_name = format_library_name(res['library_name'])
        rows.append(f"{lib_name} & {res['recall']:.4f} & {res['specificity']:.4f} \\\\")

    table_body = "\n".join(rows)

    latex_content = fr"""
\begin{{tabularx}}{{\textwidth}}{{@{{}} X r r @{{}}}}
\toprule
\textbf{{Library Name}} & \textbf{{Recall}} & \textbf{{Specificity}} \\
\midrule
{table_body}
\midrule
\textbf{{Mean}} & \textbf{{{mean_recall:.4f}}} & \textbf{{{mean_specificity:.4f}}} \\
\bottomrule
\end{{tabularx}}
"""
    filename = f"mean_results_table_{dataset_name.lower()}.tex"
    output_path = output_dir / filename
    with open(output_path, "w") as f:
        f.write(latex_content.strip())

    print(f"  -> LaTeX mean summary table saved to: {output_path}")

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
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.35), ncol=2)
    ax.set_ylim([0, 1.3])
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, linestyle='--', alpha=0.7, color='gray')

    # Append 4-decimal place rotated data labels precisely above bars
    def autolabel(rects):
        for i, rect in enumerate(rects):
            height = rect.get_height()

            # Check if this is the last bar ("Overall Mean")
            weight = 'bold' if i == len(rects) - 1 else 'normal'

            ax.annotate(f'{height:.4f}',
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


# --- Main execution loop ---
print("Starting analysis for all libraries...\n")

# Generate the missing abstracts table
print("=== Generating Context Tables ===")
generate_missing_abstracts_table(TABLES_OUTPUT_DIR)
print("")

# Grouping datasets to loop through them easily
datasets_to_process = {
    "Development": developmentLibraryNames,
    "Evaluation": evaluationLibraryNames
}

for dataset_name, library_list in datasets_to_process.items():
    print(f"=== Processing {dataset_name} Set ===")

    all_results = [] # List to store metrics dictionaries
    latex_results = [] # List specifically for table generation

    for library_name in library_list:
        # Construct paths
        prediction_filename = f"{library_name}{PREDICTION_SUFFIX}"
        ground_truth_filename = f"{library_name}/{GROUND_TRUTH_SUFFIX}"

        current_prediction_file = PREDICTION_BASE_DIR / prediction_filename
        current_ground_truth_file = GROUND_TRUTH_BASE_DIR / ground_truth_filename

        # Calculate metrics
        metrics = calculate_stats(
            prediction_file=current_prediction_file,
            ground_truth_file=current_ground_truth_file,
            threshold=2,
            library_name=library_name
        )

        if metrics:
            all_results.append(metrics)

            # Add library name to metrics for the table generator
            metrics_with_name = metrics.copy()
            metrics_with_name['library_name'] = library_name
            latex_results.append(metrics_with_name)

            print(f"  [✓] Processed: {library_name}")
        else:
            print(f"  [✗] Skipped: {library_name}")

    # --- Generate LaTeX Table for this dataset ---
    if latex_results:
        generate_latex_table(latex_results, TABLES_OUTPUT_DIR, dataset_name)

    # --- Calculate and Print Mean Statistics ---
    if not all_results:
        print(f"\nNo libraries were processed successfully for {dataset_name}.")
    else:
        num_libraries_processed = len(all_results)
        print(f"\n  --- Mean Statistics ({dataset_name} Set, {num_libraries_processed} Libraries) ---")

        mean_recall = np.mean([res['recall'] for res in all_results])
        mean_specificity = np.mean([res['specificity'] for res in all_results])
        mean_precision = np.mean([res['precision'] for res in all_results])
        mean_f1_score = np.mean([res['f1_score'] for res in all_results])
        mean_accuracy = np.mean([res['accuracy'] for res in all_results])

        print(f"  Mean Recall:             {mean_recall:.4f}")
        print(f"  Mean Specificity:        {mean_specificity:.4f}")
        print(f"  Mean Precision:          {mean_precision:.4f}")
        print(f"  Mean F1 Score:           {mean_f1_score:.4f}")
        print(f"  Mean Accuracy:           {mean_accuracy:.4f}")

        # Update table outputs + figures
        generate_mean_summary_table(latex_results, mean_recall, mean_specificity, TABLES_OUTPUT_DIR, dataset_name)
        if dataset_name == "Evaluation":
            generate_results_figure(latex_results, mean_recall, mean_specificity, FIGURES_OUTPUT_DIR, dataset_name)

    print("\n" + "="*50 + "\n")

print("Analysis finished.")