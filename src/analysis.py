import matplotlib.pyplot as plt
import os

# Define output path relative to where you run the script
FIGURES_OUTPUT_DIR = "../paper/figures/"
TABLES_OUTPUT_DIR = "../paper/tables/"
os.makedirs(FIGURES_OUTPUT_DIR, exist_ok=True)
os.makedirs(TABLES_OUTPUT_DIR, exist_ok=True)

# ... Generate plot ...
# plt.plot(x, y)

# Save as PDF (best for LaTeX) or PNG
# plt.savefig(os.path.join(OUTPUT_DIR, "results_graph.pdf"), bbox_inches='tight')