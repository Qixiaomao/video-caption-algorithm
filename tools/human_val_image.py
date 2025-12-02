import numpy as np
import matplotlib.pyplot as plt

# Categories (questions)
categories = ["Fluency", "Relevance", "Specificity", "Overall Preference"]

# Rating levels 1–5
ratings = [1, 2, 3, 4, 5]

# Percentage data for each category (rows) and rating (columns)
#           1      2      3      4      5
data_pct = np.array([
    [0.0, 0.0, 12.5, 12.5, 75],   # Fluency
    [0.0, 0.0, 12.5, 50, 37.5],   # Relevance
    [0.0, 0.0, 12.5, 50, 37.5],   # Specificity
    [0.0,  12.5,0.0, 37.5, 50],   # Overall Preference
])

# --- Plot grouped bar chart ---
x = np.arange(len(categories))           # positions of each category on x-axis
width = 0.15                             # width of each bar

fig, ax = plt.subplots(figsize=(8, 4))

bars = []
for i, rating in enumerate(ratings):
    # shift each group of bars around the category position
    bar = ax.bar(x + (i - 2) * width,
                 data_pct[:, i],
                 width,
                 label=str(rating))
    bars.append(bar)

# Add percentage labels on top of bars
for bar_group in bars:
    for bar in bar_group:
        height = bar.get_height()
        if height > 0:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height + 1,
                f"{height:.1f}%",
                ha="center",
                va="bottom",
                fontsize=8,
            )

# Axis labels and title (all in English)
ax.set_ylabel("Percentage (%)")
ax.set_xticks(x)
ax.set_xticklabels(categories)
ax.set_ylim(0, 80)  # adjust if you want more/less headroom

# Legend for rating levels 1–5
legend = ax.legend(title="Rating", loc="upper right")

# Make layout tight and save figure
plt.tight_layout()
plt.savefig("human_eval_ratings.png", dpi=300)
plt.show()
