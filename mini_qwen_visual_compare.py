
import csv
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
from datasets import load_from_disk

DATASET_PATH = "/Users/nnn/PycharmProjects/MLScratch/flickr8k_local"

CHECKPOINTS = {
    "Original": "mini_qwen_vl",
    "VL Best": "mini_qwen_vl_best",
    "Ranking Best": "mini_qwen_vl_ranking_best",
}

N_IMAGES = 8
OUTPUT_PATH = Path("results/generation/qualitative_comparison.png")

dataset = load_from_disk(DATASET_PATH)
test_dataset = dataset["test"]

all_results = {}

for name, folder in CHECKPOINTS.items():
    csv_path = Path("results/generation") / folder / "generation_results.csv"

    with open(csv_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    all_results[name] = {
        int(row["image_index"]): row
        for row in rows
    }

# Same image indices across all checkpoints
indices = list(range(N_IMAGES))

fig, axes = plt.subplots(
    N_IMAGES, 2,
    figsize=(15, N_IMAGES * 3.1),
    gridspec_kw={"width_ratios": [1, 2.3]}
)

for row_idx, image_idx in enumerate(indices):
    sample = test_dataset[image_idx]

    ax_image = axes[row_idx, 0]
    ax_text = axes[row_idx, 1]

    ax_image.imshow(sample["image"].convert("RGB"))
    ax_image.axis("off")
    ax_image.set_title(f"Test image {image_idx + 1}", fontsize=11)

    lines = [
        "REFERENCE:",
        textwrap.fill(sample["caption_0"], width=75),
        ""
    ]

    for name in CHECKPOINTS:
        caption = all_results[name][image_idx]["correct_generation"]
        lines.append(f"{name.upper()}:")
        lines.append(textwrap.fill(caption, width=75))
        lines.append("")

    ax_text.text(
        0.02, 0.5,
        "\n".join(lines),
        va="center",
        ha="left",
        fontsize=10,
        family="monospace",
        transform=ax_text.transAxes
    )
    ax_text.axis("off")

plt.tight_layout()

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(OUTPUT_PATH, dpi=180, bbox_inches="tight")
plt.show()

print("Saved:", OUTPUT_PATH)
