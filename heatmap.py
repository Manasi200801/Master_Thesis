import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

DIMENSIONS = ["Fidelity", "Safety", "Quality", "Clarity", "Cultural"]
LANG_ORDER = ["EN", "HI"]

def main():
    print("\n=== Combined heatmap started ===")

    # ---------- Load Excel ----------
    xlsx_path = "alpha_long.xlsx"
    if not os.path.exists(xlsx_path):
        raise FileNotFoundError("alpha_long.xlsx not found next to heatmap.py")

    df = pd.read_excel(xlsx_path)
    print("[INFO] Loaded rows:", len(df))

    # ---------- Normalize ----------
    df.columns = [c.lower().strip() for c in df.columns]
    df["model"] = df["model"].astype(str).str.strip()
    df["language"] = df["language"].astype(str).str.upper().str.strip()
    df["dimension"] = df["dimension"].astype(str).str.strip()
    df["alpha"] = pd.to_numeric(df["alpha"], errors="coerce")

    df = df.dropna(subset=["alpha"])

    print("[INFO] Models:", sorted(df["model"].unique()))

    # ---------- Create Model_Language row label ----------
    df["model_lang"] = df["model"] + "_" + df["language"]

    # Sort rows: model first, EN before HI
    model_order = sorted(df["model"].unique())
    row_order = []
    for m in model_order:
        for l in LANG_ORDER:
            if ((df["model"] == m) & (df["language"] == l)).any():
                row_order.append(f"{m}_{l}")

    # ---------- Pivot to matrix ----------
    mat = (
        df.pivot_table(
            index="model_lang",
            columns="dimension",
            values="alpha",
            aggfunc="mean"
        )
        .reindex(index=row_order, columns=DIMENSIONS)
    )

    data = mat.values.astype(float)

    # ---------- Plot ----------
    fig, ax = plt.subplots(figsize=(10, 0.6 * len(row_order) + 2))

    vmin = np.nanmin(data)
    vmax = np.nanmax(data)

    im = ax.imshow(data, aspect="auto", vmin=vmin, vmax=vmax)

    ax.set_xticks(np.arange(len(DIMENSIONS)))
    ax.set_xticklabels(DIMENSIONS, rotation=30, ha="right")

    ax.set_yticks(np.arange(len(row_order)))
    ax.set_yticklabels(row_order)

    ax.set_title("Krippendorff’s Alpha Heatmap (All Models × Languages × Dimensions)")

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Krippendorff’s Alpha")

    # ---------- Annotate values ----------
    for i in range(len(row_order)):
        for j in range(len(DIMENSIONS)):
            val = data[i, j]
            if np.isfinite(val):
                ax.text(j, i, f"{val:.3f}", ha="center", va="center", fontsize=8)

    fig.tight_layout()

    # ---------- Save ----------
    os.makedirs("figures", exist_ok=True)
    fig.savefig("figures/alpha_heatmap_all_models.pdf")
    fig.savefig("figures/alpha_heatmap_all_models.png", dpi=300)
    plt.close(fig)

    print("[OK] Saved figures/alpha_heatmap_all_models.pdf")
    print("\n=== DONE ===")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("[ERROR]", repr(e))
        sys.exit(1)
