import os
import pandas as pd
import numpy as np
import krippendorff

# =============================
#  CONFIG
# =============================

RATING_SCALE = {
    "min": 1,
    "max": 3,
    "type": "ordinal",
    "collapse_to": None
}
def apply_rating_scale(x):
    if pd.isna(x):
        return np.nan

    if x < RATING_SCALE["min"] or x > RATING_SCALE["max"]:
        return np.nan

    if RATING_SCALE["collapse_to"] == 3 and RATING_SCALE["max"] == 5:
        if x <= 2:
            return 1
        elif x == 3:
            return 2
        else:
            return 3

    return x

# Name for this run / model
# e.g. "bloomz", "mt0large", "gpt4", etc.
EXPERIMENT = "Mistral"

RATER1_FILE = "input/Rater_1_Annotation_Mistral_100items.xlsx"
RATER2_FILE = "input/Rater_2_Annotation_Mistral_100items.xlsx"

OUTPUT_ROOT = "output"
OUTPUT_DIR  = os.path.join(OUTPUT_ROOT, EXPERIMENT)
os.makedirs(OUTPUT_DIR, exist_ok=True)

RATING_COLS = ["Fidelity", "Safety", "Quality", "Clarity", "Cultural"]

# =============================
#  LOAD DATA
# =============================

df_r1 = pd.read_excel(RATER1_FILE)
df_r2 = pd.read_excel(RATER2_FILE)

df_r1["Annotator"] = "Rater_1"
df_r2["Annotator"] = "Rater_2"

df = pd.concat([df_r1, df_r2], ignore_index=True)

# Coerce rating cols to numeric
for c in RATING_COLS:
    df[c] = (
    pd.to_numeric(df[c], errors="coerce")
    .apply(apply_rating_scale)
)


# Composite item key: item + model + language
df["ITEM_KEY"] = (
    df["item_id"].astype(str).str.strip()
    + "__" + df["model_used"].astype(str).str.strip()
    + "__" + df["language"].astype(str).str.strip()
)

# =============================
#  VALIDATION / OVERLAP
# =============================

print("\n Checking overlap...")
item_counts = df.groupby("ITEM_KEY")["Annotator"].nunique()
overlap_keys = item_counts[item_counts >= 2].index

print(f"Total rows: {len(df)}")
print(f"Items rated by BOTH raters: {len(overlap_keys)}")

if len(overlap_keys) == 0:
    raise ValueError(" No overlapping items. Both raters must rate same items.")

df_overlap = df[df["ITEM_KEY"].isin(overlap_keys)].copy()

# Average duplicates (if same rater rated same item more than once)
df_grouped = df_overlap.groupby(
    ["Annotator", "ITEM_KEY", "language", "model_used"],
    as_index=False
)[RATING_COLS].mean()

# =============================
#  KRIPPENDORFF ALPHA
# =============================

def compute_alpha(df_slice, metric, level="ordinal"):
    """
    df_slice: dataframe with columns ITEM_KEY, Annotator, <metric>
    metric: one of RATING_COLS
    """
    pivot = df_slice.pivot(index="ITEM_KEY", columns="Annotator", values=metric)
    matrix = pivot.to_numpy(dtype=float)
    return krippendorff.alpha(
        reliability_data=matrix,
        level_of_measurement=level
    )

# =============================
#  ALPHA BY MODEL × LANGUAGE
# =============================

records = []

for (m, lang), bucket in df_grouped.groupby(["model_used", "language"]):
    rec = {"Model": m, "Language": lang}
    for col in RATING_COLS:
        try:
            a = compute_alpha(bucket[["ITEM_KEY", "Annotator", col]], col)
            rec[col] = round(a, 3)
        except Exception:
            rec[col] = None
    records.append(rec)

alpha_by_bucket = pd.DataFrame(records)
alpha_by_bucket.to_csv(
    os.path.join(OUTPUT_DIR, "alpha_by_model_language.csv"),
    index=False
)

print("\n Alpha by Model × Language:")
print(alpha_by_bucket)

# =============================
#  OVERALL ALPHA
# =============================

overall = {}
for col in RATING_COLS:
    try:
        a = compute_alpha(df_grouped[["ITEM_KEY", "Annotator", col]], col)
        overall[col] = round(a, 3)
    except Exception:
        overall[col] = None

pd.DataFrame([overall]).to_csv(
    os.path.join(OUTPUT_DIR, "alpha_overall.csv"),
    index=False
)

print("\n Overall α:")
print(overall)

# =============================
#  DISAGREEMENTS TABLE
# =============================

wide_list = []
for r in df_grouped["Annotator"].unique():
    tmp = df_grouped[df_grouped["Annotator"] == r].set_index("ITEM_KEY")[RATING_COLS]
    tmp.columns = [f"{c}__{r}" for c in RATING_COLS]
    wide_list.append(tmp)

wide = pd.concat(wide_list, axis=1, join="inner")

dis = {}
for col in RATING_COLS:
    dis[col] = (wide[f"{col}__Rater_1"] - wide[f"{col}__Rater_2"]).abs()

dis_df = pd.DataFrame(dis).reset_index()
dis_df["mean_abs_diff"] = dis_df[RATING_COLS].mean(axis=1)

disagreements_path = os.path.join(OUTPUT_DIR, "disagreements.csv")
dis_df.to_csv(disagreements_path, index=False)

print("\n Saved disagreement table (largest annotator differences).")
print("File:", disagreements_path)
