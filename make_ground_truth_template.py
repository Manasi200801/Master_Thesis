import os, glob, pandas as pd

OUT = "input/ground_truth_template.csv"
os.makedirs("input", exist_ok=True)

rows = []
for p in glob.glob("output/*_results.csv"):
    df = pd.read_csv(p, encoding="utf-8")
    if "model_used" not in df.columns:
        df["model_used"] = os.path.basename(p).replace("_results.csv", "")
    # build a stable row key (id + prompts)
    df["row_key"] = (
        df["id"].astype(str) + "||" +
        df["system_prompt_english"].fillna("").astype(str) + "||" +
        df["user_prompt_english"].fillna("").astype(str)
    )
    rows.append(df[[
        "row_key","id","model_used",
        "system_prompt_english","user_prompt_english",
        "system_prompt_hindi","user_prompt_hindi",
        "english_response","hindi_response"
    ]])

full = pd.concat(rows, ignore_index=True).drop_duplicates("row_key")
full["true_label"] = ""   # fill 1 or 0
full.to_csv(OUT, index=False, encoding="utf-8")
print(f"Template written: {OUT}")
