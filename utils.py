import pandas as pd
import random

def load_prompts(csv_path="input/input.csv"):
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()
    if "prompt_en" not in df.columns:
        raise ValueError("CSV must contain a 'prompt_en' column.")
    if "id" not in df.columns:
        df.insert(0, "id", range(1, len(df) + 1))
    if "category" not in df.columns:
        df["category"] = "neutral"
    print(f" Loaded {len(df)} prompts from {csv_path}")
    return df

def random_score():
    return round(random.uniform(0, 1), 2)
