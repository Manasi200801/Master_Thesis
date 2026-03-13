# translation.py (FULL FILE)

from pathlib import Path
import pandas as pd
import torch

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer, util   

# --- Models (free, open-source) ---
_EN_HI = "Helsinki-NLP/opus-mt-en-hi"
_HI_EN = "Helsinki-NLP/opus-mt-hi-en"
_SBERT = "sentence-transformers/paraphrase-MiniLM-L6-v2"

# --- Device ---
_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- Load once -----
_enhi_tok = AutoTokenizer.from_pretrained(_EN_HI)
_enhi_mod = AutoModelForSeq2SeqLM.from_pretrained(_EN_HI).to(_DEVICE)

_hien_tok = AutoTokenizer.from_pretrained(_HI_EN)
_hien_mod = AutoModelForSeq2SeqLM.from_pretrained(_HI_EN).to(_DEVICE)

_sbert = SentenceTransformer(_SBERT)  # SBERT can run on CPU fine for small datasets


def _translate(text: str, tok, model, max_len: int = 128) -> str:
    """Safe single-string translation with truncation and device support."""
    if text is None:
        return ""
    text = str(text).strip()
    if text == "":
        return ""

    inputs = tok(text, return_tensors="pt", truncation=True).to(_DEVICE)
    with torch.no_grad():
        out_ids = model.generate(**inputs, max_length=max_len)
    return tok.decode(out_ids[0], skip_special_tokens=True)


def translate_to_hindi(text: str, max_len: int = 128) -> str:
    return _translate(text, _enhi_tok, _enhi_mod, max_len)


def translate_to_english(text: str, max_len: int = 128) -> str:
    return _translate(text, _hien_tok, _hien_mod, max_len)


def cosine_sim_batch(a_list, b_list) -> list:
    """Compute SBERT cosine similarity for lists (faster + consistent)."""
    a_list = ["" if x is None else str(x) for x in a_list]
    b_list = ["" if x is None else str(x) for x in b_list]

    emb_a = _sbert.encode(a_list, convert_to_tensor=True, show_progress_bar=False)
    emb_b = _sbert.encode(b_list, convert_to_tensor=True, show_progress_bar=False)

    # cosine similarity per pair
    sims = util.cos_sim(emb_a, emb_b).diagonal()
    return [float(x) for x in sims]


def build_prompts_with_validation(
    in_csv: str,
    out_csv: str,
    max_len: int = 128,
    sim_threshold: float = 0.85,
    print_flagged: int = 10,
) -> None:
    """
    Reads English prompts from in_csv with columns:
      id, category, prompt_en

    Writes out_csv with columns:
      id, category, prompt_en, prompt_hi, back_en, similarity, flag_low_similarity

    Validation:
      flag_low_similarity = 1 if similarity < sim_threshold else 0
    """
    in_path = Path(in_csv)
    out_path = Path(out_csv)

    if not in_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {in_path.resolve()}")

    df = pd.read_csv(in_path)

    required = {"id", "category", "prompt_en"}
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(f"Missing required columns {missing}. Found: {list(df.columns)}")

    # Translate EN -> HI
    df["prompt_hi"] = df["prompt_en"].map(lambda x: translate_to_hindi(x, max_len=max_len))

    # Back-translate HI -> EN
    df["back_en"] = df["prompt_hi"].map(lambda x: translate_to_english(x, max_len=max_len))

    # Similarity EN vs back-EN (SBERT)
    df["similarity"] = cosine_sim_batch(df["prompt_en"].tolist(), df["back_en"].tolist())
    df["similarity"] = df["similarity"].round(4)

    # Flag low similarity for manual review
    df["flag_low_similarity"] = (df["similarity"] < sim_threshold).astype(int)

    # Sort: flagged first, then lowest similarity
    df = df.sort_values(["flag_low_similarity", "similarity"], ascending=[False, True])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False, encoding="utf-8")
    print(f"[OK] Saved {len(df)} rows to {out_path.resolve()}")
    print(f"[INFO] Similarity threshold: {sim_threshold}")
    print(f"[INFO] Flagged rows: {int(df['flag_low_similarity'].sum())} / {len(df)}")

    if print_flagged and df["flag_low_similarity"].sum() > 0:
        print("\n[FLAGGED EXAMPLES]")
        show = df[df["flag_low_similarity"] == 1].head(int(print_flagged))
        for _, r in show.iterrows():
            print(f"\nID: {r['id']} | category: {r['category']} | sim={r['similarity']}")
            print("EN:", r["prompt_en"])
            print("HI:", r["prompt_hi"])
            print("Back-EN:", r["back_en"])


if __name__ == "__main__":
    # Example run (edit paths as needed)
    build_prompts_with_validation(
        in_csv="input/prompts_30.csv",
        out_csv="output/prompts_30_translated_validated.csv",
        max_len=128,
        sim_threshold=0.85,
        print_flagged=10,
    )