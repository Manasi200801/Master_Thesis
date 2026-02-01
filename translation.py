# translation.py  (replace everything with this)

from pathlib import Path
import pandas as pd

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer, util

# --- Models (free, open-source) ---
_EN_HI = "Helsinki-NLP/opus-mt-en-hi"
_HI_EN = "Helsinki-NLP/opus-mt-hi-en"

_SBERT = "sentence-transformers/paraphrase-MiniLM-L6-v2"

# --- Load once ---
_enhi_tok = AutoTokenizer.from_pretrained(_EN_HI)
_enhi_mod = AutoModelForSeq2SeqLM.from_pretrained(_EN_HI)

_hien_tok = AutoTokenizer.from_pretrained(_HI_EN)
_hien_mod = AutoModelForSeq2SeqLM.from_pretrained(_HI_EN)

_sbert = SentenceTransformer(_SBERT)


def _translate(text: str, tok, model, max_len: int = 128) -> str:
    out_ids = model.generate(**tok(text, return_tensors="pt"), max_length=max_len)
    return tok.decode(out_ids[0], skip_special_tokens=True)


def translate_to_hindi(text: str, max_len: int = 128) -> str:
    return _translate(text, _enhi_tok, _enhi_mod, max_len)


def translate_to_english(text: str, max_len: int = 128) -> str:
    return _translate(text, _hien_tok, _hien_mod, max_len)


def cosine_sim(a: str, b: str) -> float:
    ea = _sbert.encode(a, convert_to_tensor=True)
    eb = _sbert.encode(b, convert_to_tensor=True)
    return float(util.cos_sim(ea, eb).item())


def build_prompts_with_validation(in_csv: str, out_csv: str) -> None:
    """
    Reads English prompts from in_csv with columns:
      id, category, prompt_en
    Writes out_csv with columns:
      id, category, prompt_en, prompt_hi, back_en, similarity
    """
    in_path = Path(in_csv)
    out_path = Path(out_csv)
    df = pd.read_csv(in_path)

    df["prompt_hi"] = df["prompt_en"].map(translate_to_hindi)
    df["back_en"]   = df["prompt_hi"].map(translate_to_english)
    df["similarity"] = [
        cosine_sim(e, b) for e, b in zip(df["prompt_en"], df["back_en"])
    ]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False, encoding="utf-8")
    print(f"[OK] Saved {len(df)} rows to {out_path}")
