import os
import time
import pandas as pd
import torch

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer, util

from translation import translate_to_hindi, translate_to_english
from utils import load_prompts, random_score


# ===============================
# CONFIG
# ===============================
# MODEL_NAME = "google/mt5-base"          # Seq2Seq
MODEL_NAME = "bigscience/mt0-large"   # Seq2Seq (heavy)
# MODEL_NAME = "bigscience/bloomz-560m" # Causal LM

INPUT_FILE = "input/input.csv"

OUTPUT_DIR = "output"

MAX_NEW_TOKENS = 256
MAX_INPUT_TOKENS = 512

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ===============================
# MODEL TYPE
# ===============================
def is_seq2seq(model_name: str) -> bool:
    name = model_name.lower()
    return ("mt5" in name) or ("mt0" in name) or ("t5" in name)


# ===============================
# INIT (HF)
# ===============================
print("RUNNING:", __file__)
print(f"🔹 Loading {MODEL_NAME} on {DEVICE} ...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

if is_seq2seq(MODEL_NAME):
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
else:
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    # bloomz sometimes needs pad_token_id set
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

model.to(DEVICE)
model.eval()

embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
print(f"✅ Model ready. seq2seq={is_seq2seq(MODEL_NAME)}")


# ===============================
# GENERATION
# ===============================
def generate_response(prompt: str) -> str:
    if not prompt:
        return ""

    try:
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=MAX_INPUT_TOKENS
        ).to(DEVICE)

        gen_kwargs = dict(
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False
        )

        # causal models often want pad_token_id
        if not is_seq2seq(MODEL_NAME) and tokenizer.pad_token_id is not None:
            gen_kwargs["pad_token_id"] = tokenizer.pad_token_id

        with torch.no_grad():
            output_ids = model.generate(**inputs, **gen_kwargs)

        return tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()

    except Exception as e:
        print(f"Gen error: {e}")
        return ""


# ===============================
# SIMILARITY
# ===============================
def similarity_score(a, b):
    if not a or not b:
        return 0.0
    e1 = embedder.encode(a, convert_to_tensor=True)
    e2 = embedder.encode(b, convert_to_tensor=True)
    return round(util.cos_sim(e1, e2).item(), 5)


# ===============================
# MAIN
# ===============================
def main():
    df = load_prompts(INPUT_FILE)
    print(f"Running {len(df)} prompts on {MODEL_NAME}...")

    results = []

    for i, row in df.iterrows():
        en = str(row["prompt_en"])
        hi = translate_to_hindi(en)

        en_resp = generate_response(en)
        hi_resp = generate_response(hi)

        en_hi = translate_to_hindi(en_resp) if en_resp else ""
        hi_en = translate_to_english(hi_resp) if hi_resp else ""

        results.append({
            "id": row.get("id", i + 1),
            "category": row.get("category", ""),
            "prompt_en": en,
            "prompt_hi": hi,
            "english_response": en_resp,
            "hindi_response": hi_resp,
            "english_response_translated_to_hindi": en_hi,
            "hindi_response_translated_to_english": hi_en,
            "similarity_en_hiTrans": similarity_score(en_resp, hi_en),
            "similarity_hi_enTrans": similarity_score(hi_resp, en_hi),
            "accuracy_score": random_score(),
            "safety_score": random_score(),
            "clarity_score": random_score(),
            "cultural_score": random_score(),
            "model_used": MODEL_NAME,
        })

        print(f"{i+1}/{len(df)}")
        # optional throttle if you want
        # time.sleep(0.1)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(
        OUTPUT_DIR,
        f"{MODEL_NAME.split('/')[-1].replace('-', '_')}_results.csv"
    )
    pd.DataFrame(results).to_csv(out_path, index=False, encoding="utf-8")
    print(f"\nSaved {len(results)} results → {out_path}")


if __name__ == "__main__":
    main()
