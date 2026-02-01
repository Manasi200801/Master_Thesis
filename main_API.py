import os
import time
import argparse
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from openai import OpenAI
from mistralai import Mistral

from sentence_transformers import SentenceTransformer, util
from translation import translate_to_hindi, translate_to_english
from utils import load_prompts, random_score


# ===============================
# ARGS
# ===============================
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="gpt-4o",
                   help="OpenAI: gpt-* | Mistral API: mistral-* / mixtral-*")
    p.add_argument("--input", default="input/input.csv")
    p.add_argument("--output_dir", default="output")
    p.add_argument("--max_tokens", type=int, default=256)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--sleep", type=float, default=0.0)
    p.add_argument("--translate_max_chars", type=int, default=1200)
    p.add_argument("--disable_translation", action="store_true")
    return p.parse_args()


# ===============================
# PROVIDER ROUTING
# ===============================
def is_openai_model(name: str) -> bool:
    return name.startswith("gpt")

def is_mistral_model(name: str) -> bool:
    return name.startswith("mistral") or name.startswith("mixtral")


# ===============================
# ENV + CLIENTS
# ===============================
def load_env_local():
    env_path = Path(__file__).with_name(".env")
    load_dotenv(dotenv_path=env_path, override=True)
    return env_path

def build_clients():
    openai_key = os.getenv("OPENAI_API_KEY")
    mistral_key = os.getenv("MISTRAL_API_KEY")

    openai_client = OpenAI(api_key=openai_key) if openai_key else None
    mistral_client = Mistral(api_key=mistral_key) if mistral_key else None
    return openai_client, mistral_client


# ===============================
# HELPERS
# ===============================
def clip_text(s: str, max_chars: int) -> str:
    if not s:
        return ""
    return str(s)[:max_chars]

def safe_translate(fn, text: str) -> str:
    try:
        return fn(text)
    except Exception as e:
        print(f"Translation error: {e}")
        return ""


# ===============================
# MODEL CALL
# ===============================
def generate_response(prompt, model_name, openai_client, mistral_client, max_tokens, temperature):
    try:
        if is_openai_model(model_name):
            resp = openai_client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=temperature,
            )
            return resp.choices[0].message.content.strip()

        if is_mistral_model(model_name):
            resp = mistral_client.chat.complete(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=temperature,
            )
            return resp.choices[0].message.content.strip()

        raise ValueError("Unknown model")

    except Exception as e:
        print(f"Gen error ({model_name}): {e}")
        return ""


# ===============================
# SIMILARITY
# ===============================
def similarity_score(a, b, embedder):
    if not a or not b:
        return 0.0
    e1 = embedder.encode(a, convert_to_tensor=True)
    e2 = embedder.encode(b, convert_to_tensor=True)
    return round(util.cos_sim(e1, e2).item(), 3)


# ===============================
# MAIN
# ===============================
def main():
    args = parse_args()
    model_name = args.model

    env_path = load_env_local()
    openai_client, mistral_client = build_clients()

    if is_openai_model(model_name) and openai_client is None:
        raise RuntimeError(f"OPENAI_API_KEY missing in {env_path}")
    if is_mistral_model(model_name) and mistral_client is None:
        raise RuntimeError(f"MISTRAL_API_KEY missing in {env_path}")

    print("RUNNING:", __file__)
    print(f"MODEL:", model_name)

    embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    df = load_prompts(args.input)
    print(f"Loaded {len(df)} prompts")

    results = []

    for i, row in df.iterrows():
        en = str(row["prompt_en"])

        hi = "" if args.disable_translation else safe_translate(
            translate_to_hindi, clip_text(en, args.translate_max_chars)
        )

        en_resp = generate_response(
            en, model_name, openai_client, mistral_client,
            args.max_tokens, args.temperature
        )

        hi_resp = ""
        if hi:
            hi_resp = generate_response(
                hi, model_name, openai_client, mistral_client,
                args.max_tokens, args.temperature
            )

        en_hi = "" if args.disable_translation else safe_translate(
            translate_to_hindi, clip_text(en_resp, args.translate_max_chars)
        )

        hi_en = "" if args.disable_translation else safe_translate(
            translate_to_english, clip_text(hi_resp, args.translate_max_chars)
        )

        results.append({
            "id": row.get("id", i + 1),
            "category": row.get("category", ""),
            "prompt_en": en,
            "prompt_hi": hi,
            "english_response": en_resp,
            "hindi_response": hi_resp,
            "english_response_translated_to_hindi": en_hi,
            "hindi_response_translated_to_english": hi_en,
            "similarity_en_hiTrans": similarity_score(en_resp, hi_en, embedder),
            "similarity_hi_enTrans": similarity_score(hi_resp, en_hi, embedder),
            "accuracy_score": random_score(),
            "safety_score": random_score(),
            "clarity_score": random_score(),
            "cultural_score": random_score(),
            "model_used": model_name,
        })

        print(f"{i+1}/{len(df)}")
        if args.sleep > 0:
            time.sleep(args.sleep)

    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(
        args.output_dir,
        f"{model_name.replace('-', '_')}_results.csv"
    )
    pd.DataFrame(results).to_csv(out_path, index=False, encoding="utf-8")
    print("Saved →", out_path)


if __name__ == "__main__":
    main()
