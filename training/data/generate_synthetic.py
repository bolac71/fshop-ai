"""
Generate synthetic moderation training data using Groq API.

Output: training/data/synthetic.json

Run: python training/data/generate_synthetic.py
"""

import json
import os
import random
import time
from pathlib import Path

from dotenv import load_dotenv
from groq import Groq

load_dotenv(Path(__file__).parents[2] / ".env")

DATA_DIR = Path(__file__).parent
client = Groq(api_key=os.environ["GROQ_API_KEY"])

# llama-3.1-8b-instant: ~30k tokens/min (free tier) — 5× cao hơn llama-3.3-70b
MODEL = "llama-3.1-8b-instant"

LABEL_CONFIGS = [
    {
        "label": {"toxic": 1, "spam": 0, "hate_speech": 0},
        "count": 500,
        "instruction": (
            "Generate realistic Vietnamese toxic reviews for a fashion e-commerce site. "
            "Include insults, profanity, or personal attacks against the seller/staff. "
            "NOT a legitimate negative product complaint. Keep each 10-60 words."
        ),
    },
    {
        "label": {"toxic": 0, "spam": 1, "hate_speech": 0},
        "count": 300,
        "instruction": (
            "Generate realistic Vietnamese spam comments for a fashion e-commerce product page. "
            "Examples: competitor ads, unsolicited Zalo/phone numbers, fake giveaway links, "
            "bulk copy-paste promotions. Keep each 10-50 words."
        ),
    },
    {
        "label": {"toxic": 1, "spam": 0, "hate_speech": 1},
        "count": 200,
        "instruction": (
            "Generate realistic Vietnamese hate speech comments in a fashion e-commerce context. "
            "Target ethnicity, religion, gender, or social group. Keep each 10-50 words."
        ),
    },
    {
        "label": {"toxic": 0, "spam": 0, "hate_speech": 0},
        "count": 500,
        "instruction": (
            "Generate realistic NEGATIVE but LEGITIMATE Vietnamese product reviews for fashion items "
            "(clothing, shoes, accessories). Complain about quality, size, shipping, material. "
            "NO profanity, NO personal attacks. Keep each 10-80 words."
        ),
    },
    {
        "label": {"toxic": 0, "spam": 0, "hate_speech": 0},
        "count": 300,
        "instruction": (
            "Generate realistic POSITIVE Vietnamese product reviews for fashion items. "
            "Mention fabric quality, fit, delivery speed, packaging. Keep each 10-80 words."
        ),
    },
]

BATCH_SIZE = 5       # nhỏ hơn để giảm token per request
SLEEP_SEC  = 2.0     # nghỉ giữa các batch để tránh rate limit
MAX_RETRY  = 4       # số lần retry khi bị 429


def generate_batch(instruction: str, n: int) -> list[str]:
    prompt = (
        f"{instruction}\n\n"
        f"Generate exactly {n} examples in Vietnamese.\n"
        f"Return ONLY a valid JSON object with key 'data' containing a list of strings.\n"
        f"Rules: no literal newlines inside strings, no double-quotes inside strings.\n"
        f'Format: {{"data": ["example 1", "example 2"]}}'
    )

    for attempt in range(MAX_RETRY):
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.8,
                max_tokens=1024,
                response_format={"type": "json_object"},
            )
            parsed = json.loads(resp.choices[0].message.content)
            results = parsed.get("data", [])
            if isinstance(results, list):
                return results
            return []

        except json.JSONDecodeError as e:
            print(f"  [WARN] JSON parse error (attempt {attempt+1}): {e}")
            time.sleep(SLEEP_SEC)

        except Exception as e:
            err_str = str(e)
            if "429" in err_str or "rate_limit" in err_str.lower():
                wait = SLEEP_SEC * (2 ** attempt)   # exponential backoff: 2s, 4s, 8s, 16s
                print(f"  [RATE LIMIT] waiting {wait:.0f}s... (attempt {attempt+1}/{MAX_RETRY})")
                time.sleep(wait)
            else:
                print(f"  [ERROR] {e}")
                return []

    print(f"  [SKIP] batch failed after {MAX_RETRY} attempts")
    return []


def main():
    output: list[dict] = []

    for cfg in LABEL_CONFIGS:
        label       = cfg["label"]
        total       = cfg["count"]
        instruction = cfg["instruction"]
        label_name  = next((k for k, v in label.items() if v == 1), "clean")
        print(f"\nGenerating {total} × '{label_name}'...")

        generated = 0
        while generated < total:
            n     = min(BATCH_SIZE, total - generated)
            texts = generate_batch(instruction, n)

            for text in texts:
                if isinstance(text, str) and len(text.strip()) > 5:
                    output.append({"text": text.strip(), **label, "source": "synthetic"})
                    generated += 1

            print(f"  {generated}/{total}", end="\r")
            time.sleep(SLEEP_SEC)

        print(f"  Done: {generated} examples")

    random.shuffle(output)
    out_path = DATA_DIR / "synthetic.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\nTotal: {len(output)} examples → {out_path}")


if __name__ == "__main__":
    main()
