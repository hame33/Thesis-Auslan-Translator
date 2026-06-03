"""
back_translate.py
Generates synthetic Auslan gloss annotations from English subtitles
using the Anthropic API (Claude Haiku). Outputs TSV files ready to
feed directly into train_text2gloss.py.
"""

import os
import time
import random
import pandas as pd
import anthropic
from pathlib import Path

# ── Config ─────────────────────────────────────────────────────
AUSLAN_DAILY_PATH  = "/Users/hamishdawson/Desktop/Thesis/Auslan-Daily_Communication.xlsx"
AUSLAN_DAILY_TEXT_COL = "Subtitle"

# Anthropic API key — or set env var ANTHROPIC_API_KEY instead
API_KEY = "Key"

OUTPUT_DIR = "/Users/hamishdawson/Desktop/Thesis/BacktranslationClaude"

# Train/dev/test split ratios
TRAIN_RATIO = 0.90
DEV_RATIO   = 0.05
# TEST_RATIO  = 0.05 (remainder)

# How many subtitles to send in a single API call (batching saves cost)
BATCH_SIZE = 20

# Delay between API calls in seconds (avoids rate limiting)
REQUEST_DELAY = 0.5

# Model to use — haiku is fast and cheap, good for annotation tasks
MODEL = "claude-haiku-4-5-20251001"
# ───────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an expert Auslan (Australian Sign Language) corpus linguist trained in the Auslan Corpus Annotation Guidelines (Johnston, 2024, v2.0).

Your task is to convert English sentences into Auslan ID-gloss sequences. Follow these conventions precisely:

━━━ CONVENTIONAL LEXICAL SIGNS ━━━
- Write all ID-glosses in UPPERCASE (e.g. HOUSE, GIVE, FINISH)
- If a sign needs more than one English word, join with hyphens: WRONG-MIND, ADOPT-TAKE
- If two signs share the same English word, distinguish with an underscore hint: FINISH_GOOD, FINISH_FIVE, WHO_NTH
- Compounds (lexicalised multi-sign units) are joined with hyphens: BREAKFAST (from EAT+MORNING)
- Negation incorporated into a sign uses -NOT suffix: HAVE-NOT, WANT-NOT, WILL-NOT
- Standalone negators: NOT, NOTHING, NO-WAY, BAN, DO-NOT
- Proper name signs: NS_NAME, e.g. NS_PETER, NS_SALLY
- Signed English borrowings: GLOSS_SE, e.g. GAVE_SE
- Foreign SL borrowings: GLOSS_ASL, e.g. COOL_ASL

━━━ NUMBERS ━━━
- Spell out numbers as words: NINETEEN-EIGHTY-SEVEN (not 1987)
- Number incorporation uses hash: YEAR-AGO#2, O'CLOCK#2, AGE-IN-YEARS#14

━━━ POINTING SIGNS (PT_) ━━━
General form: PT_FUNCTION(PERSON)(NUMBER)
Person values: 1=signer(I/me), 2=addressee(you), 3=third party/nearby, 3distal=far/imaginary
Number: add PL for plural referents
- Personal pronouns: PT_PRO1 (I/me), PT_PRO2 (you), PT_PRO3 (he/she/it), PT_PRO3PL (they)
- Possessive pronouns: PT_POSS1 (my/mine), PT_POSS2 (your/yours), PT_POSS3 (his/her/its)
- Reflexive pronouns: PT_REFL1 (myself), PT_REFL2 (yourself), PT_REFL3 (himself/herself)
- Demonstrative pronouns: PT_DEM3 (this/that — with eye gaze + movement stress)
- Definite determiner (article-like): PT_DET
- Locative point: PT_LOC (points to location in signing space)
- If function is genuinely ambiguous: PT_LOC/DET/PRO3
- Body part points: PT_BODY(BODYPART), e.g. PT_BODY(SHOULDER)

━━━ DEPICTING SIGNS (DS_) ━━━
Used for movement, location, size/shape depictions (classifier-like signs):
- Minimal form: DS or DS(HANDSHAPE), e.g. DS(FLAT), DS(C)
- Movement depiction: DSM(HANDSHAPE), e.g. DSM(C) = circular object moves
- Location depiction: DSL(HANDSHAPE)
- Size/shape depiction: DSS(HANDSHAPE)
- Ground depiction: DSG(HANDSHAPE)

━━━ AUSLAN GRAMMAR ━━━
- Auslan is Topic-Comment, not Subject-Verb-Object. Topic goes first.
- Drop English articles (a, an, the) — Auslan uses PT_DET instead when needed
- Drop English copulas (is, are, was, were, am) unless a specific sign exists
- Drop English auxiliaries (do, does, did, will, would, can, could, should) unless a specific Auslan sign exists
- Time signs go at the START of a clause: TODAY, YESTERDAY, TOMORROW, NOW, BEFORE, FUTURE, RECENTLY
- Negation: place NOT, NOTHING, NO-WAY after the verb/predicate, or use negative incorporation (-NOT)
- Yes/no questions: raise eyebrows (not marked in gloss, but question structure is implied)
- Wh-questions: wh-sign goes at END: WHO, WHAT, WHERE, WHEN, WHY, HOW
- Plurality: precede noun with number or MANY when needed; or use plural pointing PT_PRO3PL
- Adjectives typically follow nouns in Auslan

━━━ OUTPUT FORMAT ━━━
Return ONLY a numbered list, one gloss sequence per line, exactly matching the number of input sentences:
1. GLOSS SEQUENCE
2. GLOSS SEQUENCE

Do not include the original English, explanations, brackets around the whole line, or any other text. Only output the numbered gloss lines."""


def make_prompt(sentences: list[str]) -> str:
    numbered = "\n".join(f"{i+1}. {s}" for i, s in enumerate(sentences))
    return f"Convert these English sentences to Auslan gloss notation:\n\n{numbered}"


def parse_response(text: str, expected_count: int) -> list[str]:
    """Extract gloss lines from the model response."""
    lines = [l.strip() for l in text.strip().splitlines() if l.strip()]
    glosses = []
    for line in lines:
        # Strip leading "1. " / "1) " etc
        if line and line[0].isdigit():
            parts = line.split('.', 1) if '.' in line else line.split(')', 1)
            gloss = parts[1].strip() if len(parts) > 1 else line
            glosses.append(gloss)
    # Pad or trim to match expected count
    while len(glosses) < expected_count:
        glosses.append("")
    return glosses[:expected_count]


def generate_glosses(client: anthropic.Anthropic, sentences: list[str]) -> list[str]:
    """Call the API for a batch of sentences."""
    message = client.messages.create(
        model=MODEL,
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": make_prompt(sentences)}],
    )
    return parse_response(message.content[0].text, len(sentences))


def main():
    # Load data
    print(f"Loading {AUSLAN_DAILY_PATH}...")
    if AUSLAN_DAILY_PATH.endswith(('.xlsx', '.xls')):
        df = pd.read_excel(AUSLAN_DAILY_PATH)
    else:
        df = pd.read_csv(AUSLAN_DAILY_PATH)

    assert AUSLAN_DAILY_TEXT_COL in df.columns, \
        f"Column '{AUSLAN_DAILY_TEXT_COL}' not found. Got: {df.columns.tolist()}"

    sentences = df[AUSLAN_DAILY_TEXT_COL].fillna("").astype(str).tolist()
    print(f"Loaded {len(sentences)} sentences.")

    # Set up output dir and checkpoint file
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    checkpoint_path = os.path.join(OUTPUT_DIR, "checkpoint.tsv")

    # Load checkpoint if it exists (so we can resume if interrupted)
    if os.path.exists(checkpoint_path):
        done_df = pd.read_csv(checkpoint_path, sep="\t")
        done_count = len(done_df)
        pairs = done_df.to_dict('records')
        print(f"Resuming from checkpoint: {done_count} rows already done.")
    else:
        pairs = []
        done_count = 0

    # Set up API client
    client = anthropic.Anthropic(api_key=API_KEY)

    # Process in batches
    total = len(sentences)
    sentences_to_do = sentences[done_count:]
    batches = [sentences_to_do[i:i+BATCH_SIZE] for i in range(0, len(sentences_to_do), BATCH_SIZE)]

    print(f"Processing {len(sentences_to_do)} remaining sentences in {len(batches)} batches...")

    for batch_idx, batch in enumerate(batches):
        try:
            glosses = generate_glosses(client, batch)
            for text, gloss in zip(batch, glosses):
                pairs.append({"text": text, "gloss": gloss})

            # Save checkpoint every 10 batches
            if (batch_idx + 1) % 10 == 0:
                pd.DataFrame(pairs).to_csv(checkpoint_path, sep="\t", index=False)
                done = done_count + (batch_idx + 1) * BATCH_SIZE
                print(f"  [{min(done, total)}/{total}] Checkpoint saved.")

            time.sleep(REQUEST_DELAY)

        except anthropic.RateLimitError:
            print("  Rate limited — waiting 60 seconds...")
            time.sleep(60)
            # Retry this batch
            glosses = generate_glosses(client, batch)
            for text, gloss in zip(batch, glosses):
                pairs.append({"text": text, "gloss": gloss})

        except Exception as e:
            print(f"  Error on batch {batch_idx}: {e} — skipping.")
            for text in batch:
                pairs.append({"text": text, "gloss": ""})

    # Final checkpoint save
    pd.DataFrame(pairs).to_csv(checkpoint_path, sep="\t", index=False)
    print(f"\nAll {len(pairs)} pairs generated.")

    # ── Write gloss column back into the original Excel ────────────────────────
    glosses_in_order = [p["gloss"] for p in pairs]
    # Pad to full length in case any rows were skipped
    while len(glosses_in_order) < len(df):
        glosses_in_order.append("")
    df["gloss"] = glosses_in_order[:len(df)]

    excel_out = AUSLAN_DAILY_PATH.rsplit('.', 1)[0] + "_with_gloss.xlsx"
    df.to_excel(excel_out, index=False)
    print(f"Annotated Excel saved to: {excel_out}")
    print(df[[AUSLAN_DAILY_TEXT_COL, "gloss"]].head(10).to_string())

    # ── Also save train/dev/test TSVs for train_text2gloss.py ─────────────────
    full_df = pd.DataFrame(pairs)
    full_df = full_df[full_df["gloss"].str.strip() != ""].reset_index(drop=True)
    print(f"\n{len(full_df)} non-empty pairs available for training.")

    full_df = full_df.sample(frac=1, random_state=42).reset_index(drop=True)
    n = len(full_df)
    n_train = int(n * TRAIN_RATIO)
    n_dev   = int(n * DEV_RATIO)

    train_df = full_df.iloc[:n_train]
    dev_df   = full_df.iloc[n_train:n_train + n_dev]
    test_df  = full_df.iloc[n_train + n_dev:]

    train_path = os.path.join(OUTPUT_DIR, "auslan_gloss_text_train.tsv")
    dev_path   = os.path.join(OUTPUT_DIR, "auslan_gloss_text_dev.tsv")
    test_path  = os.path.join(OUTPUT_DIR, "auslan_gloss_text_test.tsv")

    train_df.to_csv(train_path, sep="\t", index=False)
    dev_df.to_csv(dev_path,     sep="\t", index=False)
    test_df.to_csv(test_path,   sep="\t", index=False)

    print(f"Split: {len(train_df)} train / {len(dev_df)} dev / {len(test_df)} test")
    print(f"TSVs saved to: {OUTPUT_DIR}")
    print(f"\nNext step: update TRAIN_PATH / DEV_PATH / TEST_PATH in train_text2gloss.py")
    print(f"to point to the files in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()