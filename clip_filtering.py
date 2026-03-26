import re
import pandas as pd

CSV_PATH = "Auslan-Daily_Communication.csv"
OUTPUT_MANIFEST = "filtered_manifest.csv"

CONV_PHRASES = [
    "hello", "hi", "bye", "goodbye", "see you",
    "thanks", "thank you", "welcome",
    "yes", "no", "okay", "ok", "great", "good", "please", "sorry",
    "i am", "my name", "this is",
    "what", "where", "who", "how", "why", "when",
    "can you", "do you", "are you", "is it",
    "look", "come", "go", "watch", "let us", "lets", "stop", "wait", "show", "tell"
]

def clean_text(text):
    text = str(text).lower().strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^a-z0-9\s']", "", text)
    return text.strip()

def has_conv_phrase(text):
    return any(p in text for p in CONV_PHRASES)

df = pd.read_csv(CSV_PATH, sep=";")

# Remove accidental index columns if present
drop_cols = [c for c in df.columns if str(c).lower().startswith("unnamed")]
if drop_cols:
    df = df.drop(columns=drop_cols)

df["subtitle_clean"] = df["Subtitle"].fillna("").map(clean_text)
df["word_count"] = df["subtitle_clean"].str.split().str.len()
df["has_conv_kw"] = df["subtitle_clean"].map(has_conv_phrase)

# Recommended first-pass conversational subset
filtered = df[
    (df["word_count"] <= 8) &
    (df["has_conv_kw"]) &
    (df["subtitle_clean"] != "")
].copy()

filtered.to_csv(OUTPUT_MANIFEST, index=False)

print("All clips:", len(df))
print("Filtered conversational clips:", len(filtered))
print("Train:", (filtered["Split"] == "train").sum())
print("Dev:", (filtered["Split"] == "dev").sum())
print("Test:", (filtered["Split"] == "test").sum())
print("Unique signers:", filtered["Signer_ID"].nunique())
print(f"Saved to {OUTPUT_MANIFEST}")