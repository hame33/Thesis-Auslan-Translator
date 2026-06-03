from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

MODEL_DIR = "./gloss2text_t5smallGPT5/checkpoint-1130"
PREFIX = "translate Auslan gloss to English: "

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_DIR)

def translate(gloss: str) -> str:
    inputs = tokenizer(PREFIX + gloss, return_tensors="pt")
    out = model.generate(
        **inputs,
        max_length=64,
        num_beams=4,
        early_stopping=True,
    )
    return tokenizer.decode(out[0], skip_special_tokens=True)

examples = [
    "HELLO PT_PRO2 GOOD",
    "TIME NOW WHAT",
    "PT_PRO1 KNOW READY",
    "YESTERDAY PT_PRO1 GO SHOP",
    "PT_PRO2 THINK WHAT",
    "TOMORROW PT_PRO1 GO DOCTOR",
    "NOW PT_PRO3 NOT WORK",
]

for g in examples:
    print(g, "->", translate(g))
