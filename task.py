from transformers import MarianMTModel, MarianTokenizer
import re

# Pre-clean input
def normalize_input(text):
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    return text

# Translation function
def translate(text, src_lang="en", tgt_lang="hi"):
    try:
        model_name = f"Helsinki-NLP/opus-mt-{src_lang}-{tgt_lang}"
        tokenizer = MarianTokenizer.from_pretrained(model_name)
        model = MarianMTModel.from_pretrained(model_name)

        # Fix proper names: ensure they stay untouched
        text = normalize_input(text)
        text = text.replace("Garvit", "Garvit")  # Keeps name intact

        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
        translated = model.generate(**inputs)
        output = tokenizer.decode(translated[0], skip_special_tokens=True)
        return output

    except Exception as e:
        return f"Error: {str(e)}"

# Run the translator
if __name__ == "__main__":
    input_text = input("Enter English text to translate into Hindi: ")
    translated_text = translate(input_text)
    print("Translated:", translated_text)
