from flask import Flask, render_template, request
from transformers import MarianMTModel, MarianTokenizer
import re

app = Flask(__name__)

def normalize_input(text):
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    return text

def translate(text, src_lang="en", tgt_lang="hi"):
    model_name = f"Helsinki-NLP/opus-mt-{src_lang}-{tgt_lang}"
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)

    text = normalize_input(text)
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
    translated = model.generate(**inputs)
    return tokenizer.decode(translated[0], skip_special_tokens=True)

@app.route("/", methods=["GET", "POST"])
def translate_view():
    translated_text = ""
    if request.method == "POST":
        text = request.form["text"]
        src_lang = request.form["src_lang"]
        tgt_lang = request.form["tgt_lang"]
        translated_text = translate(text, src_lang, tgt_lang)

    return render_template("index.html", translated_text=translated_text)

if __name__ == "__main__":
    app.run(debug=True)
