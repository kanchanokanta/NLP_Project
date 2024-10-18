from flask import Flask, render_template, request, jsonify
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration

app = Flask(__name__)

# Define the model path
model_path = r"C:\HereIsMyWork\KMITL\Year 3 term 1\NLP\Project\model.pt"

# Load the model
model = T5ForConditionalGeneration.from_pretrained('t5-small')
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
tokenizer = T5Tokenizer.from_pretrained('t5-small')

# Move model to CPU
model.to(torch.device('cpu'))

@app.route('/')
def upload_file():
    return render_template('upload.html')

@app.route('/uploader', methods=['POST'])

def uploader():
    data = request.get_json()
    text = data.get('text', '')
    if text:
        summary = summarize_text_by_chunks(text, model, tokenizer)
        return jsonify({'summary': summary})
    return jsonify({'error': 'No text provided'}), 400

# Function to split article into chunks
def split_text_into_chunks(text, chunk_size=512):
    words = text.split()
    return [' '.join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]


# Summarize text by chunks
def summarize_text_by_chunks(text, model, tokenizer, max_length=512):
    chunked_summaries = []
    device = torch.device('cpu')

    with torch.no_grad():
        article_chunks = split_text_into_chunks(text, chunk_size=max_length)
        full_summary = []

        for chunk in article_chunks:
            inputs = tokenizer(chunk, max_length=max_length, padding='max_length', truncation=True, return_tensors="pt")
            inputs = {key: value.to(device) for key, value in inputs.items()}

            summary_ids = model.generate(
                inputs['input_ids'],
                max_length=10000,
                min_length=10,
                num_beams=5,
                early_stopping=True,
                no_repeat_ngram_size=3,
                temperature=0.7,
                top_k=50,
                top_p=0.95,
                do_sample=True
            )
            summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
            full_summary.append(summary)

        combined_summary = " ".join(full_summary)
        chunked_summaries.append(combined_summary)

    return combined_summary

if __name__ == '__main__':
    app.run(debug=True)
