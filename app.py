from flask import Flask, request, jsonify, render_template
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('chat.html')  # Ensure chat.html is in the templates folder

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    user_text = data.get('text', '')

    prompt = f"User: {user_text}\nBot:"
    inputs = tokenizer.encode(prompt, return_tensors="pt")
    outputs = model.generate(
        inputs,
        max_length=50,
        num_return_sequences=1,
        pad_token_id=tokenizer.eos_token_id,
        no_repeat_ngram_size=2,
        top_p=0.92,
        temperature=0.7
    )
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = clean_response(generated_text, prompt)

    return jsonify({'response': response})

# Helper function to clean GPT-2 response
def clean_response(generated_text, prompt):
    response = generated_text.replace(prompt, "").strip()
    return response.split("\n")[0]

if __name__ == '__main__':
    app.run(debug=True)
