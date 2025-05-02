from flask import Flask, request, jsonify
from tokenizer import tokenize

app = Flask(__name__)

@app.route('/tokenize', methods=['POST'])
def tokenize_text():
    data = request.get_json()
    text = data.get('text', '')
    tokens = tokenize(text)
    return jsonify({'tokens': tokens})

if __name__ == '__main__':
    app.run(port=5000)
