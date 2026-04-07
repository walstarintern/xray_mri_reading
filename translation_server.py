from flask import Flask, request, jsonify
from googletrans import Translator

app = Flask(__name__)
translator = Translator()

@app.route('/translate', methods=['POST'])
def translate_text():
    try:
        data = request.json
        text = data.get('text')
        target_lang = data.get('dest', 'en')
        
        if not text:
            return jsonify({"error": "No text provided"}), 400

        # Perform the translation
        result = translator.translate(text, dest=target_lang).text
        return jsonify({"translated_text": result})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    print("🚀 Translator Microservice is running on http://127.0.0.1:5000")
    app.run(host='127.0.0.1', port=5000)