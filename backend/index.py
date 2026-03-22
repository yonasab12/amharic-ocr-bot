# api/index.py
from flask import Flask, request, jsonify, render_template
import os
import tempfile
from OCR import pipeline

app = Flask(__name__, template_folder='../templates', static_folder='../static')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'Empty file'}), 400

    # Save uploaded file to a temporary location
    suffix = os.path.splitext(file.filename)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        file.save(tmp.name)
        temp_path = tmp.name

    try:
        # Run OCR pipeline (bw=True to use grayscale, adjust as needed)
        text = pipeline(temp_path, bw=True)
    except Exception as e:
        text = f"Error during OCR: {str(e)}"
    finally:
        os.unlink(temp_path)  # Clean up uploaded file

    return jsonify({'text': text})

# Vercel requires the app to be exposed as 'app'
# For local development, we can also run with `flask run`
if __name__ == '__main__':
    app.run(debug=True)
