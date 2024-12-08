import os
from flask import Flask, request, jsonify
import requests

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    filename = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filename)
    return jsonify({"message": "File uploaded successfully", "filename": file.filename}), 200

def upload_file_to_backend(file_path):
    url = 'http://127.0.0.1:5000/upload'
    with open(file_path, 'rb') as f:
        files = {'file': (file_path, f)}
        response = requests.post(url, files=files)
        print(response.json())

if __name__ == '__main__':
    from threading import Thread
    def run_app():
        app.run(debug=True, use_reloader=False)

    thread = Thread(target=run_app)
    thread.start()

    upload_file_to_backend('path_to_your_file.ext')
