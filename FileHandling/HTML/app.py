from flask import Flask, request, jsonify, send_from_directory
import os

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
DOWNLOAD_FOLDER = 'downloads'
ALLOWED_EXTENSIONS = {'txt', 'csv', 'json', 'jpg', 'png'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['DOWNLOAD_FOLDER'] = DOWNLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
if not os.path.exists(DOWNLOAD_FOLDER):
    os.makedirs(DOWNLOAD_FOLDER)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    additional_data = request.form.get('data')

    if file and allowed_file(file.filename):
        filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filename)

        # Store additional data
        with open(os.path.join(UPLOAD_FOLDER, f"{file.filename}.txt"), 'w') as f:
            f.write(f"Additional Data: {additional_data}")

        return jsonify({'message': 'File successfully uploaded', 'data': additional_data}), 200
    else:
        return jsonify({'error': 'Invalid file type'}), 400

@app.route('/download/<filename>', methods=['GET'])
def download_file(filename):
    try:
        return send_from_directory(app.config['UPLOAD_FOLDER'], filename)
    except FileNotFoundError:
        return jsonify({'error': 'File not found'}), 404

@app.route('/download-info/<filename>', methods=['GET'])
def download_info(filename):
    try:
        with open(os.path.join(UPLOAD_FOLDER, f"{filename}.txt"), 'r') as file:
            info = file.read()
        return jsonify({'info': info}), 200
    except FileNotFoundError:
        return jsonify({'error': 'Info file not found'}), 404

@app.route('/files', methods=['GET'])
def list_files():
    files = os.listdir(UPLOAD_FOLDER)
    files = [file for file in files if allowed_file(file)]
    return jsonify({'files': files}), 200

if __name__ == '__main__':
    app.run(debug=True)
