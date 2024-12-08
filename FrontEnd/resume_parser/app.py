from flask import Flask, render_template, request, redirect, url_for, send_file
from werkzeug.utils import secure_filename
import os
from main import *

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'uploads'
RESULTS_FOLDER = 'results'
ALLOWED_EXTENSIONS = {'pdf', 'jpg', 'jpeg', 'png'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULTS_FOLDER'] = RESULTS_FOLDER

# Ensure necessary directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

def allowed_file(filename):
    """Check if a file is allowed based on its extension."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    """Render the main upload page."""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file uploads and resume processing."""
    # Validate form inputs
    if 'job_description' not in request.form or 'threshold' not in request.form:
        return redirect(url_for('index'))
    
    job_description = request.form['job_description']
    try:
        threshold = int(request.form['threshold'])
    except ValueError:
        return "Invalid threshold value! Must be an integer."

    # Handle file uploads
    files = request.files.getlist('file')
    resume_files = []

    for file in files:
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            resume_files.append(file_path)

    if not resume_files:
        return "No valid resume files uploaded!"

    # Process resumes
    try:
        all_results_file, selected_results_file, not_selected_results_file = process_resumes(
            resume_files, job_description, threshold
        )
    except Exception as e:
        return f"An error occurred during processing: {e}"

    # Redirect to the results page
    return render_template(
        'result.html',
        all_results_file=os.path.basename(all_results_file),
        selected_results_file=os.path.basename(selected_results_file),
        not_selected_results_file=os.path.basename(not_selected_results_file),
    )
from flask import Flask, render_template, request, redirect, url_for, send_file
import io

@app.route('/download/<file_type>')
def download_file(file_type):
    """Serve the in-memory files for download."""
    # The file objects are returned as part of the result of processing resumes
    all_results_file, selected_results_file, not_selected_results_file = process_resumes(
        
    )

    # Map file types to the correct in-memory file object
    file_map = {
        'all_results': all_results_file,
        'selected_results': selected_results_file,
        'not_selected_results': not_selected_results_file
    }

    # Get the requested file
    file = file_map.get(file_type)
    if not file:
        return "File not found!", 404

    # Serve the file as a downloadable response
    return send_file(file, mimetype='text/csv', as_attachment=True, download_name=f"{file_type}.csv")


if __name__ == '__main__':
    app.run(debug=True)
