from flask import Blueprint, request, jsonify
import os
from werkzeug.utils import secure_filename
from services.retrieval.indexing_service import create_and_save_index
from config import DATA_FOLDER

# Create a Blueprint for handling uploads
upload_bp = Blueprint('upload', __name__)

@upload_bp.route("/upload", methods=["POST"])
def upload():
    """
    Handle file uploads and reindex the document in the system.
    """

    # Check if a file is included in the request
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']

    # If no file is selected
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    # Save the file to the specified data folder
    filename = secure_filename(file.filename)
    save_path = os.path.join(DATA_FOLDER, filename)
    try:
        file.save(save_path)
    except Exception as e:
        return jsonify({"error": f"Failed to save file: {str(e)}"}), 500

    # Reprocess the uploaded data: chunk, embed, and update index
    try:
        # Create and save the updated index using the saved file
        create_and_save_index(
            data_folder=DATA_FOLDER,
            chunk_strategy='token',  # You can change this to use any desired chunking strategy
            chunk_size=512,
            overlap=50
        )
        return jsonify({"success": f"File '{filename}' uploaded and indexed successfully."}), 200
    except Exception as e:
        return jsonify({"error": f"Failed to process file: {str(e)}"}), 500
