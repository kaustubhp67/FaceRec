from flask import Flask, request, jsonify
import face_recognition
from PIL import Image
import io
import os

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'  # Set your uploads folder

@app.route('/')
def index():
    return "Face Recognition App is Running!"

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"})
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"})

    # Save file to uploads folder
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(file_path)

    # Open the image and process
    image = face_recognition.load_image_file(file_path)
    face_locations = face_recognition.face_locations(image)

    return jsonify({"message": f"Found {len(face_locations)} face(s) in the image."})

if __name__ == '__main__':
    app.run(debug=True)
