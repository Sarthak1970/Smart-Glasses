import sqlite3
import base64
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import pytesseract
import io

app = Flask(__name__)
CORS(app)

def extract_text_from_image(image_data):
    """Extract text from binary image data using pytesseract."""
    image = Image.open(io.BytesIO(image_data))
    text = pytesseract.image_to_string(image)
    return text

def retrieve_images(partial_caption=None, partial_timestamp=None, ocr_text=None):
    """Retrieve images from the database with optional filters."""
    conn = sqlite3.connect('captured_images.db')
    cursor = conn.cursor()

    query = "SELECT timestamp, image_data, caption, ocr_text FROM Images WHERE 1=1"
    params = []

    if partial_caption:
        query += " AND caption LIKE ?"
        params.append(f"%{partial_caption}%")

    if partial_timestamp:
        query += " AND timestamp LIKE ?"
        params.append(f"%{partial_timestamp}%")

    cursor.execute(query, params)
    rows = cursor.fetchall()
    conn.close()

    images = []
    for timestamp, img_data, caption, db_ocr_text in rows:
        img_base64 = base64.b64encode(img_data).decode('utf-8')
        image_info = {
            "timestamp": timestamp,
            "image": f"data:image/jpeg;base64,{img_base64}",
            "caption": caption,
            "ocr_text": db_ocr_text
        }

        # If OCR text is provided, check if it matches extracted or stored OCR text
        if ocr_text:
            if ocr_text.lower() in db_ocr_text.lower():
                images.append(image_info)
        else:
            images.append(image_info)

    return images

@app.route('/api/images', methods=['GET'])
def get_images():
    """API endpoint to get images based on caption, timestamp, or OCR text."""
    partial_caption = request.args.get('caption', '').strip()
    partial_timestamp = request.args.get('timestamp', '').strip()
    ocr_text = request.args.get('ocr_text', '').strip()  # Corrected here!

    if not partial_caption and not partial_timestamp and not ocr_text:
        return jsonify({"error": "At least one of caption, timestamp, or OCR text is required"}), 400

    images = retrieve_images(
        partial_caption if partial_caption else None,
        partial_timestamp if partial_timestamp else None,
        ocr_text if ocr_text else None
    )

    if not images:
        return jsonify({"message": "No images found"}), 404

    return jsonify(images)

@app.route('/api/images/latest', methods=['GET'])
def get_latest_image():
    print("GET /api/images/latest endpoint was hit")
    """API endpoint to get the most recently added image."""
    conn = sqlite3.connect('captured_images.db')
    cursor = conn.cursor()

    query = "SELECT timestamp, image_data, caption, ocr_text FROM Images ORDER BY timestamp DESC LIMIT 1"
    cursor.execute(query)
    row = cursor.fetchone()
    conn.close()

    if row:
        timestamp, img_data, caption, ocr_text = row
        img_base64 = base64.b64encode(img_data).decode('utf-8')
        image_info = {
            "timestamp": timestamp,
            # "image": f"data:image/jpeg;base64,{img_base64}",
            "caption": caption,
            "ocr_text": ocr_text
        }
        return jsonify(image_info)

    return jsonify({"message": "No image found"}), 404

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)