import sqlite3
import base64
from flask import Flask, request, jsonify
from flask_cors import CORS  

app = Flask(__name__)
CORS(app)

def retrieve_images(partial_caption=None, partial_timestamp=None):
    conn = sqlite3.connect('captured_images.db')
    cursor = conn.cursor()
    
    query = "SELECT timestamp, image_data FROM Images WHERE 1=1"
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
    for timestamp, img_data in rows:
        img_base64 = base64.b64encode(img_data).decode('utf-8')
        images.append({"timestamp": timestamp, "image": f"data:image/jpeg;base64,{img_base64}"})

    return images

@app.route('/api/images', methods=['GET'])
def get_images():
    partial_caption = request.args.get('caption', '').strip()
    partial_timestamp = request.args.get('timestamp', '').strip()

    if not partial_caption and not partial_timestamp:
        return jsonify({"error": "At least one of caption or timestamp is required"}), 400

    images = retrieve_images(partial_caption if partial_caption else None, partial_timestamp if partial_timestamp else None)

    if not images:
        return jsonify({"message": "No images found"}), 404

    return jsonify(images)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)