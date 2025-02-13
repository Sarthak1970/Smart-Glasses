import sqlite3
import io
import base64
from flask import Flask, request, jsonify
from flask_cors import CORS  # Import CORS

app = Flask(__name__)
CORS(app)  # 

def retrieve_images_by_partial_timestamp(partial_timestamp):
    conn = sqlite3.connect('captured_images.db')
    cursor = conn.cursor()
    cursor.execute("SELECT timestamp, image_data FROM Images WHERE timestamp LIKE ?", (f"%{partial_timestamp}%",))
    rows = cursor.fetchall()
    conn.close()

    images = []
    for timestamp, img_data in rows:
        img_base64 = base64.b64encode(img_data).decode('utf-8')
        images.append({"timestamp": timestamp, "image": f"data:image/jpeg;base64,{img_base64}"})

    return images

@app.route('/api/images', methods=['GET'])
def get_images():
    partial_timestamp = request.args.get('timestamp')
    if not partial_timestamp:
        return jsonify({"error": "Timestamp is required"}), 400

    images = retrieve_images_by_partial_timestamp(partial_timestamp)
    if not images:
        return jsonify({"message": "No images found"}), 404

    return jsonify(images)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True) 

