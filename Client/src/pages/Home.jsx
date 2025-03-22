import React, { useState } from "react";
import axios from "axios";

const Home = () => {
  const [caption, setCaption] = useState("");
  const [timestamp, setTimestamp] = useState("");
  const [ocrText, setOcrText] = useState(""); // New OCR text state
  const [images, setImages] = useState([]);
  const [error, setError] = useState("");

  const fetchImages = async () => {
    try {
      const response = await axios.get("http://localhost:5000/api/images", {
        params: { caption, timestamp, ocr_text: ocrText }, // Include ocr_text in the query
      });

      setImages(response.data);
      setError("");
    } catch (err) {
      setImages([]);
      setError("No images found or error fetching images.");
    }
  };

  return (
    <div className="bg-gray-100 min-h-screen flex flex-col items-center py-4">
      <h1 className="text-2xl font-bold mb-4">Retrieve Images</h1>

      {/* Caption Input */}
      <input
        type="text"
        placeholder="Enter caption"
        value={caption}
        onChange={(e) => setCaption(e.target.value)}
        className="border p-2 mb-2 w-64 text-center"
      />

      {/* Timestamp Input */}
      <input
        type="text"
        placeholder="Enter timestamp (YYYY-MM-DD HH:MM)"
        value={timestamp}
        onChange={(e) => setTimestamp(e.target.value)}
        className="border p-2 mb-2 w-64 text-center"
      />

      {/* OCR Text Input */}
      <input
        type="text"
        placeholder="Enter OCR text"
        value={ocrText}
        onChange={(e) => setOcrText(e.target.value)}
        className="border p-2 mb-4 w-64 text-center"
      />

      <button
        onClick={fetchImages}
        className="bg-blue-500 text-white px-4 py-2 rounded"
      >
        Fetch Images
      </button>

      {error && <p className="text-red-500 mt-4">{error}</p>}

      {/* Display Fetched Images */}
      <div className="mt-6 grid grid-cols-3 gap-4">
        {images.map((img, index) => (
          <div
            key={index}
            className="p-2 border rounded shadow-md flex flex-col items-center"
          >
            <img
              src={img.image}
              alt={`Timestamp: ${img.timestamp}`}
              className="w-full max-w-[500px] h-auto object-cover rounded-lg shadow-md"
            />
            <p className="text-sm mt-2 text-center font-semibold">
              📅 {img.timestamp}
            </p>
            <p className="text-xs mt-1 text-center">
              📝 <strong>Caption:</strong> {img.caption || "No caption available"}
            </p>
            <p className="text-xs mt-1 text-center">
              🔎 <strong>OCR:</strong>{" "}
              {img.ocr_text || "No OCR text detected"}
            </p>
          </div>
        ))}
      </div>
    </div>
  );
};

export default Home;
