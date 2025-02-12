import React, { useState } from "react";
import axios from "axios";

const Home = () => {
  const [timestamp, setTimestamp] = useState("");
  const [images, setImages] = useState([]);
  const [error, setError] = useState("");

  const fetchImages = async () => {
    try {
      const response = await axios.get(`http://localhost:5000/api/images`, {
        params: { timestamp },
      });

      setImages(response.data);
      setError("");
    } catch (err) {
      setImages([]);
      setError("No images found or error fetching images.");
    }
  };

  return (
    <div className="bg-gray-100 h-screen flex flex-col items-center justify-center">
      <h1 className="text-2xl font-bold mb-4">Retrieve Images</h1>
      <input
        type="text"
        placeholder="Enter timestamp (YYYY-MM-DD HH:MM)"
        value={timestamp}
        onChange={(e) => setTimestamp(e.target.value)}
        className="border p-2 mb-4 w-64"
      />
      <button onClick={fetchImages} className="bg-blue-500 text-white px-4 py-2">
        Fetch Images
      </button>

      {error && <p className="text-red-500 mt-4">{error}</p>}

      <div className="mt-4 flex flex-wrap gap-4">
        {images.map((img, index) => (
          <div key={index} className="p-2 border rounded">
            <p className="text-sm">{img.timestamp}</p>
            <img src={img.image} alt={`Timestamp: ${img.timestamp}`} className="w-48 h-48 object-cover mt-2" />
          </div>
        ))}
      </div>
    </div>
  );
};

export default Home;
