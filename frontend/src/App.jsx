const API_URL =
  import.meta.env.VITE_API_URL || "http://localhost:8000";
import { useState } from "react";

function App() {
  const [file, setFile] = useState(null);
  const [result, setResult] = useState(null);
  const [history, setHistory] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const handleFile = (event) => {
    const selectedFile = event.target.files[0];

    if (selectedFile) {
      setFile(selectedFile);
      setResult(null);
      setError("");
    }
  };

  const analyzeImage = async () => {
    if (!file) {
      setError("Please select an image first.");
      return;
    }

    setLoading(true);
    setError("");
    setResult(null);

    const formData = new FormData();
    formData.append("file", file);

    try {
      const response = await fetch(
        `${API_URL}/analyze`,
        {
          method: "POST",
          body: formData,
        }
      );

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.detail || "Analysis failed");
      }

      setResult(data);

      loadHistory();

    } catch (err) {
      setError(err.message);
    }

    setLoading(false);
  };

  const loadHistory = async () => {
    try {
      const response = await fetch(
        `${API_URL}/history`
      );

      if (!response.ok) {
        return;
      }

      const data = await response.json();
      setHistory(data);

    } catch {
      console.log("Could not load history");
    }
  };

  return (
    <div
      style={{
        maxWidth: "850px",
        margin: "40px auto",
        padding: "20px",
        fontFamily: "Arial",
      }}
    >

      <h1>Image Quality Assessment</h1>

      <p>
        Upload an image to check its visual quality.
      </p>

      <hr />

      {/* Upload */}

      <div style={{ marginTop: "25px" }}>

        <input
          type="file"
          accept="image/*"
          onChange={handleFile}
        />

        {file && (
          <div>

            <p>
              Selected: <strong>{file.name}</strong>
            </p>

            <img
              src={URL.createObjectURL(file)}
              alt="Preview"
              style={{
                maxWidth: "500px",
                maxHeight: "350px",
                display: "block",
                margin: "20px 0",
              }}
            />

            <button
              onClick={analyzeImage}
              disabled={loading}
            >
              {loading
                ? "Analyzing..."
                : "Analyze Image"}
            </button>

          </div>
        )}

      </div>

      {/* Error */}

      {error && (
        <p style={{ color: "red" }}>
          {error}
        </p>
      )}

      {/* Result */}

      {result && (
        <div style={{ marginTop: "35px" }}>

          <h2>Analysis Result</h2>

          <h3>
            Quality Score: {result.quality_score}
          </h3>

          <h3>
            Quality: {result.quality_label}
          </h3>

          <h3>Detected Issues</h3>

          {result.issues && result.issues.length > 0 ? (
            result.issues.map((issue, index) => (
              <div
                key={index}
                style={{
                  padding: "10px",
                  marginBottom: "8px",
                  background: "#f2f2f2",
                }}
              >
                <strong>{issue.type}</strong>

                <p>
                  Severity: {issue.severity}
                </p>

                <p>
                  Confidence:{" "}
                  {(issue.confidence * 100).toFixed(0)}%
                </p>

              </div>
            ))
          ) : (
            <p>No major issues detected.</p>
          )}

        </div>
      )}

      {/* History */}

      <div style={{ marginTop: "45px" }}>

        <h2>Previous Analyses</h2>

        <button onClick={loadHistory}>
          Refresh History
        </button>

        {history.length === 0 ? (
          <p>No previous analyses found.</p>
        ) : (
          <div style={{ marginTop: "20px" }}>

            {history.map((item, index) => (
              <div
                key={item.id || index}
                style={{
                  padding: "15px",
                  marginBottom: "10px",
                  border: "1px solid #ddd",
                  borderRadius: "6px",
                }}
              >

                <strong>
                  {item.filename || "Image"}
                </strong>

                <p>
                  Score: {item.quality_score}
                </p>

                <p>
                  Result: {item.quality_label}
                </p>

              </div>
            ))}

          </div>
        )}

      </div>

    </div>
  );
}

export default App;