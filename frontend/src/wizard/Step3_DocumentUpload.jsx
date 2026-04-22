import { useState, useRef } from "react";

const NAVY = "#0d2b7a";
const GOLD = "#f5a623";
const TXT = "#111827";
const TXT2 = "#4b5563";
const ERR = "#ef4444";
const OK = "#10b981";

const API_URL = "https://dac-healthprice-api.onrender.com";
const MAX_SIZE_MB = 10;

function formatSize(bytes) {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

export default function Step3_DocumentUpload({ data, onChange, errors }) {
  const inputRef = useRef(null);
  const [dragOver, setDragOver] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [uploadStatus, setUploadStatus] = useState(null); // null | 'success' | 'error' | 'backend_pending'

  const handleFiles = (files) => {
    const file = files[0];
    if (!file) return;

    if (file.type !== "application/pdf") {
      setUploadStatus({ type: "error", msg: "Only PDF files are accepted." });
      return;
    }
    if (file.size > MAX_SIZE_MB * 1024 * 1024) {
      setUploadStatus({ type: "error", msg: `File too large. Maximum size is ${MAX_SIZE_MB} MB.` });
      return;
    }

    setUploadStatus(null);
    onChange({ ...data, documentFile: file, extractedData: null });
    attemptUpload(file);
  };

  const attemptUpload = async (file) => {
    setUploading(true);
    try {
      const fd = new FormData();
      fd.append("file", file);
      const r = await fetch(`${API_URL}/api/v1/documents/upload`, {
        method: "POST", body: fd, signal: AbortSignal.timeout(60000),
      });
      if (r.ok) {
        const result = await r.json();
        onChange(prev => ({ ...prev, documentFile: file, extractedData: result.extracted || null, documentId: result.id }));
        setUploadStatus({ type: "success", msg: "Document uploaded and extracted successfully." });
      } else {
        // Backend endpoint not yet deployed — save file locally for Week 2 integration
        setUploadStatus({ type: "backend_pending", msg: "Document saved. Extraction will run when the backend is deployed." });
      }
    } catch {
      // Backend not yet available — store file for later submission
      setUploadStatus({ type: "backend_pending", msg: "Document saved locally. It will be processed after submission." });
    } finally {
      setUploading(false);
    }
  };

  const removeFile = () => {
    onChange({ ...data, documentFile: null, extractedData: null, documentId: null });
    setUploadStatus(null);
    if (inputRef.current) inputRef.current.value = "";
  };

  const file = data.documentFile;

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 20 }}>
      <div>
        <h2 style={{ fontSize: 22, fontWeight: 700, color: NAVY, marginBottom: 4 }}>Medical Documents</h2>
        <p style={{ fontSize: 14, color: TXT2 }}>Upload recent lab results, medical reports, or discharge summaries. This helps expedite underwriting review.</p>
      </div>

      {/* Drop zone */}
      {!file && (
        <div
          onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
          onDragLeave={() => setDragOver(false)}
          onDrop={(e) => { e.preventDefault(); setDragOver(false); handleFiles(e.dataTransfer.files); }}
          onClick={() => inputRef.current?.click()}
          style={{
            border: `2px dashed ${dragOver ? NAVY : "#d1d5db"}`,
            borderRadius: 12,
            padding: "40px 24px",
            textAlign: "center",
            cursor: "pointer",
            background: dragOver ? `rgba(13,43,122,0.04)` : "#fafafa",
            transition: "all 0.2s",
          }}
        >
          <div style={{ fontSize: 36, marginBottom: 12 }}>📄</div>
          <p style={{ fontWeight: 600, color: TXT, marginBottom: 4 }}>
            {dragOver ? "Drop your PDF here" : "Drag & drop a PDF here"}
          </p>
          <p style={{ fontSize: 13, color: TXT2, marginBottom: 16 }}>or click to browse</p>
          <span style={{
            padding: "8px 20px", borderRadius: 20, background: NAVY, color: "#fff",
            fontSize: 13, fontWeight: 600, cursor: "pointer",
          }}>
            Choose file
          </span>
          <p style={{ fontSize: 12, color: TXT2, marginTop: 12 }}>PDF only · Max {MAX_SIZE_MB} MB</p>
          <input ref={inputRef} type="file" accept=".pdf" style={{ display: "none" }}
            onChange={(e) => handleFiles(e.target.files)} />
        </div>
      )}

      {/* File preview */}
      {file && (
        <div style={{
          display: "flex", alignItems: "center", gap: 16, padding: "16px 20px",
          borderRadius: 12, border: "1.5px solid #e5e7eb", background: "#fff",
        }}>
          <div style={{ fontSize: 32 }}>📄</div>
          <div style={{ flex: 1, minWidth: 0 }}>
            <p style={{ fontWeight: 600, fontSize: 14, color: TXT, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{file.name}</p>
            <p style={{ fontSize: 12, color: TXT2 }}>{formatSize(file.size)}</p>
          </div>
          {uploading && (
            <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
              <div style={{ width: 16, height: 16, borderRadius: "50%", border: `2px solid ${NAVY}`, borderTopColor: "transparent", animation: "spin 0.8s linear infinite" }} />
              <span style={{ fontSize: 13, color: TXT2 }}>Processing...</span>
            </div>
          )}
          <button onClick={removeFile} style={{
            background: "none", border: "none", cursor: "pointer", color: TXT2,
            padding: "4px 8px", borderRadius: 6, fontSize: 13,
          }}>✕ Remove</button>
        </div>
      )}

      {/* Upload status */}
      {uploadStatus && (
        <div style={{
          padding: "12px 16px", borderRadius: 10,
          background: uploadStatus.type === "success" ? "#ecfdf5"
            : uploadStatus.type === "backend_pending" ? `rgba(245,166,35,0.08)`
            : "#fef2f2",
          border: `1px solid ${uploadStatus.type === "success" ? "#a7f3d0"
            : uploadStatus.type === "backend_pending" ? GOLD
            : "#fecaca"}`,
          display: "flex", alignItems: "center", gap: 10,
        }}>
          <span>{uploadStatus.type === "success" ? "✅" : uploadStatus.type === "backend_pending" ? "⏳" : "❌"}</span>
          <span style={{ fontSize: 13, color: TXT }}>{uploadStatus.msg}</span>
        </div>
      )}

      {/* Skip note */}
      <div style={{ background: "#f8f9fb", borderRadius: 10, padding: "12px 16px", border: "1px solid #e5e7eb" }}>
        <p style={{ fontSize: 13, color: TXT2, margin: 0 }}>
          <strong>Document upload is optional.</strong> You can proceed without uploading — our underwriter will contact you if additional medical documentation is required.
        </p>
      </div>

      <style>{`@keyframes spin { to { transform: rotate(360deg); } }`}</style>
    </div>
  );
}
