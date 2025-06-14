import React, { useEffect, useState } from "react";

const Popup = () => {
  const [logs, setLogs] = useState([]);

  // Load logs from chrome.storage
  const loadLogs = () => {
    chrome.storage.local.get({ alerts: [] }, ({ alerts }) => {
      setLogs(alerts.slice(-5).reverse());
    });
  };

  useEffect(() => {
    loadLogs();
  }, []);

  // Clear logs both in storage and in UI
  const clearLogs = () => {
    // Clear local storage
    chrome.storage.local.set({ alerts: [] }, () => {
      setLogs([]);
    });
    fetch("http://127.0.0.1:8000/clear-alerts", {
      method: "POST",
    }).catch((err) => console.error("Failed to clear backend logs:", err));
  };

  return (
    <div style={{ padding: 16, fontFamily: "Arial, sans-serif", width: 320 }}>
      <div
        style={{
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
          marginBottom: 8,
        }}>
        <h2 style={{ margin: 0 }}>📜 ProtectU Logs</h2>
        <button
          onClick={clearLogs}
          style={{
            background: "#c62828",
            border: "none",
            color: "#FFFFFF",
            fontSize: 14,
            padding: "4px 8px",
            borderRadius: 4,
            cursor: "pointer",
          }}>
          Clear Logs
        </button>
      </div>
      {!logs.length && <p style={{ color: "#888", margin: 0 }}>No logs yet</p>}
      <ul
        style={{
          listStyle: "none",
          padding: 0,
          marginTop: logs.length ? 0 : 8,
        }}>
        {logs.map((e, i) => {
          if (e.type === "email") {
            const isPhish = e.verdict === "Phishing";
            return (
              <li
                key={i}
                style={{
                  marginBottom: 8,
                  padding: 8,
                  borderLeft: `4px solid ${isPhish ? "#c62828" : "#388e3c"}`,
                  background: "#fff3e0",
                  borderRadius: 4,
                  fontSize: 13,
                }}>
                <div>
                  <strong>Email:</strong>{" "}
                  <span title={e.emailSnippet}>{e.emailSnippet}...</span>
                </div>
                <div>
                  <strong>Verdict:</strong> {e.verdict}
                </div>
                <div>
                  <strong>Confidence:</strong>{" "}
                  {(e.predictionScore * 100).toFixed(1)}%
                </div>
                <div>
                  <strong>Time:</strong>{" "}
                  {new Date(e.timestamp).toLocaleString()}
                </div>
              </li>
            );
          }
          // Default: URL alert
          const isPhish = e.verdict === "Phishing";
          return (
            <li
              key={i}
              style={{
                marginBottom: 8,
                padding: 8,
                borderLeft: `4px solid ${isPhish ? "#c62828" : "#388e3c"}`,
                background: "#f5f5f5",
                borderRadius: 4,
                fontSize: 13,
              }}>
              <div>
                <strong>URL:</strong> {e.url}
              </div>
              <div>
                <strong>Verdict:</strong> {e.verdict}
              </div>
              <div>
                <strong>Time:</strong> {new Date(e.timestamp).toLocaleString()}
              </div>
            </li>
          );
        })}
      </ul>
    </div>
  );
};

export default Popup;
