// background.js

const API_BASE = 'http://localhost:5030';

// Track last-checked URL per tab to avoid duplicates
const lastLoggedUrl = {};

/**
 * Send a URL to the backend and return { decision, score }.
 */
async function checkUrlWithBackend(url) {
  try {
    const res = await fetch(`${API_BASE}/check_url`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ url })
    });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    return await res.json();
  } catch (err) {
    console.error('[ProtectU] Backend check failed:', err);
    return { decision: 'LEGITIMATE', score: 0 };
  }
}

/**
 * Instruct content script to show banner and set badge.
 */
function showBanner(tabId, url, decision, score) {
  chrome.tabs.sendMessage(tabId, { type: 'RISKY_SITE', url, decision, score });
  chrome.action.setBadgeText({ text: '⚠️', tabId });
  chrome.action.setBadgeBackgroundColor({ color: '#B00020', tabId });
}

/**
 * Clear the badge on safe pages.
 */
function clearBanner(tabId) {
  chrome.action.setBadgeText({ text: '', tabId });
}

/**
 * Central handler: called on URL change. Deduplicates per tab.
 */
async function handleUrl(tabId, url) {
  if (!url.startsWith('http')) return;

  // Skip duplicates
  if (lastLoggedUrl[tabId] === url) return;
  lastLoggedUrl[tabId] = url;

  console.log(`[ProtectU] Checking URL: ${url}`);
  const { decision, score } = await checkUrlWithBackend(url);

  if (decision === 'PHISHING') {
    console.warn(`[ProtectU] Phishing! (${(score*100).toFixed(1)}%)`);
    showBanner(tabId, url, decision, score);
  } else {
    console.log(`[ProtectU] Safe (${(score*100).toFixed(1)}%)`);
    clearBanner(tabId);
  }

  // Log for popup/history
  const entry = {
    url,
    verdict: decision === 'PHISHING' ? 'Phishing' : 'Safe',
    predictionScore: score,
    timestamp: new Date().toISOString()
  };
  chrome.storage.local.get({ alerts: [] }, ({ alerts }) => {
    chrome.storage.local.set({ alerts: [...alerts, entry] });
  });

  // No longer report to backend
  console.log('[ProtectU] Not reporting to backend (report-risky-url removed).');
}

/**
 * Listen for email check requests from content or popup scripts.
 */
chrome.runtime.onMessage.addListener((msg, sender, sendResp) => {
  if (msg.type === 'EMAIL_CHECK') {
    fetch('http://localhost:5030/predict_email', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ email: msg.email })
    })
    .then(r => r.json())
    .then(data => {
      chrome.tabs.sendMessage(sender.tab.id, {
        type: 'EMAIL_RESULT',
        result: data
      });

      // --- Add this block to persist email alerts ---
      const entry = {
        type: 'email',
        verdict: data.prediction === 'Phishing Email' ? 'Phishing' : 'Safe',
        predictionScore: data.confidence,
        emailSnippet: (msg.email || '').slice(0, 100), // Save a snippet for display
        timestamp: new Date().toISOString()
      };
      chrome.storage.local.get({ alerts: [] }, ({ alerts }) => {
        chrome.storage.local.set({ alerts: [...alerts, entry] });
      });
      // --- end block ---
    });
    return true; // keep sendResp alive
  }
});

// Full-page navigations
chrome.webNavigation.onCompleted.addListener(({ tabId, url, frameId }) => {
  if (frameId === 0) handleUrl(tabId, url);
});

// SPA history API changes
chrome.webNavigation.onHistoryStateUpdated.addListener(({ tabId, url, frameId }) => {
  if (frameId === 0) handleUrl(tabId, url);
});

// Reloads and manual URL edits
chrome.tabs.onUpdated.addListener((tabId, changeInfo, tab) => {
  if (changeInfo.status === 'complete' && tab.url && tab.active) {
    handleUrl(tabId, tab.url);
  }
});

// Cleanup on tab close
chrome.tabs.onRemoved.addListener((tabId) => {
  delete lastLoggedUrl[tabId];
});

