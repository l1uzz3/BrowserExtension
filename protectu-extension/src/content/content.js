// content.js

chrome.runtime.onMessage.addListener((msg) => {
  if (msg.type === 'CLEAR_BANNER') {
    const b = document.getElementById('protectu-warning');
    if (b) b.remove();
    return;
  }

  if (msg.type !== 'RISKY_SITE' || msg.decision !== 'PHISHING') return;
  if (document.getElementById('protectu-warning')) return;

  const banner = document.createElement('div');
  banner.id = 'protectu-warning';
  Object.assign(banner.style, {
    position: 'fixed', top: '0', left: '0', width: '100%',
    backgroundColor: '#B00020', color: '#FFF', padding: '12px 16px',
    fontFamily: 'Arial,sans-serif', fontSize: '16px',
    display: 'flex', justifyContent: 'space-between', alignItems: 'center',
    boxShadow: '0 2px 8px rgba(0,0,0,0.3)', zIndex: '2147483647'
  });

  const text = document.createElement('span');
  text.innerText = '⚠️ Warning: This website appears unsafe.';

  const closeBtn = document.createElement('button');
  closeBtn.innerText = '✕';
  Object.assign(closeBtn.style, {
    background: 'transparent', border: 'none', color: '#FFF',
    fontSize: '18px', cursor: 'pointer'
  });
  closeBtn.onclick = () => banner.remove();

  banner.append(text, closeBtn);
  document.documentElement.prepend(banner);
});
