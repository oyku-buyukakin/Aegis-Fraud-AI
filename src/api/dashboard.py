"""HTML for the Aegis Fraud Detection dashboard (kept in sync with notebook 10)."""

DASHBOARD_HTML = r"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>Aegis Fraud Detection API</title>
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Space+Grotesk:wght@500;600;700&display=swap');
  :root{
    --bg:#06070d; --card:rgba(22,25,42,.5); --field:#0b0d17;
    --border:rgba(255,255,255,.08); --border-hi:rgba(125,140,255,.45);
    --text:#eef1f8; --muted:#8b91a8; --accent:#7d8bff; --accent2:#b06bff;
    --good:#7af0c0; --bad:#ff7a90;
  }
  *{box-sizing:border-box;margin:0;padding:0;}
  html{scroll-behavior:smooth;}
  body{
    background:var(--bg); color:var(--text);
    font-family:'Inter',system-ui,sans-serif; -webkit-font-smoothing:antialiased;
    min-height:100vh; position:relative; overflow-x:hidden;
  }
  body::before,body::after{content:'';position:fixed;border-radius:50%;filter:blur(130px);z-index:0;pointer-events:none;}
  body::before{width:540px;height:540px;background:#4733c9;opacity:.30;top:-180px;left:-140px;}
  body::after{width:520px;height:520px;background:#1f54e0;opacity:.22;top:180px;right:-180px;}

  /* hero */
  .hero{position:relative;z-index:1;max-width:920px;margin:0 auto;padding:90px 28px 30px;text-align:center;}
  .pill{display:inline-flex;align-items:center;gap:8px;font-size:.72rem;font-weight:600;letter-spacing:.04em;
    color:var(--muted);background:rgba(255,255,255,.04);border:1px solid var(--border);
    padding:6px 14px;border-radius:999px;text-transform:uppercase;}
  .pill .dot{color:var(--bad);font-size:.6rem;}
  body.online .pill .dot{color:var(--good);}
  .hero h1{font-family:'Space Grotesk','Inter',sans-serif;font-weight:700;font-size:3.1rem;line-height:1.05;
    letter-spacing:-.02em;margin:22px 0 16px;}
  .grad{background:linear-gradient(110deg,var(--accent),var(--accent2));-webkit-background-clip:text;background-clip:text;color:transparent;}
  .hero .sub{color:var(--muted);font-size:1.02rem;line-height:1.65;max-width:620px;margin:0 auto;}

  /* metrics */
  .metrics{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;max-width:720px;margin:40px auto 0;}
  .metric{background:var(--card);border:1px solid var(--border);border-radius:14px;padding:18px 14px;backdrop-filter:blur(8px);}
  .mv{font-family:'Space Grotesk',sans-serif;font-size:1.5rem;font-weight:700;
    background:linear-gradient(110deg,var(--accent),var(--accent2));-webkit-background-clip:text;background-clip:text;color:transparent;}
  .ml{font-size:.66rem;color:var(--muted);text-transform:uppercase;letter-spacing:.08em;margin-top:6px;}

  /* main */
  .main{position:relative;z-index:1;max-width:720px;margin:0 auto;padding:54px 28px 10px;}
  .slbl{font-size:.68rem;font-weight:700;color:var(--muted);text-transform:uppercase;letter-spacing:.16em;margin-bottom:18px;text-align:center;}

  /* cards */
  .cards{display:flex;flex-direction:column;gap:14px;}
  .card{background:var(--card);border:1px solid var(--border);border-radius:18px;overflow:hidden;backdrop-filter:blur(10px);
    transition:border-color .25s,transform .25s,box-shadow .25s;}
  .card:hover{border-color:var(--border-hi);box-shadow:0 12px 40px -18px rgba(110,120,255,.6);}
  .card-hdr{padding:20px 22px;display:flex;align-items:center;gap:14px;cursor:pointer;user-select:none;}
  .desc{color:var(--text);font-size:1rem;font-weight:600;flex:1;letter-spacing:-.01em;}
  .chev{color:var(--muted);font-size:.7rem;margin-left:auto;transition:transform .25s;}
  .chev.open{transform:rotate(180deg);}

  /* panel */
  .panel{border-top:1px solid var(--border);padding:6px 22px 24px;display:none;animation:fade .3s ease;}
  .panel.open{display:block;}
  @keyframes fade{from{opacity:0;transform:translateY(-4px);}to{opacity:1;transform:none;}}
  label{display:block;font-size:.72rem;color:var(--muted);margin-bottom:6px;margin-top:14px;font-weight:500;}
  input,select,textarea{
    width:100%;background:var(--field);border:1px solid var(--border);border-radius:10px;
    color:var(--text);padding:11px 13px;font-size:.88rem;outline:none;font-family:inherit;
    transition:border-color .18s,box-shadow .18s;
  }
  input:focus,select:focus,textarea:focus{border-color:var(--accent);box-shadow:0 0 0 3px rgba(125,140,255,.15);}
  textarea{resize:vertical;min-height:72px;font-family:'Space Grotesk',monospace;}
  .send-btn{
    margin-top:18px;background:linear-gradient(110deg,var(--accent),var(--accent2));color:#fff;border:none;
    padding:11px 26px;border-radius:11px;font-size:.86rem;font-weight:600;cursor:pointer;font-family:inherit;
    transition:transform .15s,filter .15s;box-shadow:0 8px 24px -10px rgba(125,140,255,.8);
  }
  .send-btn:hover{transform:translateY(-1px);filter:brightness(1.08);}
  .send-btn:active{transform:translateY(0);}
  .res-box{
    margin-top:18px;background:rgba(8,10,18,.7);border:1px solid var(--border);border-left:3px solid var(--accent);
    border-radius:6px 12px 12px 6px;padding:16px 18px;font-size:.9rem;color:var(--text);line-height:1.75;
    white-space:pre-wrap;word-break:break-word;display:none;
  }
  .res-box.show{display:block;animation:fade .3s ease;}

  .footer{position:relative;z-index:1;text-align:center;color:var(--muted);font-size:.72rem;padding:40px 20px 50px;margin-top:40px;}

  @media(max-width:640px){
    .hero{padding-top:56px;} .hero h1{font-size:2.2rem;}
    .metrics{grid-template-columns:repeat(2,1fr);}
  }
</style>
</head>
<body>

<header class="hero">
  <div class="pill"><span class="dot" id="live-badge">⬤</span> Aegis · llama3.2</div>
  <h1>AI Supported Fraud Detection System<br/> <span class="grad"> </span></h1>
  <p class="sub">Analyze any transaction by amount, time, or email for instant risk insights. See exactly why a transaction was flagged, or ask our fraud policy knowledge base.</p>
</header>

<div class="main">
<div class="slbl">What can this API do?</div>
<div class="cards">

<!-- SCORE -->
<div class="card" id="c-score">
  <div class="card-hdr" onclick="toggle('score')">
    <span class="desc">Is this transaction risky?</span>
    <span class="chev" id="chev-score">▼</span>
  </div>
  <div class="panel" id="p-score">
    <label>Transaction ID</label><input id="sc-id" value="TXN-001"/>
    <label>Amount ($)</label><input id="sc-amt" type="number" value="875"/>
    <label>Hour of day (0–23)</label><input id="sc-hour" type="number" value="2"/>
    <label>Account age (days)</label><input id="sc-age" type="number" value="1"/>
    <label>Transactions in last 1 hour</label><input id="sc-vel" type="number" value="9"/>
    <label>Email domain</label><input id="sc-email" value="protonmail.com"/>
    <label>Card type</label>
    <select id="sc-card"><option value="credit">Credit card</option><option value="debit">Debit card</option></select>
    <label>Country mismatch?</label>
    <select id="sc-cm"><option value="1">Yes</option><option value="0">No</option></select>
    <button class="send-btn" onclick="callScore()">Check transaction</button>
    <div class="res-box" id="r-score"></div>
  </div>
</div>

<!-- RULES -->
<div class="card" id="c-rules">
  <div class="card-hdr" onclick="toggle('rules')">
    <span class="desc">Which fraud rules can flag a transaction?</span>
    <span class="chev" id="chev-rules">▼</span>
  </div>
  <div class="panel" id="p-rules">
    <p style="color:var(--muted);font-size:.82rem;margin-bottom:6px;line-height:1.5;">See the rules that mark a transaction as fraud</p>
    <button class="send-btn" onclick="callRules()">Show fraud rules</button>
    <div class="res-box" id="r-rules"></div>
  </div>
</div>

<!-- RAG -->
<div class="card" id="c-rag">
  <div class="card-hdr" onclick="toggle('rag')">
    <span class="desc">Ask anything about fraud policies or rules.</span>
    <span class="chev" id="chev-rag">▼</span>
  </div>
  <div class="panel" id="p-rag">
    <label>Your question</label>
    <input id="rq-q" value="What happens when a new account makes a high-value transaction?"/>
    <button class="send-btn" onclick="callRAG()">Search knowledge base</button>
    <div class="res-box" id="r-rag"></div>
  </div>
</div>

</div><!-- /cards -->

</div><!-- /main -->

<div class="footer">Aegis Fraud Detection API · v1.0.0 · Local LLM: llama3.2 </div>

<script>
function toggle(id){
  const p=document.getElementById('p-'+id);
  const c=document.getElementById('chev-'+id);
  if(!p || !c) return;
  p.classList.toggle('open');
  c.classList.toggle('open');
}

function show(id, text){
  const box=document.getElementById('r-'+id);
  box.textContent = text;
  box.classList.add('show');
}

function gv(id){return document.getElementById(id).value;}
function gi(id){return parseFloat(document.getElementById(id).value);}

function txBody(prefix){
  return {
    transaction_id: gv(prefix+'-id'),
    TransactionAmt: gi(prefix+'-amt'),
    hour: parseInt(gv(prefix+'-hour')),
    account_age_days: parseInt(gv(prefix+'-age')),
    num_txn_last_1h: parseInt(gv(prefix+'-vel')),
    P_emaildomain: gv(prefix+'-email'),
    card_type: gv(prefix+'-card'),
    country_mismatch: parseInt(gv(prefix+'-cm')),
  };
}

async function post(path, body){
  try{
    const r = await fetch(path,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
    return await r.json();
  }catch(e){ return {error: e.toString()}; }
}

function fmtScore(d){
  if(d.error) return 'Error: ' + d.error;
  const pct = Math.round((d.anomaly_score??0)*100);
  const level = (d.risk_level ?? 'unknown').toUpperCase();
  const sigs = d.triggered_signals ?? [];
  const lines = [
    'Transaction:  ' + (d.transaction_id ?? '—'),
    'Risk score:   ' + pct + ' out of 100',
    'Risk level:   ' + level,
    '',
  ];
  if(sigs.length){
    lines.push('Signals that raised the score:');
    sigs.forEach(s => lines.push('  • ' + s.replace(/_/g,' ')));
  } else {
    lines.push('No suspicious signals detected.');
  }
  return lines.join('\n');
}

function fmtRules(d){
  if(d.error) return 'Error: ' + d.error;
  const rules = (d.rules ?? []).filter(r => r.action === 'FLAG_CRITICAL_REVIEW');
  if(!rules.length) return 'No fraud rules found.';
  const lines = ['These are the rules that flag a transaction as fraud:', ''];
  rules.forEach(r => {
    lines.push('• ' + (r.name ?? r.rule_id));
    if(r.description) lines.push('   ' + r.description);
    lines.push('');
  });
  return lines.join('\n');
}

function fmtRAG(d){
  if(d.error) return 'Error: ' + d.error;
  const results = d.results ?? [];
  if(!results.length) return 'No specific fraud policy was found for this question. Try rephrasing, or ask about a known fraud type such as velocity, card-not-present, account takeover, or off-hours activity.';
  return results.map(r => {
    let s = (r.text ?? '')
      .replace(/\\n/g, ' ')
      .replace(/^#+\s*/gm, '')
      .replace(/\s+/g, ' ')
      .trim();
    const m = s.match(/^(.*?)\s*-\s*Priority:.*?-\s*Description:\s*(.*?)(?:\s*-\s*Tags:|\s*-\s*Conditions:|$)/i);
    if(m){
      const name = m[1].trim().replace(/^RULE_\w+\s*-\s*/i, '');
      s = name + '\n' + m[2].trim();
    }
    return s;
  }).join('\n\n');
}

function callScore(){
  const body = txBody('sc');
  Promise.all([post('/score', body), post('/explain', body)]).then(([s, e])=>{
    let text = fmtScore(s);
    if(!e.error && e.explanation){
      text += '\n\n Why was this flagged? \n' + e.explanation;
    }
    show('score', text);
  });
}

function callRules(){
  fetch('/rules/list').then(r=>r.json()).then(d=>show('rules', fmtRules(d))).catch(e=>show('rules','⚠️ '+e.toString()));
}

function callRAG(){
  post('/rag/query',{query:gv('rq-q'),top_k:1}).then(d=>show('rag', fmtRAG(d)));
}

// live status indicator
fetch('/health').then(r=>r.json()).then(()=>{
  document.body.classList.add('online');
}).catch(()=>{ document.body.classList.remove('online'); });
</script>
</body>
</html>
"""