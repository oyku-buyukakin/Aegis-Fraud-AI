# API Development (FastAPI)


```python
import sys
import threading
import time
from pathlib import Path

import httpx
import nest_asyncio
import pandas as pd
import uvicorn
from IPython.display import display

ROOT = Path('..').resolve()
sys.path.insert(0, str(ROOT))

nest_asyncio.apply()

from src.api.main import create_app
from src.api.schemas import ScoreRequest

```


```python
API_HOST = '127.0.0.1'
API_PORT = 8765
API_BASE = f'http://{API_HOST}:{API_PORT}'

app = create_app()
print('Scoring model:', app.container.scoring_service().model_name())

```

### Start the Server


```python
def _start_server():
    uvicorn.run(app, host=API_HOST, port=API_PORT, log_level='error')

server_thread = threading.Thread(target=_start_server, daemon=True)
server_thread.start()

for _ in range(20):
    try:
        r = httpx.get(f'{API_BASE}/health', timeout=2)
        if r.status_code == 200:
            print(f'Server is up: {API_BASE}')
            print('Health:', r.json())
            break
    except Exception:
        time.sleep(0.5)
else:
    print('Server did not start.')

```

### Test — score


```python
score_payloads = [
    {
        'transaction_id': 'TXN-001',
        'TransactionAmt': 42.50,
        'hour': 14,
        'account_age_days': 220,
        'num_txn_last_1h': 1,
        'P_emaildomain': 'gmail.com',
        'card_type': 'debit',
        'country_mismatch': 0,
    },
    {
        'transaction_id': 'TXN-002',
        'TransactionAmt': 875.00,
        'hour': 2,
        'account_age_days': 1,
        'num_txn_last_1h': 9,
        'P_emaildomain': 'protonmail.com',
        'card_type': 'credit',
        'country_mismatch': 1,
    },
]

score_rows = []
for payload in score_payloads:
    r = httpx.post(f'{API_BASE}/score', json=payload)
    score_rows.append(r.json())
    print(f'POST /score  →  {r.status_code}  {r.json()}')

display(pd.DataFrame(score_rows))

```

### Test — explain


```python
explain_payload = score_payloads[1]
r = httpx.post(f'{API_BASE}/explain', json=explain_payload)
result = r.json()
print(f"transaction_id : {result['transaction_id']}")
print(f"anomaly_score  : {result['anomaly_score']}")
print(f"risk_level     : {result['risk_level']}")
print(f"explanation    :\n  {result['explanation']}")

```

### Test — rules/evaluate


```python
rules_payload = {
    'transaction_id': 'TXN-003',
    'features': {
        'adjusted_anomaly_score': 0.82,
        'anomaly_high_flag_count': 11,
        'is_high_risk_entity': 1,
        'final_context_anomaly_score': 0.77,
        'is_outlier': 1,
        'ctx_amount_zscore': 4.5,
        'TransactionAmt': 310.0,
        'is_business_hour': 0,
        'anomaly_high_flag_ratio': 0.40,
        'time_since_first_transaction': 3600,
        'is_night_transaction': 1,
        'entity_trusted_score': 0.22,
        'ctx_amount_vs_product_avg': 350,
        'ctx_amount_ratio_to_global_median': 6.2,
    },
}

r = httpx.post(f'{API_BASE}/rules/evaluate', json=rules_payload)
result = r.json()
print(f"final_action   : {result['final_action']}")
print(f"final_severity : {result['final_severity']}")
print(f"fired_rules    : {len(result['fired_rules'])}")
display(pd.DataFrame(result['fired_rules'])[['rule_id', 'name', 'severity', 'action']])

```

### Test — rules/list


```python
r = httpx.get(f'{API_BASE}/rules/list')
catalog = r.json()
display(pd.DataFrame(catalog['rules'])[['rule_id', 'name', 'severity', 'action']].head())

```

### Test — rag/query


```python
rag_queries = [
    'What happens when a new account makes a high-value transaction?',
    'What happens when an old account makes a high-value transaction?',
    'How is conflict resolution handled when multiple rules fire?',
]

for query in rag_queries:
    r = httpx.post(f'{API_BASE}/rag/query', json={'query': query, 'top_k': 1})
    result = r.json()
    print(f"\nQuery : {query}")
    print(f"Status: {r.status_code}  |  Results: {len(result['results'])}")
    for chunk in result['results']:
        print(f"  score={chunk['score']:.4f}  source={chunk['source']}")
        print(f"    {chunk['text'][:120].replace(chr(10), ' ')}...")

```
