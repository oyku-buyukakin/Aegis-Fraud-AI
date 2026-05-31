# Agentic AI


```python
import numpy as np
import pandas as pd
import json
import math
import textwrap
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any
from IPython.display import display
import ollama
import warnings
warnings.filterwarnings("ignore")
```


```python
USE_LOCAL_LLM = True
OLLAMA_MODEL = "llama3.2"
```

### Synthetic Fraud Transactions


```python
transactions = pd.DataFrame([
    {
        "transaction_id": "TXN-001",
        "customer_id": "C001",
        "TransactionAmt": 42.50,
        "hour": 14,
        "account_age_days": 220,
        "num_txn_last_1h": 1,
        "P_emaildomain": "gmail.com",
        "card_type": "debit",
        "country_mismatch": 0,
        "device_trust_score": 0.91,
    },
    {
        "transaction_id": "TXN-002",
        "customer_id": "C002",
        "TransactionAmt": 875.00,
        "hour": 2,
        "account_age_days": 1,
        "num_txn_last_1h": 9,
        "P_emaildomain": "protonmail.com",
        "card_type": "credit",
        "country_mismatch": 1,
        "device_trust_score": 0.18,
    },
    {
        "transaction_id": "TXN-003",
        "customer_id": "C003",
        "TransactionAmt": 310.20,
        "hour": 23,
        "account_age_days": 12,
        "num_txn_last_1h": 4,
        "P_emaildomain": "mailinator.com",
        "card_type": "credit",
        "country_mismatch": 0,
        "device_trust_score": 0.42,
    },
])
```

### Agent Communication Layer


```python
@dataclass
class AgentMessage:
    sender: str
    receiver: str
    task: str
    payload: dict[str, Any]
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))


@dataclass
class Blackboard:
    transaction_id: str
    raw_transaction: dict[str, Any]
    features: dict[str, Any] = field(default_factory=dict)
    anomaly: dict[str, Any] = field(default_factory=dict)
    decision: dict[str, Any] = field(default_factory=dict)
    explanation: str = ""
    messages: list[AgentMessage] = field(default_factory=list)

    def send(self, sender: str, receiver: str, task: str, payload: dict[str, Any]) -> None:
        self.messages.append(
            AgentMessage(sender=sender, receiver=receiver, task=task, payload=payload)
        )


class BaseAgent:
    name = "base_agent"

    def run(self, board: Blackboard) -> Blackboard:
        raise NotImplementedError
```

### Specialized Agents

Paralel agents: AmountTimeSignalAgent,IdentitySignalAgent,VelocitySignalAgent

Downstream agents run with dependencies: AnomalyAgent, DecisionAgent,ExplanationAgent


```python
class AmountTimeSignalAgent(BaseAgent):
    name = "amount_time_signal_agent"

    def run(self, board: Blackboard) -> Blackboard:
        tx = board.raw_transaction
        features = {
            "is_night_transaction": int(tx["hour"] < 6 or tx["hour"] >= 22),
            "is_high_amount": int(tx["TransactionAmt"] >= 500),
            "uses_credit_card": int(tx["card_type"] == "credit"),
        }
        board.features.update(features)
        board.send(
            sender=self.name,
            receiver="anomaly_agent",
            task="amount_time_features_complete",
            payload=features,
        )
        return board


class IdentitySignalAgent(BaseAgent):
    name = "identity_signal_agent"

    def run(self, board: Blackboard) -> Blackboard:
        tx = board.raw_transaction
        email_domain = str(tx.get("P_emaildomain", "")).lower()
        features = {
            "uses_disposable_email": int(email_domain in {"protonmail.com", "mailinator.com", "guerrillamail.com", "tempmail.com"}),
            "country_mismatch": int(tx["country_mismatch"]),
            "low_device_trust": int(tx["device_trust_score"] < 0.45),
        }
        board.features.update(features)
        board.send(
            sender=self.name,
            receiver="anomaly_agent",
            task="identity_features_complete",
            payload=features,
        )
        return board


class VelocitySignalAgent(BaseAgent):
    name = "velocity_signal_agent"

    def run(self, board: Blackboard) -> Blackboard:
        tx = board.raw_transaction
        features = {
            "is_new_account": int(tx["account_age_days"] <= 7),
            "has_velocity_spike": int(tx["num_txn_last_1h"] >= 5),
        }
        board.features.update(features)
        board.send(
            sender=self.name,
            receiver="anomaly_agent",
            task="velocity_features_complete",
            payload=features,
        )
        return board


class AnomalyAgent(BaseAgent): #anomaly scoring agent
    name = "anomaly_agent"

    weights = {
        "is_night_transaction": 0.10,
        "is_new_account": 0.15,
        "is_high_amount": 0.15,
        "has_velocity_spike": 0.20,
        "uses_disposable_email": 0.15,
        "uses_credit_card": 0.05,
        "country_mismatch": 0.10,
        "low_device_trust": 0.10,
    }

    def run(self, board: Blackboard) -> Blackboard:
        score = sum(board.features.get(name, 0) * weight for name, weight in self.weights.items())
        score = min(float(score), 1.0)

        triggered_signals = [name for name, value in board.features.items() if value == 1]
        board.anomaly = {
            "anomaly_score": round(score, 3),
            "triggered_signals": triggered_signals,
            "signal_count": len(triggered_signals),
        }
        board.send(
            sender=self.name,
            receiver="decision_agent",
            task="anomaly_scoring_complete",
            payload=board.anomaly,
        )
        return board


class DecisionAgent(BaseAgent): #decision making agent
    name = "decision_agent"

    def run(self, board: Blackboard) -> Blackboard:
        score = board.anomaly["anomaly_score"]
        signals = set(board.anomaly["triggered_signals"])

        if score >= 0.65 or {"is_high_amount", "uses_credit_card", "low_device_trust"}.issubset(signals):
            action = "BLOCK_OR_MANUAL_REVIEW"
            severity = "HIGH"
        elif score >= 0.35:
            action = "FLAG_REVIEW"
            severity = "MEDIUM"
        else:
            action = "APPROVE"
            severity = "LOW"

        board.decision = {
            "action": action,
            "severity": severity,
            "reason_code_count": len(signals),
        }
        board.send(
            sender=self.name,
            receiver="explanation_agent", 
            task="decision_complete",
            payload=board.decision,
        )
        return board
```


```python
def call_local_ollama(prompt: str, model: str = OLLAMA_MODEL) -> str:

    response = ollama.generate(
        model=model,
        prompt=prompt,
        options={
            "num_predict": 120,
            "num_ctx": 1536,
            "temperature": 0.1,
            "num_thread": 2,
        },
    )
    return response.response.strip()


class ExplanationAgent(BaseAgent):
    name = "explanation_agent"

    def run(self, board: Blackboard) -> Blackboard:
        tx = board.raw_transaction

        prompt = f"""
You are a local fraud analyst LLM running inside an agentic AI workflow.
Use only the structured data below. Do not invent external facts.

Transaction:
{json.dumps(tx, indent=2)}

Feature agent output:
{json.dumps(board.features, indent=2)}

Anomaly agent output:
{json.dumps(board.anomaly, indent=2)}

Decision agent output:
{json.dumps(board.decision, indent=2)}

Write a short explanation with:
1. Main fraud reason
2. Risk level
3. Recommended action
"""

        board.explanation = call_local_ollama(prompt)
        board.send(
            sender=self.name,
            receiver="fraud_orchestrator",
            task="local_llm_explanation_complete",
            payload={"model": OLLAMA_MODEL, "explanation": board.explanation},
        )
        return board
```

### Fraud Orchestrator


```python
class FraudOrchestrator:
    def __init__(self, stages: list[list[BaseAgent]]):
        self.stages = stages
        self.agents = [agent for stage in stages for agent in stage]

    def _run_stage(self, board: Blackboard, stage_idx: int, agents: list[BaseAgent]) -> Blackboard:
        stage_name = f"stage_{stage_idx}"

        for agent in agents:
            board.send(
                sender="fraud_orchestrator",
                receiver=agent.name,
                task="delegate_parallel_task" if len(agents) > 1 else "delegate_task",
                payload={"transaction_id": board.transaction_id, "stage": stage_name},
            )

        if len(agents) == 1:
            return agents[0].run(board)

        with ThreadPoolExecutor(max_workers=len(agents)) as executor:
            futures = {executor.submit(agent.run, board): agent.name for agent in agents}
            for future in as_completed(futures):
                agent_name = futures[future]
                board = future.result()
                board.send(
                    sender="fraud_orchestrator",
                    receiver="all_agents",
                    task="parallel_agent_finished",
                    payload={"stage": stage_name, "agent": agent_name},
                )

        board.send(
            sender="fraud_orchestrator",
            receiver="all_agents",
            task="parallel_stage_complete",
            payload={"stage": stage_name, "agent_count": len(agents)},
        )
        return board

    def run_transaction(self, transaction: dict[str, Any]) -> Blackboard:
        board = Blackboard(
            transaction_id=transaction["transaction_id"],
            raw_transaction=transaction,
        )

        for stage_idx, agents in enumerate(self.stages, start=1):
            board = self._run_stage(board, stage_idx, agents)

        return board

    def run_batch(self, transactions_df: pd.DataFrame) -> list[Blackboard]:
        # Transactions are kept sequential so Ollama is not called many times at once on Mac M1.
        results = []
        for record in transactions_df.to_dict(orient="records"):
            results.append(self.run_transaction(record))
        return results


orchestrator = FraudOrchestrator(
    stages=[
        [AmountTimeSignalAgent(), IdentitySignalAgent(), VelocitySignalAgent()],
        [AnomalyAgent()],
        [DecisionAgent()],
        [ExplanationAgent()],
    ]
)
```

### Multi-Agent Workflow


```python
boards = orchestrator.run_batch(transactions)

summary_rows = []
for board in boards:
    summary_rows.append({
        "transaction_id": board.transaction_id,
        "risk_score": board.anomaly["anomaly_score"],
        "severity": board.decision["severity"],
        "action": board.decision["action"],
        "triggered_signals": ", ".join(board.anomaly["triggered_signals"]),
        "explanation": board.explanation,
    })

agentic_summary_df = pd.DataFrame(summary_rows)
display(agentic_summary_df)
```

### Agent-to-Agent Communication


```python
message_rows = []
for board in boards:
    for message in board.messages:
        message_rows.append({
            "transaction_id": board.transaction_id,
            "timestamp": message.timestamp,
            "sender": message.sender,
            "receiver": message.receiver,
            "task": message.task,
            "payload": json.dumps(message.payload)[:160],
        })

messages_df = pd.DataFrame(message_rows)
messages_df.tail(10)
```

This notebook demonstrates a lightweight multi-agent orchestration layer on synthetic fraud transactions. The goal is to show the agentic AI architecture, parallel feature agents, anomaly scoring, decision making, explanation generation, task delegation, and agent-to-agent communication. The same orchestration pattern can be connected to the full fraud dataset by replacing the synthetic transaction input with scored transaction records from the previous pipeline steps.


The multi-agent workflow is implemented as an orchestration prototype using synthetic fraud cases to demonstrate agent-to-agent communication and task delegation.
