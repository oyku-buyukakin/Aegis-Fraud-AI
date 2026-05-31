# RAG Pipeline


```python
import numpy as np
import pandas as pd
import yaml
import gc
import hashlib
import json
import re
import textwrap
from pathlib import Path
import faiss
from IPython.display import display
import ollama
from huggingface_hub import logging as hf_logging
import warnings
warnings.filterwarnings("ignore")
import os

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

hf_logging.set_verbosity_error()
```


```python
OLLAMA_MODEL = "llama3.2"
SKIP_LLM = False
USE_TRANSFORMER_EMBEDDINGS = False  #For local resource efficiency, the default embedding backend is a lightweight HashingEmbedder. 
#The pipeline is designed to support transformer-based semantic embeddings by setting USE_TRANSFORMER_EMBEDDINGS=True.
EMBED_MODEL_NAME = "paraphrase-MiniLM-L3-v2"
EMBED_BATCH = 2
HASH_EMBED_DIM = 512
```


```python
KB_DIR    = Path("../docs/knowledge_base")
CACHE_DIR = Path("../data/interim")
```

### Knowledge Base


```python
def load_knowledge_base(kb_dir: Path) -> list[dict]:
    
    documents = []

    for path in sorted(kb_dir.iterdir()):
        if path.suffix == ".md":
            text = path.read_text(encoding="utf-8").strip()
            documents.append({"source": path.name, "text": text})

        elif path.suffix == ".json":
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, list):
                for item in data:
                    text = json.dumps(item, ensure_ascii=False, indent=2)
                    documents.append({"source": path.name, "text": text})
            else:
                documents.append({
                    "source": path.name,
                    "text": json.dumps(data, ensure_ascii=False, indent=2)
                })

        elif path.suffix in (".yaml", ".yml"):
            data = yaml.safe_load(path.read_text(encoding="utf-8"))
            documents.append({
                "source": path.name,
                "text": yaml.dump(data, allow_unicode=True, sort_keys=False)
            })

    return documents


raw_docs = load_knowledge_base(KB_DIR)
```


```python
doc_summary = pd.DataFrame([
    {
        "document_number": i + 1,
        "source_file": doc["source"],
        "text_length": len(doc["text"])
    }
    for i, doc in enumerate(raw_docs)
])

display(doc_summary)

source_summary = (
    doc_summary
    .groupby("source_file", as_index=False)
    .agg(document_count=("document_number", "count"), total_text_length=("text_length", "sum"))
)

display(source_summary)
```

### Test Rules in the Knowledge Base


```python
test_rules_doc = next((d for d in raw_docs if d["source"] == "extended_test_rules.md"), None)
```


```python
def chunk_document(
    doc: dict,
    chunk_size: int = 512,
    chunk_overlap: int = 64
) -> list[dict]:
    
    text = doc["text"]
    source = doc["source"]
    chunks = []

    sections = re.split(r"\n(?=#{1,3} )|(?<=\n)---+\n", text)

    for sec_idx, section in enumerate(sections):
        section = section.strip()
        if not section:
            continue

        if len(section) <= chunk_size:
            chunks.append({
                "source": source,
                "chunk_id": f"{source}::sec{sec_idx}",
                "text": section
            })
        else:
            start = 0
            sub_idx = 0
            while start < len(section):
                end = min(start + chunk_size, len(section))
                chunk_text = section[start:end].strip()
                if chunk_text:
                    chunks.append({
                        "source": source,
                        "chunk_id": f"{source}::sec{sec_idx}::chunk{sub_idx}",
                        "text": chunk_text
                    })
                sub_idx += 1
                start += chunk_size - chunk_overlap

    return chunks


all_chunks = []
for doc in raw_docs:
    all_chunks.extend(chunk_document(doc))
```

### Embedding Generation


```python
class HashingEmbedder:

    def __init__(self, dim: int = 512):
        self.dim = dim

    def get_sentence_embedding_dimension(self) -> int:
        return self.dim

    def encode(
        self,
        texts,
        batch_size: int = 1,
        show_progress_bar: bool = False,
        normalize_embeddings: bool = True,
        convert_to_numpy: bool = True,
    ):
        if isinstance(texts, str):
            texts = [texts]

        vectors = np.zeros((len(texts), self.dim), dtype="float32")
        for row_idx, text in enumerate(texts):
            tokens = re.findall(r"[a-zA-Z0-9_]+", text.lower())
            for token in tokens:
                digest = hashlib.md5(token.encode("utf-8")).digest()
                col_idx = int.from_bytes(digest[:4], "little") % self.dim
                sign = 1.0 if digest[4] % 2 == 0 else -1.0
                vectors[row_idx, col_idx] += sign

        if normalize_embeddings:
            norms = np.linalg.norm(vectors, axis=1, keepdims=True)
            vectors = vectors / np.maximum(norms, 1e-12)

        return vectors if convert_to_numpy else vectors.tolist()


if USE_TRANSFORMER_EMBEDDINGS:
    import torch
    torch.set_num_threads(2)
    from sentence_transformers import SentenceTransformer

    embed_model = SentenceTransformer(EMBED_MODEL_NAME, device="cpu")
    embed_model.eval()
    embedding_backend = EMBED_MODEL_NAME
else:
    embed_model = HashingEmbedder(dim=HASH_EMBED_DIM)
    embedding_backend = f"hashing-{HASH_EMBED_DIM}d"
```


```python
texts = [chunk["text"] for chunk in all_chunks]

CACHE_DIR.mkdir(parents=True, exist_ok=True)
_safe_backend_name = re.sub(r"[^a-zA-Z0-9_-]+", "_", embedding_backend)
_emb_cache = CACHE_DIR / f"kb_embeddings_{_safe_backend_name}.npy"
_txt_cache = CACHE_DIR / f"kb_chunk_texts_{_safe_backend_name}.json"

_cached_texts_match = (
    _emb_cache.exists()
    and _txt_cache.exists()
    and json.loads(_txt_cache.read_text()) == texts
)

if _cached_texts_match:
    embeddings = np.load(str(_emb_cache))
    print(f"Loaded embeddings from cache {embeddings.shape}; skipping encode.")
else:
    print(f"Encoding {len(texts)} chunks with {embedding_backend}...")
    all_vecs = []
    for i in range(0, len(texts), EMBED_BATCH):
        batch = texts[i : i + EMBED_BATCH]
        vecs = embed_model.encode(
            batch,
            batch_size=EMBED_BATCH,
            show_progress_bar=False,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        all_vecs.append(vecs.astype("float32"))
        gc.collect()
        print(f"  encoded {min(i + EMBED_BATCH, len(texts))}/{len(texts)}", end="\r")

    embeddings = np.vstack(all_vecs).astype("float32")
    np.save(str(_emb_cache), embeddings)
    _txt_cache.write_text(json.dumps(texts))
    print("\nEmbeddings saved to cache.")

print(f"\nEmbedding matrix shape: {embeddings.shape}")
```

### FAISS Vector Store


```python
embed_dim = embeddings.shape[1]
_idx_cache = CACHE_DIR / f"kb_faiss_{_safe_backend_name}.index"

if _cached_texts_match and _idx_cache.exists():
    faiss_index = faiss.read_index(str(_idx_cache))
    print("Loaded FAISS index from cache.")
else:
    faiss_index = faiss.IndexFlatIP(embed_dim)
    faiss_index.add(embeddings)
    faiss.write_index(faiss_index, str(_idx_cache))
    print("FAISS index built and saved to cache.")

print(f"FAISS index type  : IndexFlatIP (inner product)")
print(f"Embedding dim     : {embed_dim}")
print(f"Vectors indexed   : {faiss_index.ntotal}")
```

### Vector Search


```python
def vector_search(
    query: str,
    faiss_index,
    chunks: list[dict],
    embed_model,
    top_k: int = 5
) -> list[dict]:

    query_embedding = embed_model.encode(
        [query],
        normalize_embeddings=True
    ).astype("float32")

    scores, indices = faiss_index.search(query_embedding, top_k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue
        results.append({
            "rank": len(results) + 1,
            "score": float(score),
            "source": chunks[idx]["source"],
            "chunk_id": chunks[idx]["chunk_id"],
            "text": chunks[idx]["text"]
        })

    return results
```


```python
demo_queries = [
    "What action should be taken for a velocity fraud transaction?",
    "How is conflict resolution handled when multiple rules fire?",]
```


```python
for query in demo_queries:
    results = vector_search(query, faiss_index, all_chunks, embed_model, top_k=3)
    for r in results:
        preview = r["text"][:160].replace("\n", " ")
        print(f"  Rank {r['rank']} | score={r['score']:.4f} | source={r['source']}")
```

### LLM Context Injection


```python
def build_rag_prompt(query: str, retrieved_chunks: list[dict]) -> str:

    context_blocks = []
    for r in retrieved_chunks:
        context_blocks.append(
            f"[Source: {r['source']} | Similarity: {r['score']:.4f}]\n{r['text']}"
        )
    context_text = "\n\n---\n\n".join(context_blocks)

    prompt = f"""You are an expert fraud analyst assistant for the Aegis fraud detection system.
Use only the knowledge base context below to answer the question.
If the context does not contain enough information, say so clearly.
Do not make up rules or policies that are not in the context.

=== KNOWLEDGE BASE CONTEXT ===

{context_text}

=== END OF CONTEXT ===

QUESTION: {query}

ANSWER:"""
    return prompt
```


```python
demo_query = "What should happen when a new account makes a high-value transaction at night?"
retrieved = vector_search(demo_query, faiss_index, all_chunks, embed_model, top_k=4)
rag_prompt = build_rag_prompt(demo_query, retrieved)

print(rag_prompt)
```

### Ollama


```python
def check_ollama(model: str = OLLAMA_MODEL) -> bool:
    try:
        available = [m.model for m in ollama.list().models]
        return any(model in m for m in available)
    except Exception as exc:
        print(f"Ollama connection check failed: {exc}")
        return False


def ollama_generate(prompt: str, model: str = OLLAMA_MODEL) -> str:
    response = ollama.generate(
        model=model,
        prompt=prompt,
        options={
            "num_predict": 180,   
            "num_ctx": 2048,      
            "temperature": 0.1,
            "num_thread": 2,
        },
    )
    return response.response.strip()


if SKIP_LLM:
    ollama_available = False
    print("SKIP_LLM=True — Ollama calls disabled. Set SKIP_LLM=False to enable.")
else:
    ollama_available = check_ollama(OLLAMA_MODEL)
    if ollama_available:
        print(f"Ollama is running. Model available: {OLLAMA_MODEL}")
    else:
        print(f"Ollama model '{OLLAMA_MODEL}' is not available.")
```


```python
def rag_pipeline(
    query: str,
    faiss_index,
    chunks: list[dict],
    embed_model,
    top_k: int = 5,
    model: str = OLLAMA_MODEL,
    ollama_available: bool = True,
    verbose: bool = True
) -> dict:

    retrieved = vector_search(query, faiss_index, chunks, embed_model, top_k=top_k)

    prompt = build_rag_prompt(query, retrieved)

    if ollama_available:
        answer = ollama_generate(prompt, model=model)
    else:
        answer = (
            "[Ollama not available — showing injected context prompt only]\n\n"
            + prompt
        )

    if verbose:
        print(f"RAG QUERY: {query}")
        print("\n--- Retrieved Chunks ---")
        for r in retrieved:
            preview = r["text"][:120].replace("\n", " ")
            print(f"  [{r['rank']}] score={r['score']:.4f}  source={r['source']}")
            print(f"       {preview}...")
        print("\n--- LLM Answer ---")
        wrapped = textwrap.fill(answer, width=88)
        print(wrapped)
        print()

    return {
        "query": query,
        "retrieved_chunks": retrieved,
        "prompt": prompt,
        "answer": answer
    }
```

### RAG Pipeline


```python
rag_test_queries = ["Is there a rule for card-not-present fraud on high-value credit card transactions?",]

rag_results = []

for query in rag_test_queries:
    result = rag_pipeline(
        query=query,
        faiss_index=faiss_index,
        chunks=all_chunks,
        embed_model=embed_model,
        top_k=2,
        model=OLLAMA_MODEL,
        ollama_available=ollama_available,
        verbose=True,)
        
    rag_results.append(result)
```
