# Research Continuity Agent — Roadmap

## Goal
A personal research intelligence system for an IST Master's thesis on structured perception
for robotic grocery bagging. Designed to demonstrate ML engineering, cloud architecture,
and full-stack development skills for DACH market entry (Munich / Zurich, 2026).

---

## v1 — Local Agent (Streamlit + Ollama + SQLite + ChromaDB)
> Branch: `main` | Target tag: `v1.0.0`

The focus of v1 is the agent core — retrieval, grounding, citation, and knowledge graph.
UI is functional but not the priority.

### ✅ Done
- [x] Project scaffold (uv, pydantic settings, stable ID system)
- [x] MCP servers — filesystem + experiments
- [x] Knowledge store — SQLite graph + ChromaDB vector store
- [x] PDF extractor + boundary-aware chunker
- [x] Ingest flow — PDF → chunks → graph nodes + vector embeddings
- [x] Retrieve flow — hybrid vector + graph keyword + graph expansion
- [x] Generate flow — grounded answer generation with citation enforcement
- [x] Orchestrator — intent routing + LangGraph-ready state
- [x] Bulk ingest — 24 papers ingested
- [x] Streamlit UI — chat + workspace (ingest, knowledge map, store)
- [x] Query rewriter — LLM rewrites natural language to dense keywords before retrieval
- [x] Citation resolution — chunk IDs resolved to source IDs
- [x] Integration test — `tests/integration/test_generate_flow.py`
- [x] README

### 🔲 Remaining (complete before tagging v1.0.0)
- [ ] Note extractor — ingest markdown notes, thesis decisions, experiment logs
- [ ] Evaluation harness — 30 golden Q&A pairs, measure hit@k + citation precision + latency
- [ ] arxiv MCP server — pull papers by ID or search directly into RCA
- [ ] Zotero MCP server — sync Zotero library into RCA automatically
- [ ] Weekly digest generator — scheduled summary of new papers + thesis connections

---

## v2 — Cloud-Native (FastAPI + Next.js + AWS)
> Branch: `v2/migrate` | Starts after v1.0.0 is tagged

The focus of v2 is production deployment and a professional web interface.
The `rca/` core package is shared — only the interface and infrastructure change.

### Planned
- [ ] FastAPI backend — replace Streamlit with a proper REST + WebSocket API
- [ ] Next.js frontend — replace Streamlit UI with a real web app
- [ ] S3 — replace local PDF storage with S3 bucket + trigger on upload
- [ ] RDS / DynamoDB — replace SQLite graph store with managed DB
- [ ] SageMaker — replace local Ollama with hosted embedding + generation endpoints
- [ ] Lambda — trigger ingest pipeline when new PDF lands in S3
- [ ] CloudWatch — monitor pipeline latency, error rates, retrieval scores
- [ ] Auth — simple JWT auth so the app can be shared / demoed publicly
- [ ] CI/CD — GitHub Actions → deploy to AWS on merge to main

### AWS certifications (study alongside v2, no exam required)
- [ ] AWS Solutions Architect – Associate (knowledge, not diploma)
- [ ] AWS Machine Learning Engineer – Associate (knowledge, not diploma)

---

## Branching strategy
```
main              ← stable, always working
v2/migrate        ← v2 active development
```

```bash
# Tag v1 before starting v2
git tag v1.0.0 -m "v1: Streamlit + local Ollama + SQLite + ChromaDB"
git push origin v1.0.0

# Start v2
git checkout -b v2/migrate
```