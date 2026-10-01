# RetailRAG PHASE 6 LLM Runtime Audit

- Configured default: `qwen3-max`
- Default status: `ARREARAGE`
- Final unified LLM regression: `BLOCKED_BY_API_QUOTA`
- Mixed-model metrics: `NO`
- Credential value logged: `NO`

The runtime call sites remain centralized through `Config.LLM_MODEL`; see `RETAILRAG_PHASE6_LLM_AUDIT.json` for the file/function inventory.

| Model | Minimal smoke | Module smoke | Large evaluation |
|---|---|---|---|
| qwen3-max | SDK blocked | Not rerun | Arrearage |
| qwen3.6-plus | PASS | PASS | Generation 80/80; Full RAG 133/350 quota boundary |
| qwen3.6-flash | PASS | PASS | Cancelled after 20 Generation rows |

Historical `qwen3-max`, plus, and flash artifacts are kept separately and are not merged.
