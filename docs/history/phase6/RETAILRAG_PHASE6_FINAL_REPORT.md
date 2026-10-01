# RetailRAG PHASE 6 Final Targeted Fix Report

## Executive decision

RetailRAG remains **FROZEN_WITH_KNOWN_LIMITATIONS**. The frozen FAQ, knowledge documents and benchmark data were not changed. FAQ Complexity Gate, Dynamic M, Evidence Gate, Query Rewrite and non-LLM engineering regression remain complete. The final single-model LLM regression is **BLOCKED_BY_API_QUOTA**, not failed.

## Frozen non-LLM PHASE 6 results

| Item | Before | PHASE 6 |
|---|---:|---:|
| FAQ F1 | 0.844444 | 0.863636 |
| FAQ false accept rate | 0.038889 | 0.000000 |
| Complex FAQ wrong intercepts | 12 | 0 |
| Same-K3 All@3 | 0.761702 | 0.778723 |
| Context recall | 0.828369 | 0.837943 |
| Evidence holdout refusal recall | — | 1.000000 |

All four frozen benchmark/FAQ hashes continue to match the pre-PHASE6 snapshot. The 80 knowledge documents also remain byte-identical to the pre-PHASE6 backup.

## LLM FINAL REGRESSION LIMITATION

- The configured historical model `qwen3-max` returns `Arrearage` for real SDK calls.
- `qwen3.6-plus` passed the minimal call and all module smoke tests.
- `qwen3.6-plus` completed the fixed Generation Evaluation: **80/80**, with **0 execution errors**.
- During the complete Full RAG regression, `qwen3.6-plus` reached the **133/350 quota boundary** and then returned HTTP 403 `AllocationQuota.FreeTierOnly`.
- The raw cancellation checkpoint contains 135 successful rows because two already in-flight or FAQ-direct tasks completed while cancellation propagated. The experiment remains incomplete and is not reported as 350/350.
- `qwen3.6-flash` passed minimal and module smoke tests. Its large run was actively cancelled after 20 Generation rows to avoid exhausting another limited free quota.
- Samples from different models are not combined. No incomplete Full RAG result is presented as a final complete regression.

## qwen3.6-plus Generation experiment

This experiment is retained as `QWEN36_PLUS_GENERATION_EXPERIMENT`. Generator and Judge both used `qwen3.6-plus`, so same-family scoring may be optimistic. It is not a controlled comparison with historical `qwen3-max` metrics.

| Metric | Result |
|---|---:|
| Execution | 80/80 |
| Execution errors | 0 |
| Faithfulness | 0.995000 |
| Answer relevance | 0.992500 |
| Context precision | 0.515875 |
| Context recall | 0.774375 |
| Unanswerable refusal accuracy | 1.000000 |
| Partial-answer accuracy | 1.000000 |

## Engineering and online status

MySQL, Redis, Milvus, MinIO, etcd and FastAPI health checks passed in the completed PHASE 6 engineering regression. FAQ Direct, complex-query-to-RAG routing, evidence refusal, WebSocket transport and browser rendering were verified. The post-migration full online quality regression was not restarted after quota cancellation; its status remains `PASS_WITH_LLM_BLOCKER` rather than a complete model acceptance.

## Historical comparability

Historical PHASE 5 and Expanded Scale generation metrics used `qwen3-max`. The retained Generation 80 experiment used `qwen3.6-plus`. Retrieval-side PHASE 6 metrics are unchanged, but generation scores across these model versions are not treated as strictly controlled comparisons.

## Restart condition

Do not restart Generation 80 or Full RAG 350 until one condition is met:

1. `qwen3-max` account access is restored;
2. one model has confirmed quota sufficient for the entire unified run; or
3. a local model is deployed without API-token quota limits.

No production-readiness claim or final Git tag is recommended before that unified run completes.

## Final status

```text
PROJECT_STATE: FROZEN_WITH_KNOWN_LIMITATIONS
QWEN3_MAX: BLOCKED_ARREARAGE
QWEN3_6_PLUS_GENERATION: 80/80 COMPLETE
QWEN3_6_PLUS_FULL_RAG: 133/350 INCOMPLETE_QUOTA
QWEN3_6_FLASH: SMOKE_PASS_BUT_FULL_EVAL_CANCELLED
FINAL_LLM_REGRESSION: BLOCKED_BY_API_QUOTA
MIXED_MODEL_RESULTS: NO
CODE_CHANGE_REQUIRED: NO
```
