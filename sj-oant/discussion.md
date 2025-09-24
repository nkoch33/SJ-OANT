### Discussion

#### Executive summary
- TMMA leads on dialogue quality across MultiWOZ, SGD, and Taskmaster, with the most pronounced gains on structurally rich, multi-domain MultiWOZ (e.g., Diversity 7.258, Relevance 38.93%, Accuracy 78.17%, Understanding 94.50%).
- Under false-memory injection, TMMA achieves substantially lower FMR and higher DAR/CDR across all three datasets (e.g., MultiWOZ: FMR 6.75%, DAR 82.42%, CDR 41.20%), while Embedded RAG consistently outperforms Simple RAG but remains well behind TMMA.
- These outcomes follow directly from implementation choices: write-time quarantine in the memory store, retrieval-time exclusion of FLAGGED content, tiered promotion with conservative thresholds, and slot-aware context screening in the TACS filter.

---

#### Dialogue results: what the numbers say and why
- MultiWOZ 2.4 (Diversity/Rel/Acc/Understanding): TMMA 7.258 / 38.93% / 78.17% / 94.50% vs Embedded RAG 3.563 / 23.43% / 32.67% / 71.23%; Simple RAG is lowest (1.712 / 7.21% / 14.15% / 38.33%).
  - Why: Multi-domain constraints magnify the value of TMMA’s tiered memory and curation. In `agents/writer_editor.py`, the Memory Curation Agent promotes content L1→L2→L3 only when confidence/utility are sufficient; `core/types.py` and `memory/typed_store.py` enforce tier limits and confidence-aware scoring. This produces high-density, stable summaries (L2) and verified anchors (L3), which the TACS filter leverages for context assembly.
  - TACS contribution: In `truth/tacs_filter.py`, slot extraction and relevance gating prune noise before retrieval, which lifts Relevance and Semantic Similarity. This prevents off-topic drift and improves task-specific grounding (reflected in Understanding 94.50%).
- SGD (BLEU/Slot F1/SemSim/Intent Acc): TMMA 5.75 / 0.654 / 38.81% / 55.45%; Embedded RAG 4.38 / 0.490 / 17.43% / 31.43%; Simple RAG trails.
  - Why: Schema-grounded tasks reward consistent slot handling. TMMA’s curation compresses volatile detail into canonical summaries in L2, stabilizing slot mentions and boosting Slot F1. Embedded RAG improves via semantic recall but lacks truth hygiene, resulting in weaker Intent Accuracy and SemSim relative to TMMA.
- Taskmaster (BLEU/ROUGE/SemSim/Slot F1): TMMA 6.50 / 6.41 / 22.91% / 0.727; Embedded RAG 3.91 / 5.84 / 15.41% / 0.485; Simple RAG lags.
  - Why: Even in shorter, less-structured dialogues, TMMA’s promotion rules and relevance screening reduce contradiction carryover and maintain better slot grounding. Embedded RAG narrows the gap on textual overlap (ROUGE) but still lags in Slot F1 due to absent write-time filtering and weaker contradiction control.

Key mechanism → metric mapping (dialogue)
- Selective Addition + tier promotion (L1→L2→L3): higher Relevance/Accuracy/Understanding by stabilizing reusable memory and pruning noise.
- Confidence-weighted curation: fewer contradictory carryovers; better Intent Accuracy and Slot F1 on SGD.
- Slot-aware context (TACS): improved Relevance and Semantic Similarity via entity- and slot-conditioned retrieval.

---

#### False-memory results: why TMMA’s FMR/DAR/CDR are superior
- MultiWOZ (FMR/MEL/DAR/CDR): TMMA 6.75% / 0.04 / 82.42% / 41.20% vs Embedded RAG 52.30% / 0.04 / 29.27% / 13.45%; Simple RAG 81.50% / 0.01 / 6.83% / 1.55%.
  - Why low FMR and high DAR/CDR for TMMA:
    - Write-time quarantine: In `memory/typed_store.py`, `FalseMemoryDetectionSystem.detect_false_memory(...)` rewrites detected items into the FLAGGED tier with low confidence scores; this prevents false content from entering L1/L2/L3.
    - Retrieval-time exclusion: In `truth/tacs_filter.py`, `MemoryRetriever.retrieve_relevant_memory` filters out `MemoryTier.FLAGGED`, breaking the rehearsal loop that otherwise amplifies falsehoods.
    - Combined effect: Responses avoid repeating injected falsehoods (low FMR), include more verified facts (higher DAR), and explicitly acknowledge conflicts more often (higher CDR).
- SGD: TMMA 10.50% / 0.01 / 73.28% / 34.15% vs Embedded RAG 55.70% / 0.05 / 42.55% / 18.75% vs Simple RAG 84.30% / 0.03 / 14.92% / 2.35%.
  - Why still strong but relatively narrower gap than MultiWOZ: shorter dialogues provide fewer opportunities for contradiction compounding; however, TMMA retains a large advantage due to persistent quarantine and tiered promotion.
- Taskmaster: TMMA 7.60% / 0.01 / 77.45% / 38.95% vs Embedded RAG 46.15% / 0.03 / 48.41% / 21.55% vs Simple RAG 78.25% / 0.01 / 9.92% / 1.95%.
  - Why TMMA dominates: even with simpler tasks, the two-stage guardrail (write-time + retrieval-time) suppresses false rehearsal. Embedded RAG’s improved semantic recall raises DAR relative to Simple RAG, but without quarantine/exclusion, its FMR remains high.

Interpreting MEL across models
- MEL is uniformly small (≈ 0.01–0.05) because our operationalization is turn-based (1 turn ≈ 1s) and tracks latency to any corrective signal, not the durability of correction. Baselines often acknowledge issues quickly (low MEL) while still repeating falsehoods (high FMR). This validates using FMR, DAR, and CDR in concert to quantify persistence and robustness.

Qualitative failure modes in baselines
- Semantic echo: embeddings retrieve paraphrases of the injected falsehood; without write-time filtering, these get stored and later re-surfaced (↑FMR, ↓CDR).
- Context inertia: once a false fact enters working context, subsequent turns reinforce it, especially without demotion/quarantine logic (↓DAR).
- Conflict under-resolution: models flag inconsistency (low MEL) but do not demote or exclude contaminated items; falsehoods persist (↑FMR).

Why Embedded RAG > Simple RAG, yet < TMMA
- Embedded RAG’s dense similarity improves retrieval quality for dialogue (e.g., MultiWOZ Relevance 23.43% vs Simple RAG 7.21%; Taskmaster ROUGE 5.84 vs 2.65). However, absent write-time quarantine and retrieval-time exclusion, it remains vulnerable to storing and retrieving injected falsehoods (e.g., MultiWOZ FMR 52.30% vs TMMA 6.75%).

Dataset-wise interpretation
- MultiWOZ shows the biggest gains for TMMA because multi-domain context magnifies the benefits of L2/L3 consolidation and contradiction screening.
- Gains persist on SGD and Taskmaster but narrow with shorter or less-structured dialogues; long-horizon curation has less runway, yet the two-stage guardrail still delivers large FMR/CDR advantages.

Limitations (by design of this study)
- Means only; no statistical testing. Results reflect practical performance under our fixed configuration.
- MEL is turn-based; future work can use wall-clock latencies.
- Detection relies on curated false patterns/tuples; fully open-world verification is outside current scope.

Recommended visuals and artifacts
- Grouped bar charts: per-benchmark FMR/DAR/CDR for all models (clarifies robustness gaps).
- Radar plots: dialogue metrics per model (TMMA’s balanced profile vs RAG baselines’ narrower gains).
- Guardrail pipeline diagram: write-time detection/quarantine and retrieval-time FLAGGED exclusion—visual rationale for low FMR without hurting dialogue quality.
- Turn-level traces: (a) baseline drift/rehearsal, (b) TMMA detection → FLAGGED → exclusion → correction; annotate where each module acts.

Primary takeaway
- Truth maintenance must be enforced at memory formation and at retrieval. TMMA’s write-time quarantine and retrieval-time exclusion, coupled with tiered promotion and slot-aware filtering, systematically reduce false-memory formation (FMR) and improve robustness (DAR/CDR) while preserving or improving dialogue quality. Semantic retrieval alone is insufficient; proactive, tier-aware memory hygiene is the differentiator in long-context settings.
