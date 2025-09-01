# SJ-OANT: Truth-Maintained Memory for LLM Agents

## Methodology & Benchmark Plan

### 1. Final Model & Agentic Pipeline

#### 1.1 Truth-Maintained Memory (TMM) Agent

A multi-agent, truth-maintained memory system that proactively filters, verifies, and curates information before it reaches long-term memory. The agent runs a five-stage loop each turn:

1. **Retrieve & Collect**: User turn + candidate memory snippets + optional external docs
2. **Context Filtering**: Token-/span-level gating to down-weight/remove misleading or off-task content
3. **Verification & Scoring**: A lightweight verifier assigns truth/confidence + evidentiality
4. **Selective Memory Ops**: 
   - Selective Addition (write only high-trust, high-utility facts) 
   - Combined Deletion (periodic pruning of contradicted/low-utility entries)
5. **Compose & Respond**: Planner/arbiter assembles a final context; responder LLM answers; edits/flags are persisted

This matches the TJFVW/EStore loop (Retriever → Truth Filter → Verifier → Writer/Editor → Memory Store).

#### 1.2 Multi-tiered Memory + Voting (Internal Structure)

**Memory Tiers:**
- **L1 (Working)**: Recent raw turns; scratch-space for analysis
- **L2 (Summarized)**: Concise, referenceable summaries and entities
- **L3 (Archival/KB)**: Verified facts and stable entities; authoritative source of truth
- **Flagged Bin**: Quarantined, contradicted, or low-confidence items (kept for later review)

**Voting-based Decisions**: Specialized micro-agents (redundancy, contradiction detection, summarization/extraction, etc.) cross-check each other and vote; a memory curator adjudicates placement in L1/L2/L3 versus Flagged bin. This keeps the pipeline accountable and prevents single-agent drift.

#### 1.3 Core Capabilities

The model is built to provide:
- Accurate retrieval & search across typed memory
- Contradiction detection & resolution with proper overwrites
- Noise resistance via filtering + conservative writes
- Long-term retention of verified facts (stable over many turns)
- Multi-step reasoning over memory (compose multiple memory elements without spurious links)

These five capabilities dovetail with the evaluation framework.

### 2. FABLE Benchmark (False-memory-Aware Benchmark for Long-term Engagements)

FABLE is distinct from Letta and purpose-built to measure false-memory resilience in long, multi-turn settings. It combines standard memory competencies with truth-maintenance stress tests.

#### 2.1 Competency Axes

**Standard Memory Competencies:**
- **AR** — Accurate Retrieval
- **TTL** — Test-Time Learning (remember newly learned facts)
- **LRU** — Long-Range Understanding (distributed evidence over long threads)
- **CR** — Conflict Resolution (update/overwrite old info when corrected)

**Truth-Maintenance Axes (Novel):**
- **FMP** — False-Memory Propagation: seed an early lie; measure downstream repetition, confidence, and persistence
- **MCR** — Misleading-Context Resistance: inject confounding snippets; measure leakage into generation
- **CUT** — Contradiction/Update Tracking: later turns correct earlier claims; require timely, correct overwrites
- **EFS** — Experience-Following Safety: ensure the agent doesn't blindly copy harmful prior exemplars (prevents error replay)

#### 2.2 Scenario Design & Data Synthesis

**Dialogue Structure:**
- Scripted long dialogues (30–120 turns) with:
  - Time-stamped facts
  - Planted contradictions
  - Distractors/noise
  - Corrections
  - Multi-hop queries

**Ground-truth Mini-KB**: Per scenario to enable verification without open-web noise (cleaner science)

**Difficulty Tiers:**
- **Easy**: Single correction, low distractors
- **Medium**: Multiple corrections, intertwined entities
- **Hard**: Overlapping entities, delayed corrections, adversarial distractors
- **Optional "Forensics"**: Mini-set inspired by human false-memory literature to probe suggestibility dynamics

**Generation**: Templates + small human pass for seed sets ensure quality while keeping costs tractable.

#### 2.3 Metrics

**Task Metrics:**
- Accuracy/F1 per task family (AR/TTL/LRU/CR)

**False-Memory Metrics:**
- **FMR** — False Memory Rate: % answers repeating seeded falsehoods
- **MEL** — Memory Edit Latency: turns to correctly overwrite after a correction appears
- **DAR** — Disturbance Adaptation Rate: reliability under mixed true/false context perturbations

**System Quality:**
- Verifier quality: PR/ROC vs. truth labels; calibration error
- Systems costs: memory footprint (MB), retrieval latency (ms), tokens processed (efficiency)

Canonical scorers implemented under `benchmarks/fable/`.

### 3. Baselines, Ablations, & Backbones

#### 3.1 Baselines (Comparative)

- Long-context only (no external memory)
- Simple/Embedding/Structured RAG (BM25, dense, graph/timeline-augmented)
- Agentic memory w/out truth gates (reflection/looping, but store-first-filter-later)
- MIRIX-style typed memory (if feasible to approximate)

These baselines provide clear contrast to proactive filtering.

#### 3.2 Ablations (What Makes TMM Work)

- Remove Truth Filter (no token gating)
- Remove Selective Addition or Combined Deletion
- Single-agent (no micro-managers, no voting)
- Verifier off / threshold sweeps

This set isolates which components drive FMR/MEL gains.

#### 3.3 Model Backbones (Open Source)

Evaluate across small/medium backbones (e.g., ~7–14B) plus a larger tier (~30B+), all served via vLLM; standardize decoding & prompting for comparability. (Exact families are flexible; this plan is backbone-agnostic.)

### 4. Feasibility, Compute, and Logistics

#### 4.1 Compute Options

**Infrastructure:**
- Local/Azure/GCP GPUs: 2–4× A100 80GB (or L40S/A10 class) are sufficient to serve 2–3 backbones via vLLM and run the bench in parallel
- Databricks: feasible for orchestrating distributed evaluation and storing artifacts (scores, traces). Use DBFS for scenario shards + MLflow for metrics

**Efficiency Optimizations:**
- Enable paged attention + kv-cache reuse in vLLM
- Use 4-bit/8-bit AWQ weights for larger models to fit memory
- Shard FABLE runs across workers

#### 4.2 Data/Code Layout

```
repo/
├── agents/
│   ├── planner.py
│   ├── arbiter.py
│   ├── responder.py
│   └── writer_editor.py
├── truth/
│   ├── tacs_filter.py
│   └── verifier.py
├── memory/
│   ├── typed_store.py
│   ├── policies.py
│   └── voting.py
├── retrieval/
│   ├── hybrid_retriever.py
│   └── active_retrieval.py
├── benchmarks/fable/
│   ├── scenarios/*.jsonl
│   ├── kb/*.json
│   └── scorers/
│       ├── fmr.py
│       ├── mel.py
│       ├── dar.py
│       └── acc.py
├── runners/
│   ├── eval_fable.py
│   ├── eval_baselines.py
│   └── ablations.py
├── infra/
│   ├── vllm_server.py
│   ├── hf_loader.py
│   └── config.yaml
└── notebooks/
    └── analysis/*.ipynb
```

#### 4.3 Risk & Mitigation

**Verifier Data**: Bootstrap with synthetic labels + weak supervision; begin with rule-augmented features (exact-match to KB, contradiction patterns) to avoid cold-start.

**Scenario Quality**: Start with 100–200 seed dialogues; human pass; then grow via templates.

**Throughput**: Batch evaluation; cap scenario lengths by tier; run ablations on a subset for speed.

**Leakage**: No open-web retrieval in the main bench; only the provided mini-KB per scenario.

### 5. Evaluation Protocol

#### 5.1 Sanity Runs
Toy scenarios to ensure metrics move in the right direction (FMR↓, MEL↓ with our pipeline vs. baselines).

#### 5.2 Full E-suite
Each with 3 runs, fixed seeds:

- **E1**: Baselines vs. TMM on AR/TTL/LRU/CR + FMP/MCR/CUT/EFS
- **E2**: Error-propagation stress (seed bad demos; vary write/delete; measure FMR & MEL)
- **E3**: Budget & shift (tight memory cap; distribution shift; report stability & storage/latency)
- **E4**: Multi-agent gains (typed managers + voting vs. monolithic)
- **E5**: Truth Filter efficacy (token gating on adversarial contexts)
- **E6**: Verifier study (thresholds, PR/ROC, calibration vs. task accuracy)

#### 5.3 Reporting
- Mean ± stdev
- Paired t-tests (or bootstrap) vs. strongest baseline
- Include efficiency tables (latency, memory footprint, tokens processed) to show practicality

### 6. Novelty & Publication Value

**Core Architectural Contribution:**
- Proactive, multi-agent filtering at write-time (not just read-time)
- Voting + typed memory tiers (L1/L2/L3/Flagged) to prevent error ingress

**FABLE Benchmark:**
- Distinct benchmark from Letta that explicitly measures false-memory propagation and update/contradiction handling over long horizons

**Principled Framework:**
- Five capabilities tied to concrete, automatable metrics (FMR, MEL, DAR + accuracy)

### 7. Execution Timeline

**Week 0–1**: Finalize FABLE v0.1 (100–200 scenarios across tiers); stand up vLLM; implement truth/ + memory/ modules (Selective Addition, Combined Deletion, Flagged bin, voting).

**Week 2**: Baselines (long-context, RAGs, minimal agentic loop); E1/E2 pilot; calibrate verifier thresholds.

**Week 3**: Full E-suite on 2 backbones; ablations; start writing Results.

**Week 4**: Extend to 3rd backbone; finalize plots/tables; polish Methods/Benchmark; Limitations & Broader Impact.

### 8. Success Criteria

**Performance Targets:**
- ≥25–40% reduction in FMR vs. strongest baseline on medium tier
- MEL cut by ≥30% on CUT tasks
- Equal or better accuracy on legitimate memory tasks (AR/TTL/LRU/CR)

**Efficiency Requirements:**
- Efficiency within ≤1.3× latency and ≤1.2× tokens vs. strongest baseline at similar context length (practical deployability)

---

## 🔄 Operational Workflow: How Our Model Works

Think of the system as a memory curator for an LLM agent. Each time the agent has a new interaction, the following cycle runs:

### Step 1. Collect Information (Listening Stage)
- The agent listens to the new user input
- At the same time, it pulls up potentially relevant past memories and supporting documents
- This is like a person gathering notes before answering a question

### Step 2. Filter Out the Noise (Screening Stage)
- A filtering agent checks the information for relevance
- Distracting or unrelated details are stripped away
- Example: if the user asks about travel plans, a stray memory about food recipes won't get through

### Step 3. Check for Truth (Fact-Checking Stage)
- A verifier agent inspects the information and cross-checks it against what is already known
- It looks for contradictions (e.g., "The event was in June" vs. "The event was in July") and flags low-confidence facts
- Only reliable information gets a "green light"

### Step 4. Manage Memory (Curator Stage)
The system decides what to do with the verified information:
- **Store**: If it's new and trustworthy, it goes into long-term memory
- **Update**: If it corrects an old fact, the old entry is updated
- **Discard**: If it's misleading or too noisy, it's dropped

This prevents the agent from accidentally locking in false memories.

### Step 5. Respond & Learn (Speaking Stage)
- The agent composes a response using only the cleaned and verified memory
- At the same time, the memory store gets updated so future interactions stay consistent
- This is like answering a question and taking clean notes for later

## 📦 Internal Components

- **Retriever**: pulls candidate memories & external docs
- **Filter Agent**: removes irrelevant content
- **Verifier Agent**: checks truth, contradictions, and reliability
- **Memory Curator**: decides what to keep, update, or discard
- **Responder**: generates the final answer to the user

---

## What is the Dataset?

**FABLE (False-memory-Aware Benchmark for Long-term Engagements)** is a synthetic + semi-synthetic corpus of multi-turn dialogues (≈30–120 turns each) designed to stress long-horizon memory with truth-maintenance pressure.

### Each Scenario Contains:

- **A gold mini-KB**: The ground-truth facts and timelines for that scenario
- **A dialogue script with**: 
  - Planted misinformation
  - Later corrections/updates
  - Benign distractors
  - Multi-hop questions
- **Annotations per turn**: 
  - Which facts were referenced
  - Which were contradicted/updated
  - Which spans are distractors
  - Which answer is correct given the latest truth state

### Dataset Characteristics:

**Domains (balanced)**: Personal assistant tasks, customer support, project planning, and a small "forensics" mini-set (to probe suggestibility/false-memory pressures).

**Splits**: train/dev/test (e.g., 50/25/25 by scenario), with difficulty tiers (easy/med/hard).

**Format**: JSONL per conversation (turn-level metadata + pointers into the mini-KB).

**Scale (v1 targets)**: 100–200 scenarios (5–10k turns) to start; expandable.

---

## What is the Benchmark?

FABLE is both the dataset and the evaluation suite. It measures standard memory competencies and false-memory resilience.

### Capability Axes

**Standard Memory Competencies:**
- **AR** — Accurate Retrieval
- **TTL** — Test-Time Learning (remember new facts)
- **LRU** — Long-Range Understanding (distributed evidence)
- **CR** — Conflict Resolution (apply corrections/overwrites)

**Truth-maintenance Axes (Our Novelty):**
- **FMP** — False-Memory Propagation: do seeded lies reappear later?
- **MCR** — Misleading-Context Resistance: do distractors leak into answers?
- **CUT** — Contradiction/Update Tracking: how quickly are outdated facts replaced?
- **EFS** — Experience-Following Safety: does the agent replay bad past examples?

### Metrics (Automatically Computed)

**Task Performance:**
- Accuracy/F1 on AR/TTL/LRU/CR tasks

**False-Memory Metrics:**
- **FMR** (False Memory Rate): % answers that repeat seeded misinformation
- **MEL** (Memory-Edit Latency): turns to correct after a contradiction/update appears
- **DAR** (Disturbance Adaptation Rate): reliability under mixed true/false context

**System Costs:**
- Latency, memory footprint, tokens processed

### Baselines & Protocol

**Baselines**: Long-context (no memory), simple/embedding/structured RAG, and agentic memory without truth-gating; optionally a typed-memory baseline.

**Your System**: Truth-maintained, multi-agent pipeline evaluated head-to-head on all axes, with ablations (no filter / no verifier / no selective-write / no deletion / single-agent).

### Why This is Distinct

Unlike general leaderboards, FABLE bakes in ground-truth timelines + controlled misinformation/corrections, so we can precisely measure false-memory formation and repair—not just raw QA accuracy.
