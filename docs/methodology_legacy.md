Model Capabilities +
Benchmarks
Methodology & Benchmark Plan
1) Final model & agentic pipeline (what we are actually
building)
1.1 Truth-Maintained Memory (TMM) agent
A multi-agent, truth-maintained memory system that proactively filters, verifies,
and curates information before it reaches long-term memory. The agent runs a
five-stage loop each turn:
1. Retrieve & collect: user turn + candidate memory snippets + optional external
docs
2. Context filtering: token-/span-level gating to down-weight/remove misleading
or off-task content
3. Verification & scoring: a lightweight verifier assigns truth/confidence +
evidentiality
4. Selective memory ops: Selective Addition (write only high-trust, high-utility
facts) + Combined Deletion(periodic pruning of contradicted/low-utility
entries)
5. Compose & respond: planner/arbiter assembles a final context; responder
LLM answers; edits/flags are persisted
This matches the TJFVW/EStore loop you outlined earlier (Retriever →
Truth Filter → Verifier → Writer/Editor → Memory Store).
1.2 Multi-tiered memory + voting (internal structure)
Model Capabilities + Benchmarks 1
L1 (working): recent raw turns; scratch-space for analysis
L2 (summarized): concise, referenceable summaries and entities
L3 (archival/KB): verified facts and stable entities; authoritative source of
truth
Flagged bin: quarantined, contradicted, or low-confidence items (kept for later
review)
Voting-based decisions: specialized micro-agents (redundancy, contradiction
detection, summarization/extraction, etc.) cross-check each other and vote; a
memory curator adjudicates placement in L1/L2/L3 versus Flagged bin. This
keeps the pipeline accountable and prevents single-agent drift.
1.3 Capabilities (what the model is built to do)
Accurate retrieval & search across typed memory
Contradiction detection & resolution with proper overwrites
Noise resistance via filtering + conservative writes
Long-term retention of verified facts (stable over many turns)
Multi-step reasoning over memory (compose multiple memory elements
without spurious links)
These five capabilities dovetail with the evaluation framework you drafted in
the intro and abstract.
2) Our benchmark: F ABLE (False-memory-Aware
Benchmark for Long-term Engagements)
FABLE is distinct from Letta and purpose-built to measure false-memory
resilience in long, multi-turn settings. It combines standard memory competencies
with truth-maintenance stress tests.
2.1 Competency axes
AR — Accurate Retrieval
TTL — Test-Time Learning (remember newly learned facts)
Model Capabilities + Benchmarks 2
LRU — Long-Range Understanding (distributed evidence over long threads)
CR — Conflict Resolution (update/overwrite old info when corrected)
Plus our truth-maintenance axes:
FMP — False-Memory Propagation: seed an early lie; measure downstream
repetition, confidence, and persistence
MCR — Misleading-Context Resistance: inject confounding snippets;
measure leakage into generation
CUT — Contradiction/Update Tracking: later turns correct earlier claims;
require timely, correct overwrites
EFS — Experience-Following Safety: ensure the agent doesn’t blindly copy
harmful prior exemplars (prevents error replay)
These axes and their metrics were already sketched in your methodology doc;
we’re freezing them here as the final spec.
2.2 Scenario design & data synthesis
Scripted long dialogues (30–120 turns) with: time-stamped facts, planted
contradictions, distractors/noise, corrections, and multi-hop queries
Ground-truth mini-KB per scenario to enable verification without open-web
noise (cleaner science)
Difficulty tiers:
Easy: single correction, low distractors
Medium: multiple corrections, intertwined entities
Hard: overlapping entities, delayed corrections, adversarial distractors
Optional “forensics” mini-set inspired by human false-memory literature to
probe suggestibility dynamics
Generation templates + small human pass for seed sets ensure quality while
keeping costs tractable.
2.3 Metrics
Task metrics: accuracy/F1 per task family (AR/TTL/LRU/CR)
Model Capabilities + Benchmarks 3
False-memory metrics:
FMR — False Memory Rate: % answers repeating seeded falsehoods
MEL — Memory Edit Latency: turns to correctly overwrite after a
correction appears
DAR — Disturbance Adaptation Rate: reliability under mixed true/false
context perturbations
Verifier quality: PR/ROC vs. truth labels; calibration error
Systems costs: memory footprint (MB), retrieval latency (ms), tokens
processed (efficiency)
As drafted in your methodology doc; we’ll implement canonical scorers
under benchmarks/fable/.
3) Baselines, ablations, & backbones
3.1 Baselines (comparative)
Long-context only (no external memory)
Simple/Embedding/Structured RAG (BM25, dense, graph/timeline-
augmented)
Agentic memory w/out truth gates (reflection/looping, but store-first-filter-
later)
MIRIX-style typed memory (if feasible to approximate)
These baselines mirror what your doc proposes and give clear contrast to
proactive filtering.
3.2 Ablations (what makes TMM work)
Remove Truth Filter (no token gating)
Remove Selective Addition or Combined Deletion
Single-agent (no micro-managers, no voting)
Verifier off / threshold sweeps
Model Capabilities + Benchmarks 4
This set isolates which components drive FMR/MEL gains.
3.3 Model backbones (open source)
Evaluate across small/medium backbones (e.g., ~7–14B) plus a larger
tier (~30B+), all served via vLLM; standardize decoding & prompting for
comparability. (Exact families are flexible; this plan is backbone-agnostic.)
4) Feasibility, compute, and logistics
4.1 Compute options
Local/Azure/GCP GPUs: 2–4× A100 80GB (or L40S/A10 class) are sufficient to
serve 2–3 backbones via vLLM and run the bench in parallel.
Databricks: feasible for orchestrating distributed evaluation and storing
artifacts (scores, traces). Use DBFS for scenario shards + MLflow for metrics.
Efficiency: enable paged attention + kv-cache reuse in vLLM; use 4-bit/8-bit
AWQ weights for larger models to fit memory; shard FABLE runs across
workers.
4.2 Data/Code layout (ready to implement)
repo/
agents/
planner.py | arbiter.py | responder.py | writer_editor.py
truth/
tacs_filter.py | verifier.py
memory/
typed_store.py | policies.py | voting.py
retrieval/
hybrid_retriever.py | active_retrieval.py
benchmarks/fable/
scenarios/* .jsonl | kb/* .json | scorers/{fmr.py, mel.py, dar.py, acc.py}
runners/
eval_fable.py | eval_baselines.py | ablations.py
infra/
Model Capabilities + Benchmarks 5
vllm_server.py | hf_loader.py | config.yaml
notebooks/
analysis/* .ipynb
This mirrors the structure in your methodology file; we’re just making it more
concrete.
4.3 Risk & mitigation
Verifier data: bootstrap with synthetic labels + weak supervision; begin with
rule-augmented features (exact-match to KB, contradiction patterns) to avoid
cold-start.
Scenario quality: start with 100–200 seed dialogues; human pass; then grow
via templates.
Throughput: batch evaluation; cap scenario lengths by tier; run ablations on a
subset for speed.
Leakage: no open-web retrieval in the main bench; only the provided mini-KB
per scenario.
5) Evaluation protocol (what we will measure, exactly)
1. Sanity runs on toy scenarios to ensure metrics move in the right direction
(FMR↓, MEL↓ with our pipeline vs. baselines).
2. Full E-suite (from your doc), each with 3 runs, fixed seeds:
E1: Baselines vs. TMM on AR/TTL/LRU/CR + FMP/MCR/CUT/EFS
E2: Error-propagation stress (seed bad demos; vary write/delete; measure
FMR & MEL)
E3: Budget & shift (tight memory cap; distribution shift; report stability &
storage/latency)
E4: Multi-agent gains (typed managers + voting vs. monolithic)
E5: Truth Filter efficacy (token gating on adversarial contexts)
Model Capabilities + Benchmarks 6
E6: Verifier study (thresholds, PR/ROC, calibration vs. task accuracy)
All six experiments are already specified in your methodology doc; this
locks them in.
3. Reporting: mean±stdev; paired t-tests (or bootstrap) vs. strongest baseline.
Include efficiency tables (latency, memory footprint, tokens processed) to
show practicality.
6) Novelty & why this is publishable
Proactive, multi-agent filtering at write-time (not just read-time),
with voting + typed memory tiers(L1/L2/L3/Flagged) to prevent error ingress
—this is the core architectural contribution.
FABLE: a distinct benchmark from Letta that explicitly measures false-
memory propagation and update/contradiction handling over long horizons.
Principled capability framework (your five capabilities) tied to concrete,
automatable metrics (FMR, MEL, DAR + accuracy).
7) Execution timeline (aggressive but feasible)
Week 0–1: finalize FABLE v0.1 (100–200 scenarios across tiers); stand up
vLLM; implement truth/ + memory/ modules (Selective Addition, Combined
Deletion, Flagged bin, voting).
Week 2: baselines (long-context, RAGs, minimal agentic loop); E1/E2 pilot;
calibrate verifier thresholds.
Week 3: full E-suite on 2 backbones; ablations; start writing Results.
Week 4: extend to 3rd backbone; finalize plots/tables; polish
Methods/Benchmark; Limitations & Broader Impact.
8) Success criteria
≥25–40% reduction in FMR vs. strongest baseline on medium tier; MEL cut by
≥30% on CUT tasks; equal or better accuracy on legitimate memory tasks
(AR/TTL/LRU/CR).
Model Capabilities + Benchmarks 7
Efficiency within ≤1.3× latency and ≤1.2× tokens vs. strongest baseline at
similar context length (practical deployability).
🔄 Operational Workflow: How Our Model
Works
Think of the system as a memory curator for an LLM agent. Each time the agent
has a new interaction, the following cycle runs:
Step 1. Collect Information (Listening Stage)
The agent listens to the new user input.
At the same time, it pulls up potentially relevant past memories and supporting
documents.
This is like a person gathering notes before answering a question.
Step 2. Filter Out the Noise (Screening Stage)
A filtering agent checks the information for relevance.
Distracting or unrelated details are stripped away.
Example: if the user asks about travel plans, a stray memory about food
recipes won’t get through.
Step 3. Check for Truth (Fact-Checking Stage)
A verifier agent inspects the information and cross-checks it against what is
already known.
It looks for contradictions (e.g., “The event was in June” vs. “The event was in
July”) and flags low-confidence facts.
Only reliable information gets a “green light.”
Model Capabilities + Benchmarks 8
Step 4. Manage Memory (Curator Stage)
The system decides what to do with the verified information:
Store: If it’s new and trustworthy, it goes into long-term memory.
Update: If it corrects an old fact, the old entry is updated.
Discard: If it’s misleading or too noisy, it’s dropped.
This prevents the agent from accidentally locking in false memories.
Step 5. Respond & Learn (Speaking Stage)
The agent composes a response using only the cleaned and verified memory.
At the same time, the memory store gets updated so future interactions stay
consistent.
This is like answering a question and taking clean notes for later.
📦 Internal Components
Retriever: pulls candidate memories & external docs.
Filter Agent: removes irrelevant content.
Verifier Agent: checks truth, contradictions, and reliability.
Memory Curator: decides what to keep, update, or discard.
Responder: generates the final answer to the user.
What is the dataset?
FABLE (False-memory-Aware Benchmark for Long-term Engagements) is
a synthetic + semi-synthetic corpus of multi-turn dialogues (≈30–120 turns each)
designed to stress long-horizon memory with truth-maintenance pressure. This is
just the proposed name of the benchmark.
Each scenario contains:
A gold mini-KB (the ground-truth facts and timelines for that scenario).
Model Capabilities + Benchmarks 9
A dialogue script with: planted misinformation, later corrections/updates,
benign distractors, and multi-hop questions.
Annotations per turn: which facts were referenced, which were
contradicted/updated, which spans are distractors, and which answer is
correct given the latest truth state.
Domains (balanced): personal assistant tasks, customer support, project
planning, and a small “forensics” mini-set (to probe suggestibility/false-memory
pressures).
Splits: train/dev/test (e.g., 50/25/25 by scenario), with difficulty tiers
(easy/med/hard).
Format: JSONL per conversation (turn-level metadata + pointers into the mini-
KB).
Scale (v1 targets): 100–200 scenarios (5–10k turns) to start; expandable.
What is the benchmark?
FABLE is both the dataset and the evaluation suite. It measures standard memory
competencies and false-memory resilience.
Capability axes
AR Accurate Retrieval
TTL Test-Time Learning (remember new facts)
LRU Long-Range Understanding (distributed evidence)
CR Conflict Resolution (apply corrections/overwrites)
Truth-maintenance axes (our novelty)
FMP False-Memory Propagation — do seeded lies reappear later?
MCR Misleading-Context Resistance — do distractors leak into answers?
CUT Contradiction/Update Tracking — how quickly are outdated facts
replaced?
EFS Experience-Following Safety — does the agent replay bad past
examples?
Model Capabilities + Benchmarks 10
Metrics (automatically computed)
Accuracy/F1 on AR/TTL/LRU/CR tasks
FMR (False Memory Rate): % answers that repeat seeded misinformation
MEL (Memory-Edit Latency): turns to correct after a contradiction/update
appears
DAR (Disturbance Adaptation Rate): reliability under mixed true/false context
Cost: latency, memory footprint, tokens processed
Baselines & protocol
Baselines: long-context (no memory), simple/embedding/structured RAG, and
agentic memory without truth-gating; optionally a typed-memory baseline.
Your system (truth-maintained, multi-agent pipeline) is evaluated head-to-
head on all axes, with ablations (no filter / no verifier / no selective-write / no
deletion / single-agent).
Why this is distinct: Unlike general leaderboards, FABLE bakes in ground-truth
timelines + controlled misinformation/corrections, so we can precisely measure
false-memory formation and repair—not just raw QA accuracy.