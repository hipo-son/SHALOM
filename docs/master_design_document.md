# SHALOM Master Design Document

## 1. Vision and Scope

**SHALOM** (System of Hierarchical Agents for Logical Orchestration of Materials) is a general-purpose multi-agent orchestration framework for computational materials science. It combines a highly extensible Python library with an MCP (Model Context Protocol)-based architecture to enable autonomous, LLM-driven workflows across diverse materials science tasks.

### Framework vs. Use Case

| Aspect | SHALOM Framework | Material Discovery Pipeline |
|--------|------------------|-----------------------------|
| **Purpose** | General-purpose agent orchestration infrastructure | First proof-of-concept validating the framework |
| **Scope** | Any materials science workflow expressible as hierarchical agent collaboration | Closed-loop search for novel materials with target properties |
| **Components** | LLMProvider, SafeExecutor, Pydantic schemas, agent base classes, HPC/MCP integration | CoarseSelector, FineSelector, GeometryGenerator, FormFiller, GeometryReviewer, ReviewAgent |
| **Extensibility** | New agent hierarchies can be composed for arbitrary tasks | Fixed 3-layer pipeline (Design → Simulation → Review) |

The material discovery pipeline demonstrates the framework's capabilities but is not its boundary. The same core infrastructure supports defect screening, catalyst optimization, phase-diagram exploration, and any other multi-step computational workflow.

## 2. Framework Architecture

The framework provides four foundational pillars:

### 2.1 Agent Framework

Agents are the primary building blocks. Each agent:

- Wraps a specific LLM interaction pattern (selection, code generation, evaluation, etc.).
- Communicates through strongly typed Pydantic schemas — no free-form string passing between agents.
- Can be nested hierarchically: a parent agent delegates subtasks to child agents and aggregates results.

Agent roles can be categorized as:

- **Planner Agent**: Defines objectives, allocates budget, manages search strategies.
- **Executor Agent**: Submits Slurm jobs, monitors execution, handles failures.
- **Evaluator Agent**: Parses results, computes metrics, ranks candidates.
- **Critic/Auditor Agent**: Validates tool usage, ensures safety and reproducibility.

### 2.2 LLM Provider Interface

A unified `LLMProvider` class abstracts multiple LLM backends (OpenAI, Anthropic) behind a single `generate_structured_output()` method. This allows:

- Swapping providers with a one-line change.
- Enforcing structured JSON output via Pydantic models on every call.
- Deterministic replay through seed-locked prompts.

### 2.3 Externalized Configuration System

All LLM system prompts and physics constants are externalized from Python source code:

```text
shalom/
├── prompts/           # LLM system prompts (.md, version-tagged)
│   ├── coarse_selector.md
│   ├── fine_selector.md
│   ├── geometry_generator.md
│   ├── review_agent.md
│   └── eval_*.md      # 6 specialist evaluator prompts + confidence rule
│
├── config/            # Physics/DFT settings (.yaml, with literature refs)
│   ├── potcar_mapping.yaml       # PBE_54 POTCAR variants
│   ├── enmax_values.yaml         # Per-element ENMAX (eV)
│   ├── magnetic_elements.yaml    # Default MAGMOM values
│   ├── hubbard_u.yaml            # Dudarev U values (PBE)
│   ├── metallic_elements.yaml    # Pure metal detection set
│   ├── incar_presets.yaml        # INCAR presets by calc_type × accuracy
│   ├── error_patterns.yaml       # VASP error detection patterns
│   ├── correction_strategies.yaml # Progressive error correction
│   ├── evaluator_weights.yaml    # Multi-agent scoring weights
│   ├── sssp_metadata.yaml        # SSSP Efficiency v1.3.0 pseudopotential metadata
│   └── qe_presets.yaml           # QE pw.x presets (scf/relax/vc-relax/bands/nscf)
│
├── _config_loader.py  # load_prompt(), load_config() with caching
├── _config_schemas.py # Pydantic validation for critical configs
└── _defaults.py       # Hardcoded fallback (works without external files)
```

**Design decisions:**

| Decision | Rationale |
|----------|-----------|
| `.md` for prompts | Human-readable; domain experts can edit without touching Python; clean git diffs |
| `.yaml` for configs | Structured data with inline comments for literature references |
| `_defaults.py` fallback | Package works even without data files (e.g., incomplete wheel) |
| `copy.deepcopy()` on config return | Prevents cache poisoning when callers mutate returned dicts |
| Fail-Fast on YAML syntax errors | Corrupted config must never silently fall back (avoids wasted HPC hours) |
| CRLF → LF normalization | Windows git may inject `\r\n`; prompts are always `\n`-normalized |
| Pydantic schema validation | Critical configs (POTCAR mapping, Hubbard U) are checked at load time |

### 2.4 Sandboxed Execution

The `SafeExecutor` provides a secure environment for running LLM-generated Python code:

- Import restrictions (`__import__` blocked in builtins).
- Cross-platform timeout enforcement (SIGALRM on POSIX, ThreadPoolExecutor on Windows).
- Explicit allowlists for injected variables (e.g., ASE's `bulk`, `surface`).

### 2.5 Dual DFT Backend

SHALOM treats the DFT solver as a swappable backend behind a unified abstraction:

| Backend | License | Typical Environment | I/O Formats |
|---------|---------|---------------------|-------------|
| **Quantum ESPRESSO** | Open-source (GPL) | Personal workstations, small clusters | `pw.x` input / XML output |
| **VASP** | Commercial license | HPC clusters with Slurm | POSCAR, INCAR, KPOINTS / OUTCAR |

Individual researchers typically use **Quantum ESPRESSO** for local prototyping and validation; groups with institutional HPC resources use **VASP** for production-scale screening. Agents operate on a common schema (structure → energy/forces/convergence) so switching backends requires no changes to agent logic.

### 2.6 HPC Integration (MCP)

The MCP-based integration module provides controlled HPC access:

- **DFT-Slurm MCP Server**: Predefined tools (`submit_slurm_job`, `check_job_status`, `read_dft_output`, etc.) ensure the LLM can only access HPC through safe, auditable interfaces. Separate tool implementations handle VASP and Quantum ESPRESSO specifics.
- **Self-Correction Algorithm**: When SCF fails, the review agent parses logs, adjusts parameters (INCAR for VASP, input namelist for QE), and resubmits.

## 3. Proof-of-Concept: Material Discovery Pipeline

The first use case validates the framework through an autonomous material discovery loop.

### 3.1 Design Layer — Triage-Ranking Logic

To prevent random exploration and reduce computational cost, a triage-ranking architecture is adopted:

- **Coarse Selector**: Screens 3–5 promising candidates from the vast chemical space (leveraging periodic table trends, d-band center theory, etc.).
- **Fine Selector**: Ranks candidates within the small pool based on expected properties and selects exactly one material closest to the target.

### 3.2 Simulation Layer — Pre-Validation and Automation

A three-step self-correction loop validates the physical soundness of DFT input files (POSCAR for VASP, `pw.x` input for QE) before submitting jobs:

- **Geometry Generator**: Writes ASE-based Python scripts from natural language requirements to produce initial structures.
- **Form Filler**: Analyzes the generated structure, evaluating layer count, vacuum thickness, atomic overlap, etc., using a standardized form.
- **Geometry Reviewer**: Makes pass/fail decisions based on the evaluation and issues correction instructions (up to 3–5 retries).

### 3.3 Review Layer — Result Evaluation

- Parses DFT output files (VASP OUTCAR or QE XML) to extract convergence status, total energy, forces, and band structure.
- Determines whether the target objective was achieved.
- On failure, generates structured feedback that is injected back into the Design Layer, closing the loop.

## 4. Release Strategy & Development Milestones

### Publication Strategy

| Paper | Venue | Content | Role |
|-------|-------|---------|------|
| **#1 Tech Report** | arXiv only | SHALOM v1.0 architecture — Pydantic schemas, SafeExecutor, VASP/QE dual backend (6–8 pages) | Stake priority; zero review overhead; cited as foundation by subsequent papers |
| **#2 Main Paper** | Nature Computational Science / npj Computational Materials / Digital Discovery | SHALOM v2.0 — autonomous multi-scale workflows, Dynamic Recipe Generator, self-correction AI with 100+ benchmark cases | Primary journal publication |

**Why arXiv-only for #1:**
- Eliminates 3–6 months of peer-review overhead — all effort redirected to the main paper.
- Open-access maximizes developer adoption (no paywall). Precedent: Word2Vec, GAN, Transformer papers accumulated thousands of citations from arXiv alone.
- Main paper (#2) references "#1 for technical details" and devotes 100% of page budget to novelty (autonomy, recipe generation, benchmark results).

---

### Phase 1: Infrastructure & Priority Staking (current)

**Goal:** Establish global technical priority for "LLM-based sandboxed DFT framework with dual-backend support."

#### 1a. Complete v1.0 codebase

| Milestone | Scope | Status |
|-----------|-------|--------|
| Core Library & VASP Automation | Agent pipeline (Design → Simulation → Review), VASP input generation, error recovery engine | Complete |
| Configuration Externalization | Prompt/config to `.md`/`.yaml`, loader with caching + validation + fallback, Pydantic schemas | Complete |
| QE Backend & Token Compression | QE backend (write/parse), SSSP metadata, MP client, CLI, `compress_error_log`, 676 tests at 95.7% coverage | Complete |
| Flexible Pipeline & Code Cleanup | `PipelineStep` enum, `synthesize_ranked_material()`, step-selective execution, config serialization, timing, bug fixes | Complete |
| DFT Execution & Self-Correction | Local QE runner (`subprocess`), `--execute` CLI flag, execution → ReviewAgent auto-loop, error recovery retry (max 3) | In Progress |
| VASP-Slurm HPC Integration | Slurm job submission/monitoring, VASP execution runner, end-to-end testing with bulk materials | Planned |

#### 1b. arXiv submission + open-source release

- Write 6–8 page tech report covering architecture, schema design, dual-backend abstraction, and benchmark.
- `pip install shalom` on PyPI.
- README links to arXiv preprint + short demo video.

---

### Phase 2: Multi-scale Engine & AI Brain Expansion (2–3 months)

**Goal:** Evolve from "command executor" to "scientist-level autonomous agent" with multi-scale simulation capability.

| Milestone | Scope | Status |
|-----------|-------|--------|
| LAMMPS & AIMD Integration | LAMMPS backend (classical MD), AIMD workflow support, trajectory analysis agents. "First unified pipeline spanning quantum (DFT) to classical (MD) scales." | Planned |
| Review Layer Hardening | Self-correction algorithm tuning; collect 100+ convergence benchmark cases (broken initial structures → agent-driven optimization → converged result) | Planned |
| Dynamic Recipe Generator | `WorkflowRecipe` Pydantic model, Recipe Recommender Agent that infers optimal execution DAG from user intent (e.g., "calculate Li-ion diffusion" → [VASP relax → VASP bandgap → LAMMPS NVT MD]) | Planned |

---

### Phase 3: Journal Submission & Advanced Use Cases

**Goal:** Submit main paper (#2) to top-tier journal with compelling benchmark data.

| Milestone | Scope | Status |
|-----------|-------|--------|
| Advanced Use Cases | 2D/TMD systems, defect screening, catalyst design, phase-diagram exploration | Planned |
| Performance & Metrics | LLM API cost tracking, search performance benchmarks, reproducibility audit tooling | Planned |
| Main Paper | Figures, benchmark tables (100+ cases), comparison with manual workflows (time-to-result), recipe generation demo | Planned |

## 5. Software Architecture

Library-centric design implemented in Python for reproducibility and HPC integration:

| Layer | Components | Status |
|-------|------------|--------|
| **Agent Framework** | Base agent classes, hierarchical composition, schema-driven communication | Complete |
| **Configuration** | `_config_loader` (prompt/config loading with caching, deepcopy, Fail-Fast), `_config_schemas` (Pydantic validation), `_defaults` (fallback) | Complete |
| **Tool System** | ASE builder, DFT I/O parsers (VASP POSCAR/OUTCAR, QE pw.x/XML), structure validators | Complete |
| **DFT Backend** | Quantum ESPRESSO (local), VASP (HPC), unified solver abstraction, error recovery engine, token-aware context compression | Complete |
| **Execution Layer** | Local QE runner, Slurm job submission, MCP server, SafeExecutor sandbox | Partial (sandbox complete, runners planned) |
| **Provider Interface** | LLMProvider (OpenAI, Anthropic), structured output enforcement | Complete |
| **Recipe System** | Dynamic Recipe Generator — LLM infers optimal execution DAG (`WorkflowRecipe`) from natural language | Planned (v2.0) |
