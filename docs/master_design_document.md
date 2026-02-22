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

### v1.0 — Static DFT Foundation (arXiv preprint target)

Positioning: "A robust, sandbox-secured, Pydantic-schema-driven foundation for autonomous DFT workflows with dual-backend support."

| Phase | Scope | Status |
|-------|-------|--------|
| **Phase 1: Core Library & VASP Automation** | Agent pipeline (Design → Simulation → Review), VASP input generation with structure-aware auto-detection, error recovery engine | Complete |
| **Phase 2: Configuration Externalization** | Prompt/config externalization to `.md`/`.yaml`, loader with caching + validation + fallback, Pydantic schema validation | Complete |
| **Phase 3: QE Backend & Token Compression** | Quantum ESPRESSO backend (write/parse), SSSP metadata, Materials Project client, CLI, token-aware context compression (`compress_error_log`), 643 tests at 95.6% coverage | Complete |
| **Phase 4: DFT Execution & Self-Correction** | Local QE runner (`subprocess`), `--execute` CLI flag, execution → ReviewAgent auto-loop, error recovery retry (max 3) | In Progress |
| **Phase 5: VASP-Slurm HPC Integration** | Slurm job submission/monitoring, VASP execution runner, end-to-end testing with bulk materials | Planned |

### v2.0 — Multi-scale & Autonomous (journal submission target)

Positioning: "Beyond static structure analysis — agents autonomously orchestrate multi-scale simulations (DFT + MD) and generate optimal execution recipes from natural language objectives."

Target journals: Nature Computational Science, npj Computational Materials, Digital Discovery.

| Phase | Scope | Status |
|-------|-------|--------|
| **Phase 6: Dynamic Recipe Generator** | `WorkflowRecipe` Pydantic model, Recipe Recommender Agent that infers optimal execution DAG from user intent (e.g., "calculate Li-ion diffusion" → [VASP relax → VASP bandgap → LAMMPS NVT MD]) | Planned |
| **Phase 7: LAMMPS & AIMD Integration** | LAMMPS backend (classical MD), AIMD workflow support, trajectory analysis agents | Planned |
| **Phase 8: Advanced Use Cases** | 2D/TMD systems, defect screening, catalyst design, phase-diagram exploration | Planned |
| **Phase 9: Performance & Metrics** | LLM API cost tracking, search performance benchmarks, reproducibility audit tooling | Planned |

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
