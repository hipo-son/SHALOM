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

### 2.3 Sandboxed Execution

The `SafeExecutor` provides a secure environment for running LLM-generated Python code:

- Import restrictions (`__import__` blocked in builtins).
- Cross-platform timeout enforcement (SIGALRM on POSIX, ThreadPoolExecutor on Windows).
- Explicit allowlists for injected variables (e.g., ASE's `bulk`, `surface`).

### 2.4 Dual DFT Backend

SHALOM treats the DFT solver as a swappable backend behind a unified abstraction:

| Backend | License | Typical Environment | I/O Formats |
|---------|---------|---------------------|-------------|
| **Quantum ESPRESSO** | Open-source (GPL) | Personal workstations, small clusters | `pw.x` input / XML output |
| **VASP** | Commercial license | HPC clusters with Slurm | POSCAR, INCAR, KPOINTS / OUTCAR |

Individual researchers typically use **Quantum ESPRESSO** for local prototyping and validation; groups with institutional HPC resources use **VASP** for production-scale screening. Agents operate on a common schema (structure → energy/forces/convergence) so switching backends requires no changes to agent logic.

### 2.5 HPC Integration (MCP)

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

## 4. Development Milestones

- **Phase 1: Core Library & MCP Environment Setup** — Python library structure, ASE-based Geometry Generator prompts, and Form Filler setup. *(Current)*
- **Phase 2: Triage-Ranking Agent Loop** — Coarse/Fine Selector prompt pipeline construction.
- **Phase 3: DFT Integration & Self-Correction** — Quantum ESPRESSO local runner, VASP-Slurm HPC integration, end-to-end testing with bulk materials.
- **Phase 4: Advanced Use Cases & Open-Source Release** — Extension to 2D/TMD systems, new workflow templates (defect screening, catalyst design), search performance metrics tooling.

## 5. Software Architecture

Library-centric design implemented in Python for reproducibility and HPC integration:

| Layer | Components |
|-------|------------|
| **Agent Framework** | Base agent classes, hierarchical composition, schema-driven communication |
| **Tool System** | ASE builder, DFT I/O parsers (VASP POSCAR/OUTCAR, QE pw.x/XML), structure validators |
| **DFT Backend** | Quantum ESPRESSO (local), VASP (HPC), unified solver abstraction |
| **Execution Layer** | Slurm job submission, MCP server, SafeExecutor sandbox |
| **Provider Interface** | LLMProvider (OpenAI, Anthropic), structured output enforcement |
