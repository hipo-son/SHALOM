<h1 align="center">SHALOM</h1>

<p align="center">
  <strong>System of Hierarchical Agents for Logical Orchestration of Materials</strong>
</p>

<p align="center">
  <a href="https://github.com/hipo-son/SHALOM/actions"><img src="https://github.com/hipo-son/SHALOM/actions/workflows/ci.yml/badge.svg" alt="CI Status"></a>
  <a href="https://codecov.io/gh/hipo-son/SHALOM"><img src="https://codecov.io/gh/hipo-son/SHALOM/branch/main/graph/badge.svg" alt="Coverage (>85%)"></a>
  <!-- PyPI badge — uncomment after publishing to PyPI: -->
  <!-- <a href="https://pypi.org/project/shalom/"><img src="https://img.shields.io/pypi/v/shalom.svg" alt="PyPI version"></a> -->
  <!-- ReadTheDocs badge — uncomment after RTD project setup: -->
  <!-- <a href="https://shalom.readthedocs.io/en/latest/"><img src="https://readthedocs.org/projects/shalom/badge/?version=latest" alt="Documentation Status"></a> -->
  <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT"></a>
  <!-- DOI badge will be added after Zenodo integration -->
  <!-- <a href="https://zenodo.org/badge/latestdoi/XXXXXX"><img src="https://zenodo.org/badge/XXXXXX.svg" alt="DOI"></a> -->
</p>

---

## Overview

**SHALOM** is a general-purpose hierarchical agent orchestration framework for computational materials science. Rather than being a single-purpose tool, SHALOM provides the **infrastructure** — agent lifecycle management, structured LLM communication, sandboxed code execution, HPC integration, and closed-loop feedback — needed to compose autonomous multi-agent workflows for any materials science task.

The framework is **domain-aware but task-agnostic**: agents can be assembled into arbitrary hierarchies to tackle material screening, structure optimization, property prediction, or any workflow expressible as a sequence of LLM-driven decisions and computational validations.

### Why a Framework?

Existing LLM-agent systems are either too general (lacking domain primitives for crystallography, DFT, and HPC) or too narrow (hard-coded for a single workflow). SHALOM bridges this gap by offering:

- **Hierarchical agent composition** — Nest agents in configurable layers (planning, execution, evaluation, auditing) that communicate via typed Pydantic schemas.
- **Secure sandboxed execution** — Run LLM-generated Python/ASE code safely with whitelist-only builtins, import restrictions, timeouts, and audit logging.
- **Pluggable LLM backends** — Swap between OpenAI, Anthropic, or local/self-hosted models (Ollama, vLLM, llama.cpp) via `base_url` parameter.
- **Dual DFT backend support** — Quantum ESPRESSO (open-source, personal workstations) and VASP (licensed, HPC clusters) as first-class DFT solvers, with a unified abstraction layer.
- **HPC-native design** — First-class Slurm integration for submitting and monitoring jobs on supercomputers.
- **MCP server for Claude Code** — 10 tools accessible via natural language through Claude Code (no API key needed for deterministic tools).
- **Deterministic reproducibility** — Seed-locked prompts and structured outputs ensure every run is traceable and repeatable.

### Proof of Concept: Autonomous Material Discovery

To validate the framework, SHALOM ships with a complete **material discovery pipeline** as its first use case. This pipeline demonstrates the full agent lifecycle through three cooperating layers:

```text
+-------------------------------------------------------------+
|                     1. DESIGN LAYER                         |
|  [NL Objective] -> (Coarse Selector) -> (Fine Selector)     |
+------------------------------+------------------------------+
                               | RankedMaterial winner
+------------------------------v------------------------------+
|                   2. SIMULATION LAYER                       |
|  (Geometry Generator) -> SafeExecutor -> (Form Filler)      |
|           ^                                 | Input Files   |
|           | (Self-Correction Loop)          v               |
+-----------|-------------------------------------------------+
            |                                 | DFT Execution
            | Feedback                        v (QE or VASP)
+-----------|-------------------------------------------------+
|           |         3. REVIEW LAYER                         |
|  (Review Agent) <- Evaluate DFT Output (Energy, Forces)     |
+-------------------------------------------------------------+
```

This pipeline is one instantiation of the SHALOM framework. The same core components — `LLMProvider`, `SafeExecutor`, Pydantic schemas, and the agent base classes — can be reused to build entirely different workflows (e.g., defect screening, catalyst optimization, phase-diagram exploration).

### Supported DFT Backends

| Backend | License | Typical Environment | I/O Formats |
|---------|---------|---------------------|-------------|
| **Quantum ESPRESSO** | Open-source (GPL) | Personal workstations, small clusters | `pw.x` input / XML output |
| **VASP** | Commercial license | HPC clusters with Slurm | POSCAR, INCAR, KPOINTS / OUTCAR |

SHALOM abstracts DFT-specific details behind a unified interface so that agents operate on the same schema regardless of the solver. Individual researchers can prototype with Quantum ESPRESSO locally, then scale to VASP on institutional HPC resources without changing agent logic.

## Getting Started

> **Note**: SHALOM is not yet published to PyPI — install from source via GitHub clone (see below).
> `pip install shalom` is planned for a future release.

### Step 1 — Clone the Repository

Clone into your `Desktop` or `projects` folder to keep things organized:

```bash
cd ~/Desktop          # or: cd ~/projects
git clone https://github.com/hipo-son/SHALOM.git
cd SHALOM
```

### Step 2 — Create a Python Environment

**Option A: conda (recommended)** — best for managing pymatgen/ASE dependencies:

```bash
conda env create -f environment.yml
conda activate shalom-env
```

**Option B: venv** — if you don't have conda:

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e ".[dev,mp]"
```

To enable band/DOS plotting (matplotlib + seekpath):

```bash
pip install -e ".[plotting]"
# or with everything:
pip install -e ".[all]"
```

Verify the installation:

```bash
python -m shalom --help
```

### Step 3 — Configure API Keys

Copy the template and fill in your keys:

```bash
cp .env.example .env
# Edit .env with your keys, then:
source .env
```

| Key | Purpose | Where to Get |
|-----|---------|--------------|
| `OPENAI_API_KEY` | LLM agents (Design / Review layer) | [platform.openai.com/api-keys](https://platform.openai.com/api-keys) |
| `ANTHROPIC_API_KEY` | Alternative to OpenAI | [console.anthropic.com](https://console.anthropic.com/) |
| `MP_API_KEY` | Fetch structures by MP ID or formula | [next-gen.materialsproject.org/api](https://next-gen.materialsproject.org/api) (free) |
| `SHALOM_PSEUDO_DIR` | QE pseudopotential directory | Set after running `setup-qe --download` |
| `SHALOM_LLM_BASE_URL` | Local LLM server URL (Ollama, vLLM, etc.) | Optional — bypasses API key requirement |
| `SHALOM_AUDIT_LOG` | Audit log file path | Optional — e.g. `~/.shalom/audit.log` |

Only **one** of `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` is needed (or use `SHALOM_LLM_BASE_URL` for local models). `MP_API_KEY` is required only when looking up structures by MP ID or chemical formula.

### Step 4 — Set Up Quantum ESPRESSO (QE backend only)

Skip this step if you only use the VASP backend.

```bash
# Linux / WSL2 (Ubuntu/Debian)
sudo apt install quantum-espresso

# conda
conda install -c conda-forge qe

# Windows (native) → pw.x is not supported; use WSL2:
wsl -d Ubuntu-22.04
sudo apt install quantum-espresso
```

Download SSSP pseudopotentials for your elements:

```bash
python -m shalom setup-qe --elements Si,Fe,O --download

# Check overall QE environment
python -m shalom setup-qe
```

### Step 5 — Run Your First Calculation

Choose the path that matches what you have available:

```bash
# Path A — local structure file (no API keys needed)
python -m shalom run --structure POSCAR --backend vasp

# Path B — Materials Project ID (MP_API_KEY required)
python -m shalom run mp-19717 --backend vasp

# Path C — chemical formula + QE (MP_API_KEY + QE install required)
python -m shalom run Si --backend qe --calc scf
```

Each run creates a timestamped output folder (e.g., `Si_qe_static/`) containing the DFT input files and a **`README.md`** explaining what was generated and how to proceed.

---

## CLI Reference

```bash
# ── Basic runs ────────────────────────────────────────────────────────────────
python -m shalom run mp-19717                              # VASP vc-relax (default)
python -m shalom run Fe2O3 --backend qe --calc scf         # QE SCF
python -m shalom run --structure POSCAR --backend vasp      # Local structure file
python -m shalom run mp-19717 --set ENCUT=600               # Override a parameter
python -m shalom run Fe2O3 --backend qe --set ecutwfc=80    # QE parameter override
python -m shalom run Si --backend qe --calc scf --execute   # Generate + run pw.x
python -m shalom run Si --backend qe -x -np 4 --timeout 7200  # 4 MPI procs, 2h timeout

# ── Output location ───────────────────────────────────────────────────────────
# Default: ~/Desktop/shalom-runs/{formula}_{mp_id}_{backend}_{calc}/
python -m shalom run Si -p silicon_study                    # group runs: shalom-runs/silicon_study/Si_.../
python -m shalom run Si -w /data/dft_runs                  # custom workspace root
python -m shalom run Si -o ./my_output                     # explicit path (bypasses workspace)
# Set permanently:  export SHALOM_WORKSPACE=~/Desktop/shalom-runs

# ── QE setup ──────────────────────────────────────────────────────────────────
python -m shalom setup-qe                                   # Check QE environment
python -m shalom setup-qe --elements Si,Fe --download       # Download pseudopotentials

# ── Band/DOS plotting (requires pip install shalom[plotting]) ──────────────────
python -m shalom plot ./03_bands --bands                   # Band structure plot
python -m shalom plot ./04_nscf --dos                      # DOS plot
python -m shalom plot ./03_bands --bands --fermi-from ./04_nscf  # NSCF Fermi energy
python -m shalom plot ./03_bands --bands --emin -8 --emax 6 --title "Si bands"

# ── 5-step QE workflow (vc-relax → scf → bands → nscf → dos.x → plots) ───────
python -m shalom workflow Si -o ./si_wf -np 4             # Full sequential workflow
python -m shalom workflow Si --skip-relax -np 4           # Start from SCF (no relax)
python -m shalom workflow mp-19717 -b qe -np 8 --dos-emin -20  # Custom DOS window

# ── Convergence tests (run cutoff first, then kpoints) ────────────────────────
python -m shalom converge Si --test cutoff --values 30,40,50,60,80 -np 2
python -m shalom converge Si --test kpoints --values 20,30,40,50 --ecutwfc 60

# ── LLM-driven autonomous pipeline ──────────────────────────────────────────
python -m shalom pipeline "Find a 2D HER catalyst"              # Full pipeline (OpenAI)
python -m shalom pipeline "Stable cathode" --provider anthropic  # Use Claude
python -m shalom pipeline "MoS2 band structure" --material MoS2  # Skip Design layer
python -m shalom pipeline "Find catalyst" -b qe -x -np 4        # Execute DFT

# ── Local LLM (no API key needed) ─────────────────────────────────────────
python -m shalom pipeline "Find HER catalyst" --base-url http://localhost:11434/v1
```

## Quick Demo

Try the interactive multi-agent demo with rich console output:

```bash
export OPENAI_API_KEY='sk-...'          # or ANTHROPIC_API_KEY

python -m shalom.demo --dry-run                    # preview (no API calls, $0)
python -m shalom.demo --scenario smoke_test         # real API (~$0.03)
python -m shalom.demo --scenario her_catalyst       # multi-agent (~$0.12)
python -m shalom.demo --list                        # list available scenarios
```

The demo runs the full pipeline (Design → Simulation) with real-time cost tracking and JSON report generation. Use `--dry-run` for zero-cost offline testing.

## MCP Server (Claude Code Integration)

SHALOM includes an MCP (Model Context Protocol) server that lets Claude Code call DFT tools directly via natural language — no separate API key needed for deterministic tools.

```bash
# Install MCP support
pip install "shalom[mcp]"

# Register in Claude Code (one-time setup)
claude mcp add shalom -- python -m shalom.mcp_server

# Or use the project-scoped .mcp.json (already in repo root)
```

After setup, tell Claude Code things like:
- "Si의 SCF 계산 입력 파일을 만들어줘"
- "mp-1040425 밴드 구조 계산해줘"

**10 MCP tools**: `search_material`, `generate_dft_input`, `run_workflow`, `execute_dft`, `parse_dft_output`, `plot_bands`, `plot_dos`, `run_convergence`, `check_qe_setup`, `run_pipeline`

The `run_pipeline` tool runs the full multi-agent LLM pipeline and supports `base_url` for local LLM servers as an alternative to external API keys.

## Python API

```python
from shalom.core.llm_provider import LLMProvider
from shalom.agents.design_layer import CoarseSelector, FineSelector

# Initialize LLM backend (reads OPENAI_API_KEY from environment)
llm = LLMProvider(provider_type="openai", model_name="gpt-4o")

# Or use a local LLM (no API key needed):
# llm = LLMProvider(provider_type="openai", model_name="llama3", base_url="http://localhost:11434/v1")

objective = "Find a stable 2D transition metal dichalcogenide with bandgap > 1.0eV"

coarse = CoarseSelector(llm)
candidates = coarse.select(objective)

fine = FineSelector(llm)
winner = fine.rank_and_select(objective, candidates)
print(f"Top Material: {winner.candidate.material_name} (Score: {winner.score})")
```

For the full pipeline (Simulation + Review layers), see the [Documentation](https://shalom.readthedocs.io/en/latest/).

## Known Issues

- **VASP OUTCAR parsing**: 19 tests fail with `pymatgen>=2025.10` due to an upstream `IndexError` in `Outcar.__init__`. `environment.yml` pins `pymatgen<2025.10` to avoid this. QE, agent, and CLI tests are unaffected.
- **QE on Windows (native)**: `pw.x` requires WSL2. File generation works on native Windows; execution does not.

## HPC / Docker

For containerized deployments on HPC clusters:

```bash
docker pull ghcr.io/hipo-son/shalom:latest
```

## Roadmap

| Phase | Target | Key Features | Status |
|-------|--------|-------------|--------|
| **Phase 1** | arXiv preprint + PyPI | VASP + QE dual backend, 3-layer agent pipeline, error recovery, local QE execution, CLI, MCP server (10 tools), local LLM support, token-aware compression, band/DOS plotting, convergence tests, 5-step workflow, audit logging | Code complete (1124 tests, 94.6% coverage) |
| **Phase 2** | Engine expansion | VASP-Slurm HPC, LAMMPS/AIMD integration, Dynamic Recipe Generator, 100+ self-correction benchmarks | Planned |
| **Phase 3** | Journal submission | Main paper with benchmark data, advanced use cases (2D, defects, catalysts) | Planned |

See the [Master Design Document](docs/master_design_document.md) for detailed milestones and publication strategy.

## Documentation
Read the full API reference and tutorials on [ReadTheDocs](https://shalom.readthedocs.io).

## Contributing
We welcome contributions! Please review our [Contribution Guidelines](CONTRIBUTING.md) and [Code of Conduct](CODE_OF_CONDUCT.md).

## Citation
If you use SHALOM in your research, please cite our paper:
```bibtex
@misc{shalom2026,
  author = {Shinwon Son},
  title = {SHALOM: System of Hierarchical Agents for Logical Orchestration of Materials},
  year = {2026},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/hipo-son/SHALOM}}
}
```

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
