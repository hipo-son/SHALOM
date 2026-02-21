<h1 align="center">SHALOM</h1>

<p align="center">
  <strong>System of Hierarchical Agents for Logical Orchestration of Materials</strong>
</p>

<p align="center">
  <a href="https://github.com/hipo-son/SHALOM/actions"><img src="https://github.com/hipo-son/SHALOM/actions/workflows/ci.yml/badge.svg" alt="CI Status"></a>
  <a href="https://codecov.io/gh/hipo-son/SHALOM"><img src="https://codecov.io/gh/hipo-son/SHALOM/branch/main/graph/badge.svg" alt="Coverage (>85%)"></a>
  <a href="https://pypi.org/project/shalom/"><img src="https://img.shields.io/pypi/v/shalom.svg" alt="PyPI version"></a>
  <a href="https://shalom.readthedocs.io/en/latest/"><img src="https://readthedocs.org/projects/shalom/badge/?version=latest" alt="Documentation Status"></a>
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
- **Secure sandboxed execution** — Run LLM-generated Python/ASE code safely with import restrictions, timeouts, and resource limits.
- **Pluggable LLM backends** — Swap between OpenAI, Anthropic, or custom providers with a single parameter change.
- **HPC-native design** — First-class Slurm/MCP integration for submitting and monitoring jobs on supercomputers.
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
|           ^                                 | POSCAR        |
|           | (Self-Correction Loop)          v               |
+-----------|-------------------------------------------------+
            |                                 | HPC Execution
            | Feedback                        v OUTCAR
+-----------|-------------------------------------------------+
|           |         3. REVIEW LAYER                         |
|  (Review Agent) <- Evaluate VASP Output (Energy, Forces)    |
+-------------------------------------------------------------+
```

This pipeline is one instantiation of the SHALOM framework. The same core components — `LLMProvider`, `SafeExecutor`, Pydantic schemas, and the agent base classes — can be reused to build entirely different workflows (e.g., defect screening, catalyst optimization, phase-diagram exploration).

## Installation

```bash
pip install shalom            # core dependencies
pip install shalom[all]       # includes pymatgen, HPC toolkits, docs
pip install -e ".[dev]"       # development (tests, linting, type-checking)
```

For HPC clusters with containerized deployments:
```bash
docker pull ghcr.io/hipo-son/shalom:latest
```

## Quick Start

The example below runs the built-in material discovery pipeline — the first proof-of-concept use case:

```python
from shalom.core.llm_provider import LLMProvider
from shalom.agents.design_layer import CoarseSelector, FineSelector

# 1. Initialize an LLM backend
llm = LLMProvider(provider_type="openai", model_name="gpt-4o")
objective = "Find a stable 2D transition metal dichalcogenide with bandgap > 1.0eV"

# 2. Design Layer: Coarse Selection
coarse = CoarseSelector(llm)
candidates = coarse.select(objective)

# 3. Design Layer: Fine Ranking
fine = FineSelector(llm)
winner = fine.rank_and_select(objective, candidates)

print(f"Top Material: {winner.candidate.material_name} (Score: {winner.score})")
```

For the full pipeline (Simulation + Review layers) and guidance on building custom agent hierarchies, see the [Documentation](https://shalom.readthedocs.io/en/latest/).

## Documentation
Read the full API reference and tutorials on [ReadTheDocs](https://shalom.readthedocs.io).

## Contributing
We welcome contributions! Please review our [Contribution Guidelines](CONTRIBUTING.md) and [Code of Conduct](CODE_OF_CONDUCT.md).

## Citation
If you use SHALOM in your research, please cite our paper:
```bibtex
@misc{shalom2026,
  author = {User Name and SHALOM Contributors},
  title = {SHALOM: System of Hierarchical Agents for Logical Orchestration of Materials},
  year = {2026},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/hipo-son/SHALOM}}
}
```

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
