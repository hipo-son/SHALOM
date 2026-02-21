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

## 🔬 Overview

**SHALOM** is a premier autonomous multi-agent framework tailored for computational materials science and reasoning-driven workflow orchestration. Powered by Large Language Models (LLMs), it executes closed-loop materials discovery by simulating human-expert workflows—from hypothesizing novel compositions (Design) and building 3D configurations (Simulation), to verifying structural integrity via VASP (Review). 

SHALOM embraces absolute **reproducibility** with deterministic prompt tracing, high-performance containerization (HPC/SLURM), and secure sandboxed pipeline executions.

## 🏗️ Architecture

The framework is constructed around a tri-layer hierarchy implementing a closed-loop self-correction cycle:

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

## 🚀 Installation

SHALOM is available on PyPI and can be installed via pip. We recommend using a Conda environment for HPC deployments.

```bash
# Basic Installation
pip install shalom

# Full Installation (includes pymatgen and HPC toolkits)
pip install shalom[all]
```

### HPC & Slurm
For clusters where containerization is preferred, pull the SHALOM Docker image:
```bash
docker pull ghcr.io/hipo-son/shalom:latest
```

## ⚡ Quick Start

```python
from shalom.core.llm_provider import LLMProvider
from shalom.agents.design_layer import CoarseSelector, FineSelector

# 1. Provide an API Key and Initialize
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

For advanced usage integrating the `SimulationLayer` and `ReviewLayer`, see the [Documentation](https://shalom.readthedocs.io/en/latest/).

## 📖 Documentation
Read the full API reference and tutorials on [ReadTheDocs](https://shalom.readthedocs.io).

## 🤝 Contributing
We welcome contributions! Please review our [Contribution Guidelines](CONTRIBUTING.md) and [Code of Conduct](CODE_OF_CONDUCT.md).

## 📝 Citation
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

## 📄 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
