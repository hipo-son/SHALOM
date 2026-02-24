# SHALOM Agent Guidelines & Operating Principles

This document serves as the **Centralized Domain-Knowledge Injection Module** for all LLM agents within the SHALOM framework. It defines the core rules, expected behaviors, and constraints for autonomous materials science simulations.

All agents (Design, Simulation, Review, Orchestrator) MUST adhere to these guidelines when reasoning, generating scripts, or self-correcting errors.

---

## 🏗️ 1. Architecture: The Flexible DAG Pipeline
SHALOM is not a rigid linear script. It is a highly flexible, state-driven orchestrator of quantum (DFT) and classical (MD) simulations.

*   **Autonomy:** Agents should infer the most efficient path (Recipe) based on the user's objective.
    *   *Example:* If a relaxed structure is provided, skip the `Design` phase and go straight to `Simulation`.
*   **Modularity:** Always communicate between layers exclusively using the defined **Pydantic Schemas**. Do not pass unstructured text between agents.
*   **Resilience (Self-Correction):** The system's primary goal is 100% convergence without human intervention. The `Review` agent must critically analyze failures and dynamically route back to `Simulation` with strategic parameter adjustments.

---

## 🧪 2. Domain Knowledge & Physics Constraints
When generating computational setups (VASP, QE, LAMMPS), agents must prioritize physical rigor over rapid execution.

### General Rules
*   Never invent (hallucinate) pseudopotentials or empirical potentials that do not exist in the standard libraries.
*   Ensure that the simulation cell size is appropriate for the target property (e.g., minimum 10-15 Å padding for molecules/surfaces, sufficient supercell size for defect calculations).

### VASP Specifics
*   **Convergence:** Always check both electronic (EDIFF) and ionic (EDIFFG) convergence. Default to high precision (`PREC = Accurate`).
*   **Smearing (ISMEAR):**
    *   Metals: Methfessel-Paxton (`ISMEAR = 1` or `2`) with `SIGMA = 0.1` or `0.2`.
    *   Semiconductors/Insulators: Tetrahedron method with Blöchl corrections (`ISMEAR = -5`).
*   **K-points:** Use a dense-enough $\Gamma$-centered grid. The density must scale inversely with the lattice vectors.

### QE (Quantum ESPRESSO) Specifics
*   **Units (Rydberg):** degauss = SIGMA(eV) / 13.6057, conv_thr = EDIFF(eV) / 13.6057.
*   **Pseudopotentials:** SSSP Efficiency v1.3.0 (PBE) only. Never hallucinate filenames — look up from `sssp_metadata.yaml`.
*   **Magnetism:** starting_magnetization = MAGMOM / z_valence (from SSSP, NOT /10).
*   **Relaxation:** VASP ISIF=3 → QE `vc-relax` (not `relax`).
*   **2D:** assume_isolated='2D', vdw_corr='dft-d3', dftd3_version=4. For vc-relax only: cell_dofree='2Dxy'.
*   **ecutrho:** Per-element from SSSP metadata (not blanket 8×ecutwfc).

### Error Handling & Self-Correction (The Review Layer)
When analyzing a failed run, the framework applies **automatic context compression** via `compress_error_log()` in `shalom/backends/_compression.py`:
1.  Error keywords (loaded from `error_patterns.yaml`) are preserved with ±3 lines of context, plus the tail of the output. The result is truncated to a token budget before LLM evaluation.
2.  If electronic steps fail to converge (e.g., hitting `NELM` without reaching `EDIFF`):
    *   Retry with modified mixing parameters (`AMIX`, `BMIX`).
    *   Change the algorithm (`ALGO = Fast` $\to$ `Normal` $\to$ `VeryFast`).
3.  If ionic relaxation fails (geometry not converging):
    *   Reduce step size (`POTIM`).
    *   Change the optimization algorithm (`IBRION = 2` $\to$ `1` or `3`).

---

## ⚡ 3. Software Engineering & API Efficiency
To maintain a scalable and cost-effective framework:

*   **Context Compression:** Never pass raw, massive log files (`OUTCAR`, `lammps.log`) directly to the LLM. Always use parsers to extract only the relevant physical quantities (Energy, Forces, Bandgap) and the exact error trace.
*   **Deterministic Output:** Ensure all code generated for the `SafeExecutor` is strictly sandboxed. Do not include commands that attempt network access or write outside the designated scratch directory.
*   **Testing State:** Treat every generated simulation as a "hypothesis test." If the test (calculation) fails, the agent must propose a new hypothesis (parameter fix) and retry until success or a hard limit is reached.

---

## 🔒 4. Sandbox Security & Audit

### SafeExecutor Constraints
The `SafeExecutor` (`shalom/core/sandbox.py`) uses a **whitelist-only** approach for builtins. Generated code runs in a restricted namespace where:
*   **Blocked builtins:** `eval`, `exec`, `compile` (arbitrary code execution), `open` (filesystem access), `breakpoint` (debugger escape), `getattr`/`setattr`/`delattr` (attribute introspection → `__class__` sandbox escape), `globals`/`locals`/`vars` (scope inspection), `type` (metaclass manipulation → `__subclasses__` escape).
*   **Allowed builtins:** `abs`, `all`, `any`, `bool`, `dict`, `enumerate`, `float`, `hasattr`, `int`, `isinstance`, `len`, `list`, `map`, `max`, `min`, `print`, `range`, `round`, `set`, `sorted`, `str`, `sum`, `tuple`, `zip`, `True`, `False`, `None`.
*   **Import restriction:** `__import__` is set to `None`. Trusted libraries (ASE, NumPy) are passed explicitly via `local_vars`.
*   **Timeout:** POSIX uses `signal.SIGALRM`; Windows uses `ThreadPoolExecutor` fallback.

Agents MUST NOT generate code that relies on blocked builtins. Use the allowed utilities and the libraries injected by the framework.

### Audit Logging
All LLM API calls and pipeline executions are logged when `SHALOM_AUDIT_LOG` is set:
*   **Events:** `llm_call` (provider, model, base_url, response_model), `pipeline_start` (objective, backend, steps).
*   **Format:** JSON-line file, one JSON object per line with UTC timestamp.
*   Audit logging never blocks execution — failures are silently caught.
