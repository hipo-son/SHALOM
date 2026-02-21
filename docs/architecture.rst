Architecture
============

SHALOM is a **general-purpose hierarchical agent orchestration framework**. This page describes both the framework-level architecture and the material discovery pipeline that serves as the first proof-of-concept use case.

Framework Architecture
----------------------

The framework provides four foundational layers that can be composed into arbitrary agent hierarchies:

.. code-block:: text

    +---------------------------------------------------------+
    |              SHALOM Framework Architecture               |
    +---------------------------------------------------------+
    |                                                         |
    |  +---------------------------------------------------+  |
    |  |  Agent Framework                                  |  |
    |  |  - Hierarchical agent composition                 |  |
    |  |  - Typed Pydantic schema communication            |  |
    |  |  - Role-based agents (Planner, Executor,          |  |
    |  |    Evaluator, Critic)                              |  |
    |  +---------------------------------------------------+  |
    |                          |                              |
    |  +---------------------------------------------------+  |
    |  |  Tool System                                      |  |
    |  |  - ASE structure builder                          |  |
    |  |  - DFT I/O parsers (VASP, Quantum ESPRESSO)      |  |
    |  |  - Structure validators (FormFiller)              |  |
    |  +---------------------------------------------------+  |
    |                          |                              |
    |  +---------------------------------------------------+  |
    |  |  DFT Backend Layer                                |  |
    |  |  - Quantum ESPRESSO (open-source, local)          |  |
    |  |  - VASP (licensed, HPC/Slurm)                     |  |
    |  |  - Unified solver abstraction                     |  |
    |  +---------------------------------------------------+  |
    |                          |                              |
    |  +---------------------------------------------------+  |
    |  |  Execution & HPC Layer                            |  |
    |  |  - SafeExecutor (sandboxed code execution)        |  |
    |  |  - Slurm/MCP job submission                       |  |
    |  |  - Cross-platform timeout enforcement             |  |
    |  +---------------------------------------------------+  |
    |                          |                              |
    |  +---------------------------------------------------+  |
    |  |  Provider Interface                               |  |
    |  |  - LLMProvider (OpenAI, Anthropic)                |  |
    |  |  - Structured output via Pydantic models          |  |
    |  |  - Seed-locked deterministic prompts              |  |
    |  +---------------------------------------------------+  |
    |                                                         |
    +---------------------------------------------------------+

Key design principles:

* **Composability** — Agents are assembled into hierarchies. A parent agent delegates subtasks to children and aggregates their results.
* **Type safety** — All inter-agent communication uses Pydantic models. No free-form string passing between agents.
* **Pluggability** — LLM backends, DFT solvers (Quantum ESPRESSO / VASP), HPC schedulers, and tool implementations can be swapped without changing agent logic.
* **Reproducibility** — Seed-locked prompts and structured outputs make every run traceable and repeatable.

DFT Backend Strategy
^^^^^^^^^^^^^^^^^^^^

SHALOM supports two first-class DFT backends to serve both individual researchers and HPC-equipped teams:

.. list-table::
   :header-rows: 1
   :widths: 20 15 30 35

   * - Backend
     - License
     - Typical Environment
     - I/O Formats
   * - **Quantum ESPRESSO**
     - Open-source (GPL)
     - Personal workstations, small clusters
     - ``pw.x`` input / XML output
   * - **VASP**
     - Commercial license
     - HPC clusters with Slurm
     - POSCAR, INCAR, KPOINTS / OUTCAR

A unified DFT abstraction layer ensures agents operate on a common schema (structure in, energy/forces/convergence out) regardless of the underlying solver. This lets researchers prototype locally with Quantum ESPRESSO and then scale to VASP on institutional HPC without changing any agent logic.

Use Case: Material Discovery Pipeline
--------------------------------------

The built-in material discovery pipeline demonstrates the framework by instantiating a three-layer closed-loop agent hierarchy:

.. code-block:: mermaid

    flowchart TD
        %% Define Nodes
        NL[("NL Objective")]
        DL_CS["Coarse Selector\n(Design Layer)"]
        MC["List[MaterialCandidate]"]
        DL_FS["Fine Selector\n(Design Layer)"]
        RM["RankedMaterial\n(Winner)"]

        SL_GG["Geometry Generator\n(Simulation Layer)"]
        ASE["ASE Python Code"]
        ATOMS[("ASE Atoms Object")]
        SL_FF["Form Filler\n(Simulation Layer)"]
        INPUT[("DFT Input Files\n(POSCAR / pw.x input)")]

        DFT["DFT Solver\n(QE or VASP)"]
        OUTPUT[("DFT Output\n(XML / OUTCAR)")]

        RL["Review Agent\n(Review Layer)"]
        RES["ReviewResult\n(Success & Metrics)"]
        FB["Feedback Generation"]

        %% Flow
        NL --> DL_CS
        DL_CS -- Selection --> MC
        MC --> DL_FS
        DL_FS -- Ranking --> RM
        RM --> SL_GG
        SL_GG -- Code Generation --> ASE
        ASE -- exec() via SafeExecutor --> ATOMS

        %% Simulation Validations
        ATOMS --> SL_FF
        SL_FF -- Validation Check --> SL_GG : Structure Invalid (Self-Correction)
        SL_FF -- Valid --> INPUT

        %% DFT Execution
        INPUT --> DFT
        DFT --> OUTPUT

        %% Review & Feedback Loop
        OUTPUT --> RL
        RL -- Parsing --> RES

        %% Closed-loop backward arrow
        RES -- Fail --> FB
        FB --> DL_CS : Inject Reason as Guidance

        RES -- Success --> END((Success!))

Layer Descriptions
^^^^^^^^^^^^^^^^^^

1. **Design Layer** — Triage-ranking architecture. The Coarse Selector screens 3–5 candidates; the Fine Selector ranks and picks the winner.
2. **Simulation Layer** — Self-correcting geometry pipeline. The Geometry Generator produces ASE code, the Form Filler validates physical soundness, and the Geometry Reviewer orchestrates retries.
3. **Review Layer** — Evaluates DFT output (energy, forces, convergence) from either Quantum ESPRESSO or VASP and generates feedback for the next iteration.

Verification Checkpoints
^^^^^^^^^^^^^^^^^^^^^^^^^

* **Geometry Generator & Form Filler**: Deterministic checking ensures Python syntax correctness and that atoms do not overlap.
* **Review Layer Evaluation**: Parses computational logs to extract physical property metrics and generate the feedback loop.

Extending SHALOM
-----------------

To build a new workflow (e.g., defect screening, catalyst optimization):

1. Define Pydantic schemas for your domain-specific data.
2. Implement agents that wrap ``LLMProvider.generate_structured_output()`` calls.
3. Compose agents into a hierarchy with parent agents delegating to children.
4. Use ``SafeExecutor`` for any LLM-generated code that needs sandboxed execution.
5. Connect to HPC via the MCP tool interface if DFT calculations are required.

The material discovery pipeline in ``shalom/agents/`` serves as a reference implementation.
