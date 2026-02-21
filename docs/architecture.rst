Architecture
============

The SHALOM framework relies on a closed-loop autonomous feedback mechanism to execute robust material discoveries.
At its core, the simulated materials (as ASE `Atoms` objects) undergo geometric verification and VASP simulation testing.

Closed-Loop Feedback Loop
-------------------------

The following Mermaid diagram visualizes the autonomous self-correction cycle spanning across the three major agents:

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
        POSCAR[("POSCAR File")]

        VASP["VASP Simulation\n(HPC/Local)"]
        OUTCAR[("OUTCAR File")]

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
        SL_FF -- Valid --> POSCAR
        
        %% HPC
        POSCAR --> VASP
        VASP --> OUTCAR
        
        %% Review & Feedback Loop
        OUTCAR --> RL
        RL -- Parsing --> RES
        
        %% Closed-loop backward arrow
        RES -- Fail --> FB
        FB --> DL_CS : Inject Reason as Guidance
        
        RES -- Success --> END((Success!))

Verification Checkpoints
------------------------
* **Geometry Generator & Form Filler**: Deterministic checking to ensure that Python syntax behaves normally, and Atoms do not overlap.
* **ReviewLayer Evaluation**: Evaluates physical property achievement parsing computational logs. Extracts parameters to generate the feedback loop.
