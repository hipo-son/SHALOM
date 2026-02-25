Quickstart
==========

This page demonstrates the main ways to use SHALOM, from simple CLI commands
to the full LLM-driven autonomous pipeline.

CLI: Generate DFT Input Files
------------------------------

The simplest use case requires **no API keys** at all:

.. code-block:: bash

   # Generate VASP input files for Silicon
   python -m shalom run Si --backend vasp

   # Generate QE input files
   python -m shalom run Si --backend qe --calc scf

   # Use a local structure file
   python -m shalom run --structure POSCAR --backend vasp

   # Override DFT parameters
   python -m shalom run Si --backend vasp --set ENCUT=600
   python -m shalom run Si --backend qe --set ecutwfc=80

Each command creates a folder with DFT input files and a README explaining
what was generated.

CLI: 5-Step QE Workflow
------------------------

Run a complete Quantum ESPRESSO workflow (requires ``pw.x`` and pseudopotentials):

.. code-block:: bash

   # Full: vc-relax -> SCF -> bands -> NSCF -> DOS
   python -m shalom workflow Si -o ./si_workflow -np 4

   # Skip relaxation (start from SCF)
   python -m shalom workflow Si --skip-relax -np 4

This produces band structure and DOS plots automatically.

CLI: Convergence Tests
-----------------------

Test energy cutoff and k-point convergence:

.. code-block:: bash

   # Cutoff convergence (run this first)
   python -m shalom converge Si --test cutoff --values 30,40,50,60,80 -np 2

   # K-point convergence (use the converged cutoff)
   python -m shalom converge Si --test kpoints --values 20,30,40,50 --ecutwfc 60

CLI: Band/DOS Plotting
-----------------------

Plot results from completed QE calculations:

.. code-block:: bash

   python -m shalom plot ./03_bands --bands
   python -m shalom plot ./04_nscf --dos
   python -m shalom plot ./03_bands --bands --fermi-from ./04_nscf --emin -8 --emax 6

CLI: Post-DFT Analysis
-----------------------

Six analysis modules for computed results (some require optional dependencies):

.. code-block:: bash

   # Crystal symmetry (requires: pip install shalom[symmetry])
   python -m shalom analyze symmetry --structure POSCAR

   # Powder XRD pattern (requires: pip install shalom[analysis])
   python -m shalom analyze xrd --structure POSCAR -o xrd.png

   # Electronic band structure (numpy only — no extra deps)
   python -m shalom analyze electronic --calc-dir ./03_bands

   # Magnetic properties from QE output (no extra deps)
   python -m shalom analyze magnetic --pw-out pw.out

   # Elastic tensor analysis (requires: pip install shalom[analysis])
   python -m shalom analyze elastic --file elastic_tensor.json

   # Phonon properties (requires: pip install shalom[phonon])
   python -m shalom analyze phonon --structure POSCAR --supercell 2x2x2 --force-constants fc.hdf5

Python API: LLM-Driven Pipeline
---------------------------------

For the autonomous material discovery pipeline, you need an LLM API key:

.. code-block:: python

    from shalom.core.llm_provider import LLMProvider
    from shalom.agents.design_layer import CoarseSelector, FineSelector

    # Initialize the provider with your preferred LLM
    llm = LLMProvider(provider_type="openai", model_name="gpt-4o")

    # Or use a local LLM (no API key needed):
    # llm = LLMProvider(
    #     provider_type="openai", model_name="llama3",
    #     base_url="http://localhost:11434/v1"
    # )

    objective = "Find a stable 2D material with bandgap > 1.0eV"

    # Step 1: Coarse Selection
    coarse = CoarseSelector(llm)
    candidates = coarse.select(objective)

    # Step 2: Fine Selection & Ranking
    fine = FineSelector(llm)
    winner = fine.rank_and_select(objective, candidates)

    print(f"Top Material: {winner.candidate.material_name}")
    print(f"Score: {winner.score}")

CLI Pipeline (One Command):

.. code-block:: bash

   # Full autonomous pipeline
   python -m shalom pipeline "Find a 2D HER catalyst"

   # Use Claude instead of OpenAI
   python -m shalom pipeline "Stable cathode" --provider anthropic

   # Skip design layer, go straight to simulation
   python -m shalom pipeline "MoS2 band structure" --material MoS2

   # Use a local LLM (no API key needed)
   python -m shalom pipeline "Find HER catalyst" --base-url http://localhost:11434/v1

MCP Server (Claude Code Integration)
--------------------------------------

SHALOM can run as an MCP server, letting Claude Code call DFT tools via
natural language:

.. code-block:: bash

   # Install MCP support
   pip install "shalom[mcp]"

   # Register in Claude Code (one-time)
   claude mcp add shalom -- python -m shalom.mcp_server

   # Or use the project-scoped .mcp.json (already in repo root)

After setup, tell Claude Code things like:

- "Generate QE SCF input for Silicon"
- "Run the full workflow for mp-1040425"
- "Plot the band structure in ./03_bands"
- "Analyze symmetry of this structure"

**16 MCP tools**: ``search_material``, ``generate_dft_input``, ``run_workflow``,
``execute_dft``, ``parse_dft_output``, ``plot_bands``, ``plot_dos``,
``run_convergence``, ``check_qe_setup``, ``run_pipeline``,
``analyze_elastic``, ``analyze_phonon_properties``,
``analyze_electronic_structure``, ``analyze_xrd_pattern``,
``analyze_symmetry_properties``, ``analyze_magnetic_properties``
