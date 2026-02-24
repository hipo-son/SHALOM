Installation
============

.. note::

   SHALOM is not yet published to PyPI.
   Install from source by cloning the GitHub repository (see steps below).

Requirements
------------

- Python 3.11 recommended (3.9+ supported)
- conda **or** Python venv
- Git

Step 1 -- Clone the Repository
-------------------------------

Clone into a convenient location such as your Desktop or a projects folder:

.. code-block:: bash

   cd ~/Desktop          # or: cd ~/projects
   git clone https://github.com/hipo-son/SHALOM.git
   cd SHALOM

Step 2 -- Create a Python Environment
---------------------------------------

**Option A: conda (recommended)**

Best for managing scientific dependencies (pymatgen, ASE):

.. code-block:: bash

   conda env create -f environment.yml
   conda activate shalom-env

**Option B: venv**

If you do not have conda installed:

.. code-block:: bash

   python -m venv .venv

Activate the environment:

.. code-block:: bash

   # Linux / macOS
   source .venv/bin/activate

   # Windows (Command Prompt)
   .venv\Scripts\activate

   # Windows (PowerShell)
   .venv\Scripts\Activate.ps1

Then install:

.. code-block:: bash

   pip install -e ".[dev,mp]"

   # Optional: band/DOS plotting
   pip install -e ".[plotting]"

   # Optional: everything (plotting + MP + MCP)
   pip install -e ".[all]"

Verify the installation:

.. code-block:: bash

   python -m shalom --help

Step 3 -- Run Your First Calculation (No API Keys Needed)
----------------------------------------------------------

SHALOM can generate DFT input files using built-in ASE structures.
**No API keys are required** for this step:

.. code-block:: bash

   # Generate VASP input files for bulk Silicon
   python -m shalom run Si --backend vasp

   # Generate QE input files for bulk Silicon
   python -m shalom run Si --backend qe --calc scf

   # Use a local structure file (CIF, POSCAR, etc.)
   python -m shalom run --structure my_structure.cif --backend vasp

Each command creates an output folder (e.g., ``~/Desktop/shalom-runs/Si_vasp_relaxation/``)
with DFT input files and a ``README.md`` explaining what was generated.

.. note::

   Without ``MP_API_KEY``, SHALOM falls back to ASE's built-in bulk structures
   for simple elements (Si, Fe, Cu, etc.). For complex compounds or Materials
   Project structures, configure the API key in Step 4.

Step 4 -- Configure API Keys (Optional)
-----------------------------------------

API keys unlock additional features but are **not required** for basic usage.

.. code-block:: bash

   cp .env.example .env
   # Edit .env with your keys, then:
   source .env    # Linux/macOS
   # Windows: set variables in System Environment Variables

.. list-table:: API Keys
   :header-rows: 1
   :widths: 25 35 15 25

   * - Variable
     - Purpose
     - Required?
     - Where to Get
   * - ``MP_API_KEY``
     - Fetch structures by MP ID or formula
     - Optional (free)
     - `Materials Project <https://next-gen.materialsproject.org/api>`_
   * - ``OPENAI_API_KEY``
     - LLM agents (Design / Review layer)
     - ``pipeline`` only
     - `OpenAI <https://platform.openai.com/api-keys>`_
   * - ``ANTHROPIC_API_KEY``
     - Alternative to OpenAI
     - ``pipeline`` only
     - `Anthropic <https://console.anthropic.com/>`_
   * - ``SHALOM_PSEUDO_DIR``
     - QE pseudopotential directory
     - QE execution only
     - Set after ``setup-qe --download``
   * - ``SHALOM_LLM_BASE_URL``
     - Local LLM server URL
     - Optional
     - Ollama, vLLM, llama.cpp

With ``MP_API_KEY``, you can fetch real structures:

.. code-block:: bash

   python -m shalom run mp-19717 --backend vasp    # Silicon from Materials Project
   python -m shalom run Fe2O3 --backend qe          # Iron oxide

Step 5 -- Set Up Quantum ESPRESSO (QE Execution Only)
------------------------------------------------------

.. note::

   Skip this step if you only generate input files or use the VASP backend.
   File generation works on all platforms; **execution** requires ``pw.x``.

.. code-block:: bash

   # Linux (Ubuntu/Debian)
   sudo apt install quantum-espresso

   # conda (cross-platform, but not native Windows)
   conda install -c conda-forge qe

   # Windows: pw.x requires WSL2
   # Run: python -m shalom setup-qe  (will show Windows-specific instructions)

Download SSSP pseudopotentials:

.. code-block:: bash

   python -m shalom setup-qe --elements Si,Fe,O --download
   python -m shalom setup-qe    # Check environment

Run a calculation end-to-end:

.. code-block:: bash

   python -m shalom run Si --backend qe --calc scf --execute -np 4

DFT Solver Notes
-----------------

Quantum ESPRESSO (open-source)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`Quantum ESPRESSO <https://www.quantum-espresso.org/>`_ is open-source (GPL)
and suitable for personal workstations and small clusters.
SHALOM communicates with QE through ``pw.x`` input files and XML output.

VASP (commercial / HPC)
^^^^^^^^^^^^^^^^^^^^^^^^

`VASP <https://www.vasp.at/>`_ requires a commercial license and is typically
available on institutional HPC clusters.
SHALOM generates POSCAR / INCAR / KPOINTS input files and parses OUTCAR output.

.. note::

   SHALOM abstracts DFT-specific I/O behind a unified interface.
   Agents operate on the same schema regardless of backend,
   so switching from QE to VASP requires no changes to agent logic.

Known Issues
------------

- **VASP OUTCAR parsing**: 19 tests fail with ``pymatgen>=2025.10`` due to an
  upstream ``IndexError`` in ``Outcar.__init__``. The provided ``environment.yml``
  pins ``pymatgen<2025.10`` to avoid this. QE, agent, and CLI tests are unaffected.
- **QE on Windows (native)**: ``pw.x`` execution requires WSL2.
  DFT input file generation works on native Windows; running ``--execute`` does not.

Docker / HPC
------------

For containerized HPC deployments:

.. code-block:: bash

   docker pull ghcr.io/hipo-son/shalom:latest

   # Run CLI commands
   docker run --rm ghcr.io/hipo-son/shalom:latest run Si --backend vasp

   # Run MCP server (for Claude Code integration)
   docker run --rm -i ghcr.io/hipo-son/shalom:latest mcp-server
