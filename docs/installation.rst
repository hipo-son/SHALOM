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

Step 1 — Clone the Repository
------------------------------

Clone into a convenient location such as your Desktop or a projects folder:

.. code-block:: bash

   cd ~/Desktop          # or: cd ~/projects
   git clone https://github.com/hipo-son/SHALOM.git
   cd SHALOM

Step 2 — Create a Python Environment
--------------------------------------

**Option A: conda (recommended)**

Best for managing scientific dependencies (pymatgen, ASE):

.. code-block:: bash

   conda env create -f environment.yml
   conda activate shalom-env

**Option B: venv**

If you do not have conda installed:

.. code-block:: bash

   python -m venv .venv
   source .venv/bin/activate        # Windows: .venv\Scripts\activate
   pip install -e ".[dev,mp]"

Verify the installation:

.. code-block:: bash

   python -m shalom --help

Step 3 — Configure API Keys
-----------------------------

Copy the provided template and edit it with your keys:

.. code-block:: bash

   cp .env.example .env
   # Open .env in a text editor, fill in your keys, then:
   source .env

.. list-table:: Required / Optional API Keys
   :header-rows: 1
   :widths: 30 40 30

   * - Variable
     - Purpose
     - Where to Get
   * - ``OPENAI_API_KEY``
     - LLM agents (Design / Review layer)
     - https://platform.openai.com/api-keys
   * - ``ANTHROPIC_API_KEY``
     - Alternative to OpenAI
     - https://console.anthropic.com/
   * - ``MP_API_KEY``
     - Fetch structures by MP ID or formula
     - https://next-gen.materialsproject.org/api (free)
   * - ``SHALOM_PSEUDO_DIR``
     - QE pseudopotential directory
     - Set after running ``setup-qe --download``

Only **one** of ``OPENAI_API_KEY`` / ``ANTHROPIC_API_KEY`` is required.
``MP_API_KEY`` is needed only for structure lookup via MP ID or chemical formula.

Step 4 — Quantum ESPRESSO Setup (QE backend only)
---------------------------------------------------

Skip this step if you only use the VASP backend.

.. code-block:: bash

   # Ubuntu / Debian (or WSL2 on Windows)
   sudo apt install quantum-espresso

   # conda
   conda install -c conda-forge qe

.. note::

   **Windows users**: ``pw.x`` cannot run on native Windows Python.
   You must use WSL2 (Windows Subsystem for Linux):

   .. code-block:: bash

      wsl -d Ubuntu-22.04
      sudo apt install quantum-espresso

Download SSSP pseudopotentials for your elements:

.. code-block:: bash

   python -m shalom setup-qe --elements Si,Fe,O --download

   # Check overall QE environment
   python -m shalom setup-qe

Step 5 — First Run
-------------------

Choose the path matching your available resources:

.. code-block:: bash

   # Path A — local structure file (no API keys needed)
   python -m shalom run --structure POSCAR --backend vasp

   # Path B — Materials Project ID (MP_API_KEY required)
   python -m shalom run mp-19717 --backend vasp

   # Path C — chemical formula + QE (MP_API_KEY + QE install required)
   python -m shalom run Si --backend qe --calc scf

Each run creates an output folder (e.g., ``Si_qe_static/``) containing DFT input
files and a ``README.md`` explaining what was generated and the next steps.

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

Contact your HPC administrator to confirm VASP availability:

.. code-block:: bash

   module load vasp/6.4.1

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
