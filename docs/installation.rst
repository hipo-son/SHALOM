Installation
============

SHALOM requires **Python 3.9** or later.

Requirements
------------

- Python >= 3.9
- ASE (Atomic Simulation Environment)
- OpenAI and/or Anthropic Python SDKs
- Pydantic v2

Basic Installation
------------------

.. code-block:: bash

    pip install shalom

Full Installation
-----------------

Install all optional dependencies (pymatgen, development tools, documentation):

.. code-block:: bash

    pip install shalom[all]

Development Installation
------------------------

For local development and testing:

.. code-block:: bash

    git clone https://github.com/hipo-son/SHALOM.git
    cd SHALOM
    pip install -e ".[dev]"

Docker (HPC / SLURM)
---------------------

For containerized deployments on HPC clusters:

.. code-block:: bash

    docker pull ghcr.io/hipo-son/shalom:latest

DFT Solver Setup
-----------------

SHALOM supports two DFT backends. You only need the one matching your environment.

Quantum ESPRESSO (personal / open-source)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`Quantum ESPRESSO <https://www.quantum-espresso.org/>`_ is open-source (GPL) and
suitable for personal workstations and small clusters.

.. code-block:: bash

    # Ubuntu / Debian
    sudo apt install quantum-espresso

    # or build from source
    # see https://www.quantum-espresso.org/download/

SHALOM communicates with QE through ``pw.x`` input files and parses XML output.
No license is required.

VASP (HPC / licensed)
^^^^^^^^^^^^^^^^^^^^^^

`VASP <https://www.vasp.at/>`_ requires a commercial license and is typically
available on institutional HPC clusters. SHALOM communicates with VASP through
POSCAR/INCAR/KPOINTS input files and parses OUTCAR output.

Contact your HPC administrator to confirm VASP availability and module loading:

.. code-block:: bash

    # Typical HPC module usage
    module load vasp/6.4.1

.. note::

    SHALOM abstracts DFT-specific I/O behind a unified interface.
    Agents operate on the same schema regardless of whether the backend is
    Quantum ESPRESSO or VASP, so switching solvers requires no changes to
    agent logic.
