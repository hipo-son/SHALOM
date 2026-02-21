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
