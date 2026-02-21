SHALOM: System of Hierarchical Agents for Logical Orchestration of Materials
=============================================================================

**SHALOM** is a general-purpose hierarchical agent orchestration framework for
computational materials science. It provides the core infrastructure — agent
lifecycle management, structured LLM communication, sandboxed code execution,
HPC integration, and closed-loop feedback — needed to compose autonomous
multi-agent workflows for arbitrary materials science tasks.

The framework is **domain-aware but task-agnostic**: agents can be assembled
into configurable hierarchies to tackle material screening, structure
optimization, property prediction, or any workflow expressible as a sequence
of LLM-driven decisions and computational validations.

To validate the framework, SHALOM ships with a complete
**autonomous material discovery pipeline** as its first proof-of-concept use
case, demonstrating the full agent lifecycle from hypothesis generation through
DFT simulation to result evaluation.

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   installation
   quickstart

.. toctree::
   :maxdepth: 2
   :caption: Reference

   architecture
   api_reference

.. toctree::
   :maxdepth: 1
   :caption: Design Documents

   master_design_document
