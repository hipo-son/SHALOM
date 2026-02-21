# SHALOM Master Design Document

This project is a general-purpose multi-agent materials discovery framework combining a highly extensible Python library with an MCP (Model Context Protocol)-based architecture.

## 1. System Architecture

The system consists of three independent layers with clearly separated roles:

- **Design Layer**: A group of upper-level agents that decide which material to simulate next based on a natural language objective.
- **Simulation Layer**: Converts the selected material into actual DFT input files and validates structural integrity before simulation.
- **Review Layer**: Analyzes simulation results (e.g., VASP OUTCAR) to determine whether the target objective was achieved and closes the feedback loop.

## 2. Design Layer: Triage-Ranking Logic

To prevent random exploration and reduce computational cost, a triage-ranking architecture is adopted:

- **Coarse Selector**: Screens 3-5 promising candidates from the vast chemical space (leveraging periodic table trends, d-band center theory, etc.).
- **Fine Selector**: Ranks candidates within the small pool based on expected properties and selects exactly one material closest to the target.

## 3. Simulation Layer: Pre-Validation and Automation

A three-step self-correction loop validates the physical soundness of POSCAR files before submitting jobs to HPC resources (Slurm):

- **Geometry Generator**: Writes ASE-based Python scripts from natural language requirements to produce initial structures.
- **Form Filler**: Analyzes the generated structure, evaluating layer count, vacuum thickness, atomic overlap, etc., using a standardized form.
- **Geometry Reviewer**: Makes pass/fail decisions based on the evaluation and issues correction instructions (up to 3-5 retries).

## 4. MCP-Based HPC Integration Module

- **VASP-Slurm MCP Server**: Ensures security by allowing the LLM to access HPC only through predefined tools (`submit_slurm_job`, `check_job_status`, `read_vasp_outcar`, etc.).
- **Self-Correction Algorithm**: When SCF fails, the Review Agent parses logs, adjusts INCAR parameters, and resubmits.

## 5. Development Milestones

- **Phase 1: Core Library & MCP Environment Setup** — Python library structure, ASE-based Geometry Generator prompts, and Form Filler setup.
- **Phase 2: Triage-Ranking Agent Loop** — Coarse/Fine Selector prompt pipeline construction.
- **Phase 3: HPC VASP Integration & Self-Correction** — Slurm integration, end-to-end testing with bulk materials.
- **Phase 4: Advanced Systems & Open-Source Release** — Extension to 2D/TMD systems, search performance metrics tooling.

---

## Research Objective

A multi-agent orchestration framework for autonomous first-principles materials discovery that can plan/execute workflows, manage resources, evaluate candidates, and adapt search strategies.

## System Architecture Roles

- **Planner Agent**: Defines objectives, allocates budget, manages strategies.
- **Executor Agent**: Submits Slurm jobs, monitors execution, handles failures.
- **Evaluator Agent**: Parses results, computes metrics, ranks candidates.
- **Critic/Auditor Agent**: Validates tool usage, ensures safety/reproducibility.

## Software Architecture

Library-Centric Design implemented in Python for reproducibility and HPC integration. Includes Agent Framework, Tool System, Backend Layer (Slurm), and Provider Interface (LLM APIs).
