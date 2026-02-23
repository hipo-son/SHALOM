"""Built-in fallback defaults for prompts and configs.

These are identical copies of the external .md/.yaml files.
They ensure SHALOM works even without the data files installed.

WARNING: When editing prompts or configs, update BOTH the external file
AND this fallback module to keep them in sync. The test suite verifies
that external files and these defaults produce identical values.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Prompt Defaults (mirrors shalom/prompts/*.md)
# ---------------------------------------------------------------------------

PROMPTS: dict[str, str] = {
    "coarse_selector": (
        "[v1.0.0]\n"
        "        You are the \"Coarse Selector\", working as a world-class computational materials scientist.\n"
        "        Select the 3 to 5 most promising material candidates to achieve the given\n"
        "        natural language Target Objective.\n"
        "\n"
        "        [Selection Guidelines]\n"
        "        1. Use established physical/chemical intuition such as periodic table trends,\n"
        "           electronegativity, and d-band center theory.\n"
        "        2. Each selection must have clear scientific reasoning — no random picks.\n"
        "        3. Candidates should include some diversity (alloys, doping, etc.)\n"
        "           rather than being limited to a single obvious material.\n"
        "        4. You MUST respond in JSON format as a Candidates list."
    ),
    "fine_selector": (
        "[v1.0.0]\n"
        "        You are the \"Fine Selector\".\n"
        "        You are given a small pool of material candidates that passed the coarse screening.\n"
        "        Your task is to precisely evaluate each candidate against the Target Objective\n"
        "        on a scale of 0.0 to 1.0, and select exactly ONE \"Winner\".\n"
        "\n"
        "        [Evaluation Guidelines]\n"
        "        1. Deeply analyze each candidate's 'reasoning' and 'expected_properties'.\n"
        "        2. Consider the simulation (DFT) cost vs. success probability\n"
        "           (overly complex or large cells are penalized).\n"
        "        3. The candidate with the highest score advances to the Simulation Layer."
    ),
    "geometry_generator": (
        "[v1.0.0]\n"
        "        You are the \"Geometry Generator\".\n"
        "        Read the given material (Winner Candidate) information and the user's simulation objective,\n"
        "        then write Python code using the ASE (Atomic Simulation Environment) library\n"
        "        to generate the corresponding physical structure.\n"
        "\n"
        "        [Instructions]\n"
        "        1. Use `from ase.build import bulk, surface` and similar ASE utilities.\n"
        "        2. The final Atoms object MUST be assigned to a variable named `atoms`.\n"
        "        3. The returned `python_code` must contain logic like `atoms = bulk(...)`.\n"
        "        4. (Important) Do NOT include markdown code fences (```python). Return pure Python code only."
    ),
    "review_agent": (
        "[v2.0.0]\n"
        "        You are the \"Review Agent\" — a computational materials science expert.\n"
        "        You evaluate DFT simulation results against the user's Target Objective.\n"
        "\n"
        "        [Evaluation Guidelines]\n"
        "        1. Check convergence: both electronic (SCF) and ionic loops.\n"
        "        2. Verify physical reasonableness: energy per atom, max force, bandgap.\n"
        "        3. Assess alignment with target objective properties.\n"
        "        4. If entropy T*S > 1 meV/atom, recommend reducing SIGMA.\n"
        "        5. If GGA+U was used, note that Wang et al. U values are empirical.\n"
        "           For strongly-correlated systems, recommend r2SCAN or linear-response U.\n"
        "        6. If ISIF=4 was used for 2D materials, note the volume-conservation\n"
        "           limitation for extreme in-plane deformations.\n"
        "        7. Review error correction history: flag if BRMIX occurred (charge sloshing).\n"
        "        8. In feedback_for_design, provide specific INCAR parameter suggestions\n"
        "           when possible (e.g., \"increase ENCUT to 600\", \"try ALGO=Damped\")."
    ),
    "eval_confidence_rule": (
        "[CONFIDENCE RULE] Set confidence > 0.5 ONLY if you can cite specific "
        "evidence from crystallographic databases (ICSD, Materials Project, AFLOW) "
        "or published literature. If your assessment is based on chemical intuition "
        "without concrete references, confidence MUST be <= 0.5."
    ),
    "eval_stability": (
        '[v1.0.0] You are the "Stability Evaluator".\n'
        "Score each candidate material on thermodynamic and kinetic stability (0.0 to 1.0).\n"
        "\n"
        "[Evaluation Criteria]\n"
        "1. Evaluate using formation energy trends, convex hull distance, and known phase diagrams.\n"
        "2. Reference Hume-Rothery rules for alloys (atomic size, electronegativity, valence electron count).\n"
        "3. If evaluating as metastable (when explicitly noted), identify the required stabilization\n"
        "   mechanism (e.g., epitaxial substrate matching, high-pressure synthesis phase, kinetic\n"
        "   trapping via rapid quenching).\n"
        "4. Consider decomposition pathways and competing phases."
    ),
    "eval_target_property": (
        '[v1.0.0] You are the "Target Property Evaluator".\n'
        "Score each candidate on alignment with the target objective (0.0 to 1.0).\n"
        "\n"
        "[Evaluation Criteria]\n"
        "1. First, identify the dominant physical descriptors for the given target objective:\n"
        "   - Catalysis: d-band center, adsorption energy, surface reactivity\n"
        "   - Quantum materials: band topology, flat bands, spin-orbit coupling\n"
        "   - Battery cathodes: ion radius, diffusion activation barrier, voltage\n"
        "   - Thermoelectrics: Seebeck coefficient, lattice thermal conductivity, power factor\n"
        "2. Then score each candidate based on alignment with these descriptors.\n"
        "3. Use known structure-property relationships and electronegativity differences."
    ),
    "eval_dft_feasibility": (
        '[v1.0.0] You are the "DFT Feasibility Evaluator".\n'
        "Score each candidate on computational cost and convergence difficulty (0.0 to 1.0).\n"
        "\n"
        "[Evaluation Criteria]\n"
        "1. Number of atoms in the primitive cell (fewer = cheaper).\n"
        "2. Magnetic ordering complexity (AFM, spin-frustrated systems are harder).\n"
        "3. Electron correlation: does it need DFT+U or hybrid functionals (HSE06)?\n"
        "4. Expected SCF convergence difficulty (metallic surfaces, charge sloshing).\n"
        "5. A simple bulk crystal scores high; a large supercell with defects scores low."
    ),
    "eval_synthesizability": (
        '[v1.0.0] You are the "Synthesizability Evaluator".\n'
        "Score each candidate on experimental synthesis feasibility (0.0 to 1.0).\n"
        "\n"
        "[Evaluation Criteria]\n"
        "1. Competing phase analysis and synthesis energy barriers.\n"
        "2. Availability of known precursor reaction pathways.\n"
        "3. Goldschmidt tolerance factor for perovskites, Hume-Rothery rules for alloys.\n"
        "4. Do NOT penalize solely for lack of prior experimental reports — assess the\n"
        "   thermodynamic/kinetic pathway feasibility instead.\n"
        "5. Consider whether similar compositions or structural motifs have been synthesized."
    ),
    "eval_novelty": (
        '[v1.0.0] You are the "Novelty Evaluator".\n'
        "Score each candidate on scientific novelty and originality (0.0 to 1.0).\n"
        "\n"
        "[Evaluation Criteria]\n"
        "1. Is this a trivial elemental substitution of a known material (low novelty)?\n"
        "2. Does it introduce genuinely new structural motifs, compositions, or design principles?\n"
        "3. Would this material generate interest in top-tier journals (Nature, Science)?\n"
        "4. Consider the gap between known materials and this candidate."
    ),
    "eval_environmental_cost": (
        '[v1.0.0] You are the "Environmental & Cost Evaluator".\n'
        "Score each candidate on element availability and environmental impact (0.0 to 1.0).\n"
        "\n"
        "[Evaluation Criteria]\n"
        "1. Penalize use of rare/expensive elements: Ir, Ru, Re, Os, Rh, Pd.\n"
        "2. Penalize toxic elements: Pb, Tl, Cd, Hg, As.\n"
        "3. Penalize conflict minerals where applicable.\n"
        "4. Prefer earth-abundant alternatives (Fe, Cu, Mn, Ni, Ti, Zn, Al).\n"
        "5. A material using only abundant, non-toxic elements scores ~1.0."
    ),
}

# ---------------------------------------------------------------------------
# Config Defaults (mirrors shalom/config/*.yaml)
# ---------------------------------------------------------------------------

CONFIGS: dict = {
    "potcar_mapping": {
        "potcar_version": "54",
        "vasp_recommended": {
            # Main group
            "H": "H", "He": "He",
            "Li": "Li_sv", "Be": "Be", "B": "B", "C": "C", "N": "N", "O": "O",
            "F": "F", "Ne": "Ne",
            "Na": "Na_pv", "Mg": "Mg", "Al": "Al", "Si": "Si", "P": "P", "S": "S",
            "Cl": "Cl", "Ar": "Ar",
            "K": "K_sv", "Ca": "Ca_sv",
            # 3d transition metals
            "Sc": "Sc_sv", "Ti": "Ti_sv", "V": "V_sv", "Cr": "Cr_pv",
            "Mn": "Mn_pv", "Fe": "Fe_pv", "Co": "Co", "Ni": "Ni",
            "Cu": "Cu", "Zn": "Zn",
            # p-block row 4
            "Ga": "Ga_d", "Ge": "Ge_d", "As": "As", "Se": "Se", "Br": "Br", "Kr": "Kr",
            # 4d transition metals
            "Rb": "Rb_sv", "Sr": "Sr_sv",
            "Y": "Y_sv", "Zr": "Zr_sv", "Nb": "Nb_sv", "Mo": "Mo_sv",
            "Tc": "Tc_pv", "Ru": "Ru_pv", "Rh": "Rh_pv", "Pd": "Pd",
            "Ag": "Ag", "Cd": "Cd",
            # p-block row 5
            "In": "In_d", "Sn": "Sn_d", "Sb": "Sb", "Te": "Te", "I": "I", "Xe": "Xe",
            # 5d transition metals
            "Cs": "Cs_sv", "Ba": "Ba_sv",
            "Hf": "Hf_pv", "Ta": "Ta_pv", "W": "W_sv",
            "Re": "Re", "Os": "Os", "Ir": "Ir", "Pt": "Pt",
            "Au": "Au", "Hg": "Hg",
            # p-block row 6
            "Tl": "Tl_d", "Pb": "Pb_d", "Bi": "Bi_d",
            # Lanthanides
            "La": "La", "Ce": "Ce", "Pr": "Pr_3", "Nd": "Nd_3",
            "Sm": "Sm_3", "Eu": "Eu_2", "Gd": "Gd_3",
            "Tb": "Tb_3", "Dy": "Dy_3", "Ho": "Ho_3",
            "Er": "Er_3", "Tm": "Tm_3", "Yb": "Yb_2", "Lu": "Lu_3",
            # Actinides
            "U": "U", "Np": "Np", "Pu": "Pu",
        },
        "mp_default_overrides": {
            "Fe": "Fe_pv", "Ti": "Ti_pv", "V": "V_pv",
            "Mo": "Mo_pv", "W": "W_pv",
            "Cr": "Cr_pv", "Mn": "Mn_pv",
        },
    },
    "enmax_values": {
        "H": 250.0, "He": 479.0,
        "Li": 499.0, "Be": 309.0, "B": 318.7, "C": 400.0, "N": 400.0, "O": 400.0,
        "F": 400.0, "Ne": 344.0,
        "Na": 260.0, "Mg": 200.0, "Al": 240.3, "Si": 245.3, "P": 255.0, "S": 280.0,
        "Cl": 262.0, "Ar": 266.0,
        "K": 259.0, "Ca": 267.0,
        "Sc": 223.0, "Ti": 495.0, "V": 476.0, "Cr": 227.0,
        "Mn": 270.0, "Fe": 267.9, "Co": 268.0, "Ni": 270.0,
        "Cu": 295.4, "Zn": 277.0,
        "Ga": 135.0, "Ge": 174.0, "As": 209.0, "Se": 212.0, "Br": 213.0,
        "Rb": 220.0, "Sr": 229.0,
        "Y": 203.0, "Zr": 230.0, "Nb": 209.0, "Mo": 225.0,
        "Ru": 213.0, "Rh": 229.0, "Pd": 251.0, "Ag": 250.0, "Cd": 274.0,
        "In": 96.0, "Sn": 103.0, "Sb": 172.0, "Te": 175.0, "I": 176.0,
        "Cs": 220.0, "Ba": 238.0,
        "Hf": 220.0, "Ta": 224.0, "W": 224.0,
        "Re": 226.0, "Os": 228.0, "Ir": 211.0, "Pt": 230.0,
        "Au": 230.0,
        "La": 219.0, "Ce": 273.0, "Gd": 256.0, "U": 253.0,
    },
    "magnetic_elements": {
        "default_magmom": {
            # 3d transition metals
            "Ti": 1.0, "V": 3.0, "Cr": 4.0, "Mn": 5.0,
            "Fe": 5.0, "Co": 3.0, "Ni": 2.0, "Cu": 1.0,
            # 4d/5d (surface/nano structures may show magnetism)
            "Ru": 2.0, "Rh": 1.0, "Os": 2.0, "Ir": 1.0,
            # Lanthanides (4f electron count based)
            "Ce": 1.0, "Pr": 2.0, "Nd": 3.0, "Sm": 5.0,
            "Eu": 7.0, "Gd": 7.0, "Tb": 6.0, "Dy": 5.0,
            "Ho": 4.0, "Er": 3.0, "Tm": 2.0, "Yb": 1.0,
            # Actinides
            "U": 2.0, "Np": 3.0, "Pu": 5.0,
        },
    },
    "hubbard_u": {
        "functional": "PBE",
        "values": {
            "Fe": {"L": 2, "U": 5.3, "J": 0.0},
            "Co": {"L": 2, "U": 3.32, "J": 0.0},
            "Ni": {"L": 2, "U": 6.2, "J": 0.0},
            "Mn": {"L": 2, "U": 3.9, "J": 0.0},
            "V": {"L": 2, "U": 3.25, "J": 0.0},
            "Cr": {"L": 2, "U": 3.7, "J": 0.0},
            "Cu": {"L": 2, "U": 4.0, "J": 0.0},
            "Ti": {"L": 2, "U": 0.0, "J": 0.0},
        },
        # NOTE: Future expansion should include halides (F, Cl, Br, I)
        # and pnictides (N) for broader GGA+U coverage.
        "anion_elements": ["O", "S", "Se", "Te"],
    },
    "metallic_elements": {
        "elements": [
            "Li", "Be", "Na", "Mg", "Al", "K", "Ca",
            "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
            "Ga", "Rb", "Sr", "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd",
            "In", "Sn", "Cs", "Ba", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au",
            "La", "Ce", "Pr", "Nd", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm",
            "Yb", "Lu", "U", "Np", "Pu",
        ],
    },
    "incar_presets": {
        # Keys are "calc_type:accuracy" strings, converted to tuple keys in code.
        "relaxation:standard": {
            "ENCUT": 520, "EDIFF": 1e-5, "EDIFFG": -0.02,
            "ISMEAR": 0, "SIGMA": 0.05,
            "NSW": 99, "IBRION": 2, "ISIF": 3,
            "PREC": "Accurate", "LREAL": "Auto", "LORBIT": 11,
            "NELM": 100,
        },
        "static:standard": {
            "ENCUT": 520, "EDIFF": 1e-5,
            "ISMEAR": 0, "SIGMA": 0.05,
            "NSW": 0, "IBRION": -1,
            "PREC": "Accurate", "LREAL": "Auto", "LORBIT": 11,
            "NELM": 100,
        },
        "band_structure:standard": {
            "ENCUT": 520, "EDIFF": 1e-5,
            "ISMEAR": 0, "SIGMA": 0.05,
            "NSW": 0, "IBRION": -1, "ICHARG": 11,
            "PREC": "Accurate", "LREAL": False, "LORBIT": 11,
            "NELM": 100,
        },
        "dos:standard": {
            "ENCUT": 520, "EDIFF": 1e-5,
            "ISMEAR": -5, "SIGMA": 0.05,
            "NSW": 0, "IBRION": -1, "ICHARG": 11,
            "NEDOS": 3001,
            "PREC": "Accurate", "LREAL": False, "LORBIT": 11,
            "NELM": 100,
        },
        "elastic:standard": {
            "ENCUT": 520, "EDIFF": 1e-6, "EDIFFG": -0.01,
            "ISMEAR": 0, "SIGMA": 0.05,
            "NSW": 1, "IBRION": 6, "ISIF": 3,
            "PREC": "Accurate", "LREAL": False, "LORBIT": 11,
            "NELM": 100,
        },
        "relaxation:precise": {
            "ENCUT": 520, "EDIFF": 1e-6, "EDIFFG": -0.01,
            "ISMEAR": 0, "SIGMA": 0.05,
            "NSW": 99, "IBRION": 2, "ISIF": 3,
            "PREC": "Accurate", "LREAL": False, "LORBIT": 11,
            "NELM": 200,
        },
        "static:precise": {
            "ENCUT": 520, "EDIFF": 1e-6,
            "ISMEAR": 0, "SIGMA": 0.05,
            "NSW": 0, "IBRION": -1,
            "PREC": "Accurate", "LREAL": False, "LORBIT": 11,
            "NELM": 200,
        },
        "band_structure:precise": {
            "ENCUT": 520, "EDIFF": 1e-6,
            "ISMEAR": 0, "SIGMA": 0.05,
            "NSW": 0, "IBRION": -1, "ICHARG": 11,
            "PREC": "Accurate", "LREAL": False, "LORBIT": 11,
            "NELM": 200,
        },
        "dos:precise": {
            "ENCUT": 520, "EDIFF": 1e-6,
            "ISMEAR": -5, "SIGMA": 0.05,
            "NSW": 0, "IBRION": -1, "ICHARG": 11,
            "NEDOS": 5001,
            "PREC": "Accurate", "LREAL": False, "LORBIT": 11,
            "NELM": 200,
        },
        "elastic:precise": {
            "ENCUT": 520, "EDIFF": 1e-7, "EDIFFG": -0.005,
            "ISMEAR": 0, "SIGMA": 0.05,
            "NSW": 1, "IBRION": 6, "ISIF": 3,
            "PREC": "Accurate", "LREAL": False, "LORBIT": 11,
            "NELM": 200,
        },
    },
    "error_patterns": [
        {"pattern": "ZBRENT: fatal internal in bracketing", "type": "ZBRENT", "severity": "correctable"},
        {"pattern": "ZBRENT: fatal error in bracketing", "type": "ZBRENT", "severity": "correctable"},
        {"pattern": "BRMIX: very serious problems", "type": "BRMIX", "severity": "correctable"},
        {"pattern": "EDDDAV: sub-space matrix is not hermitian", "type": "EDDDAV", "severity": "correctable"},
        {"pattern": "NELM reached", "type": "SCF_UNCONVERGED", "severity": "correctable"},
        {"pattern": "WARNING: DENTET", "type": "DENTET", "severity": "correctable"},
        {"pattern": "POSMAP internal error", "type": "POSMAP", "severity": "correctable"},
        {"pattern": "VERY BAD NEWS! internal error in subroutine SGRCON", "type": "SGRCON", "severity": "fatal"},
        {"pattern": "PRICEL: found a more primitive cell", "type": "PRICEL", "severity": "correctable"},
        {"pattern": "Error EDDDAV: Call to ZHEGV failed", "type": "ZHEGV", "severity": "correctable"},
        {"pattern": "EDDRMM: call to GR_CGGR failed", "type": "EDDRMM", "severity": "correctable"},
    ],
    "correction_strategies": {
        "SCF_UNCONVERGED": [
            {"NELM": 200},
            {"ISMEAR": 1, "SIGMA": 0.1},
            {"ALGO": "Damped", "NELM": 200, "TIME": 0.5},
            {"AMIX": 0.05, "BMIX": 0.001, "NELM": 300},
            {"ALGO": "All", "NELM": 500, "AMIX": 0.02},
        ],
        "BRMIX": [
            {"ISMEAR": 1, "SIGMA": 0.1},
            {"AMIX": 0.1, "BMIX": 0.01},
            {"ALGO": "Damped", "AMIX": 0.05, "BMIX": 0.001},
            {"IMIX": 1, "AMIX": 0.02, "BMIX": 0.001, "AMIX_MAG": 0.1},
            {"ALGO": "All", "AMIX": 0.02, "BMIX": 0.001},
        ],
        "ZBRENT": [
            {"POTIM": 0.2},
            {"IBRION": 1, "POTIM": 0.3},
            {"IBRION": 3, "POTIM": 0.05, "SMASS": 0.5},
        ],
        "IONIC_SLOSHING": [
            {"POTIM": 0.2},
            {"IBRION": 1},
            {"IBRION": 3, "POTIM": 0.05, "SMASS": 0.5},
        ],
        "EDDDAV": [
            {"ALGO": "Normal"},
            {"ALGO": "Damped", "TIME": 0.5},
            {"ALGO": "All"},
        ],
        "ZHEGV": [
            {"ALGO": "Exact"},
            {"ALGO": "All", "NELM": 200},
        ],
        "EDDRMM": [
            {"ALGO": "Normal"},
            {"ALGO": "Damped", "TIME": 0.5},
        ],
        "DENTET": [
            {"ISMEAR": 0, "SIGMA": 0.05},
        ],
        "PRICEL": [
            {"SYMPREC": 1e-8},
        ],
        "POSMAP": [
            {"SYMPREC": 1e-6},
        ],
    },
    "evaluator_weights": {
        "weights": {
            "stability": 0.20,
            "target_property": 0.25,
            "dft_feasibility": 0.15,
            "synthesizability": 0.15,
            "novelty": 0.15,
            "environmental_cost": 0.10,
        },
        "veto_thresholds": {
            "stability": 0.3,
            "target_property": 0.2,
            "dft_feasibility": 0.2,
            "synthesizability": 0.2,
            "novelty": 0.0,
            "environmental_cost": 0.2,
        },
    },
    "sssp_metadata": {
        "library": "SSSP_efficiency_1.3.0",
        "functional": "PBE",
        "elements": {
            "H":  {"pseudo": "H.pbe-rrkjus_psl.1.0.0.UPF",      "ecutwfc": 60, "ecutrho": 480,  "z_valence": 1},
            "He": {"pseudo": "He.pbe-kjpaw_psl.1.0.0.UPF",       "ecutwfc": 50, "ecutrho": 400,  "z_valence": 2},
            "Li": {"pseudo": "li_pbe_v1.4.uspp.F.UPF",           "ecutwfc": 40, "ecutrho": 320,  "z_valence": 3},
            "Be": {"pseudo": "Be_ONCV_PBE-1.2.upf",              "ecutwfc": 50, "ecutrho": 200,  "z_valence": 4},
            "B":  {"pseudo": "B.pbe-n-kjpaw_psl.1.0.0.UPF",      "ecutwfc": 40, "ecutrho": 320,  "z_valence": 3},
            "C":  {"pseudo": "C.pbe-n-kjpaw_psl.1.0.0.UPF",      "ecutwfc": 45, "ecutrho": 360,  "z_valence": 4},
            "N":  {"pseudo": "N.pbe-n-rrkjus_psl.1.0.0.UPF",     "ecutwfc": 80, "ecutrho": 640,  "z_valence": 5},
            "O":  {"pseudo": "O.pbe-n-kjpaw_psl.1.0.0.UPF",      "ecutwfc": 50, "ecutrho": 400,  "z_valence": 6},
            "F":  {"pseudo": "F.pbe-n-rrkjus_psl.1.0.0.UPF",     "ecutwfc": 60, "ecutrho": 480,  "z_valence": 7},
            "Ne": {"pseudo": "Ne.pbe-n-kjpaw_psl.1.0.0.UPF",     "ecutwfc": 50, "ecutrho": 400,  "z_valence": 8},
            "Na": {"pseudo": "Na_pbe_v1.5.uspp.F.UPF",           "ecutwfc": 50, "ecutrho": 400,  "z_valence": 9},
            "Mg": {"pseudo": "Mg.pbe-spnl-kjpaw_psl.1.0.0.UPF",  "ecutwfc": 35, "ecutrho": 280,  "z_valence": 10},
            "Al": {"pseudo": "Al.pbe-n-kjpaw_psl.1.0.0.UPF",     "ecutwfc": 30, "ecutrho": 240,  "z_valence": 3},
            "Si": {"pseudo": "Si.pbe-n-rrkjus_psl.1.0.0.UPF",    "ecutwfc": 30, "ecutrho": 240,  "z_valence": 4},
            "P":  {"pseudo": "P.pbe-n-rrkjus_psl.1.0.0.UPF",     "ecutwfc": 30, "ecutrho": 240,  "z_valence": 5},
            "S":  {"pseudo": "S.pbe-n-rrkjus_psl.1.0.0.UPF",     "ecutwfc": 35, "ecutrho": 280,  "z_valence": 6},
            "Cl": {"pseudo": "Cl.pbe-n-rrkjus_psl.1.0.0.UPF",    "ecutwfc": 40, "ecutrho": 320,  "z_valence": 7},
            "Ar": {"pseudo": "Ar.pbe-n-rrkjus_psl.1.0.0.UPF",    "ecutwfc": 50, "ecutrho": 400,  "z_valence": 8},
            "K":  {"pseudo": "K.pbe-spn-kjpaw_psl.1.0.0.UPF",    "ecutwfc": 50, "ecutrho": 400,  "z_valence": 9},
            "Ca": {"pseudo": "Ca_pbe_v1.0.uspp.F.UPF",           "ecutwfc": 30, "ecutrho": 240,  "z_valence": 10},
            "Sc": {"pseudo": "Sc.pbe-spn-kjpaw_psl.1.0.0.UPF",   "ecutwfc": 40, "ecutrho": 320,  "z_valence": 11},
            "Ti": {"pseudo": "Ti.pbe-spn-kjpaw_psl.1.0.0.UPF",   "ecutwfc": 35, "ecutrho": 280,  "z_valence": 12},
            "V":  {"pseudo": "V.pbe-spnl-kjpaw_psl.1.0.0.UPF",   "ecutwfc": 35, "ecutrho": 280,  "z_valence": 13},
            "Cr": {"pseudo": "cr_pbe_v1.5.uspp.F.UPF",           "ecutwfc": 40, "ecutrho": 320,  "z_valence": 14},
            "Mn": {"pseudo": "Mn.pbe-spn-kjpaw_psl.0.3.1.UPF",   "ecutwfc": 65, "ecutrho": 780,  "z_valence": 15},
            "Fe": {"pseudo": "Fe.pbe-spn-kjpaw_psl.1.0.0.UPF",   "ecutwfc": 90, "ecutrho": 1080, "z_valence": 16},
            "Co": {"pseudo": "Co.pbe-spn-kjpaw_psl.0.3.1.UPF",   "ecutwfc": 50, "ecutrho": 600,  "z_valence": 17},
            "Ni": {"pseudo": "Ni.pbe-spn-kjpaw_psl.0.3.1.UPF",   "ecutwfc": 55, "ecutrho": 660,  "z_valence": 18},
            "Cu": {"pseudo": "Cu.pbe-dn-kjpaw_psl.1.0.0.UPF",    "ecutwfc": 55, "ecutrho": 440,  "z_valence": 11},
            "Zn": {"pseudo": "Zn.pbe-dn-kjpaw_psl.1.0.0.UPF",    "ecutwfc": 45, "ecutrho": 360,  "z_valence": 12},
            "Ga": {"pseudo": "Ga.pbe-dn-kjpaw_psl.1.0.0.UPF",    "ecutwfc": 55, "ecutrho": 440,  "z_valence": 13},
            "Ge": {"pseudo": "Ge.pbe-dn-kjpaw_psl.1.0.0.UPF",    "ecutwfc": 40, "ecutrho": 320,  "z_valence": 14},
            "As": {"pseudo": "As.pbe-n-rrkjus_psl.0.2.UPF",      "ecutwfc": 35, "ecutrho": 280,  "z_valence": 5},
            "Se": {"pseudo": "Se.pbe-n-rrkjus_psl.1.0.0.UPF",    "ecutwfc": 30, "ecutrho": 240,  "z_valence": 6},
            "Br": {"pseudo": "Br.pbe-n-rrkjus_psl.1.0.0.UPF",    "ecutwfc": 30, "ecutrho": 240,  "z_valence": 7},
            "Kr": {"pseudo": "Kr.pbe-n-rrkjus_psl.1.0.0.UPF",    "ecutwfc": 50, "ecutrho": 400,  "z_valence": 8},
            "Rb": {"pseudo": "Rb_ONCV_PBE-1.2.upf",              "ecutwfc": 40, "ecutrho": 160,  "z_valence": 9},
            "Sr": {"pseudo": "Sr_pbe_v1.uspp.F.UPF",             "ecutwfc": 30, "ecutrho": 240,  "z_valence": 10},
            "Y":  {"pseudo": "Y.pbe-spn-kjpaw_psl.1.0.0.UPF",    "ecutwfc": 35, "ecutrho": 280,  "z_valence": 11},
            "Zr": {"pseudo": "Zr.pbe-spn-kjpaw_psl.1.0.0.UPF",   "ecutwfc": 30, "ecutrho": 240,  "z_valence": 12},
            "Nb": {"pseudo": "Nb.pbe-spn-kjpaw_psl.1.0.0.UPF",   "ecutwfc": 40, "ecutrho": 320,  "z_valence": 13},
            "Mo": {"pseudo": "Mo_ONCV_PBE-1.2.upf",              "ecutwfc": 60, "ecutrho": 480,  "z_valence": 14},
            "Ru": {"pseudo": "Ru_ONCV_PBE-1.2.upf",              "ecutwfc": 55, "ecutrho": 220,  "z_valence": 16},
            "Rh": {"pseudo": "Rh_ONCV_PBE-1.2.upf",              "ecutwfc": 55, "ecutrho": 220,  "z_valence": 17},
            "Pd": {"pseudo": "Pd_ONCV_PBE-1.2.upf",              "ecutwfc": 45, "ecutrho": 180,  "z_valence": 18},
            "Ag": {"pseudo": "Ag_ONCV_PBE-1.2.upf",              "ecutwfc": 45, "ecutrho": 180,  "z_valence": 11},
            "Cd": {"pseudo": "Cd.pbe-dn-rrkjus_psl.1.0.0.UPF",   "ecutwfc": 50, "ecutrho": 400,  "z_valence": 12},
            "In": {"pseudo": "In.pbe-dn-rrkjus_psl.1.0.0.UPF",   "ecutwfc": 50, "ecutrho": 400,  "z_valence": 13},
            "Sn": {"pseudo": "Sn_pbe_v1.uspp.F.UPF",             "ecutwfc": 60, "ecutrho": 480,  "z_valence": 14},
            "Sb": {"pseudo": "sb_pbe_v1.4.uspp.F.UPF",           "ecutwfc": 40, "ecutrho": 320,  "z_valence": 5},
            "Te": {"pseudo": "Te.pbe-n-rrkjus_psl.1.0.0.UPF",    "ecutwfc": 30, "ecutrho": 240,  "z_valence": 6},
            "I":  {"pseudo": "I.pbe-n-rrkjus_psl.1.0.0.UPF",     "ecutwfc": 35, "ecutrho": 280,  "z_valence": 7},
            "Cs": {"pseudo": "Cs_pbe_v1.uspp.F.UPF",             "ecutwfc": 30, "ecutrho": 240,  "z_valence": 9},
            "Ba": {"pseudo": "Ba.pbe-spn-kjpaw_psl.1.0.0.UPF",   "ecutwfc": 25, "ecutrho": 200,  "z_valence": 10},
            "La": {"pseudo": "La.pbe-spfn-kjpaw_psl.1.0.0.UPF",  "ecutwfc": 40, "ecutrho": 320,  "z_valence": 11},
            "W":  {"pseudo": "W_pbe_v1.2.uspp.F.UPF",            "ecutwfc": 30, "ecutrho": 240,  "z_valence": 14},
            "Pt": {"pseudo": "Pt.pbe-spfn-rrkjus_psl.1.0.0.UPF", "ecutwfc": 35, "ecutrho": 280,  "z_valence": 18},
            "Au": {"pseudo": "Au_ONCV_PBE-1.2.upf",              "ecutwfc": 55, "ecutrho": 220,  "z_valence": 11},
            "Pb": {"pseudo": "Pb.pbe-dn-kjpaw_psl.1.0.0.UPF",    "ecutwfc": 40, "ecutrho": 320,  "z_valence": 14},
            "Bi": {"pseudo": "Bi.pbe-dn-kjpaw_psl.1.0.0.UPF",    "ecutwfc": 45, "ecutrho": 360,  "z_valence": 15},
        },
    },
    "qe_presets": {
        "scf:standard": {
            "control": {"calculation": "scf", "prefix": "shalom", "outdir": "./tmp", "tprnfor": True, "tstress": True},
            "system": {"occupations": "smearing", "smearing": "methfessel-paxton", "degauss": 0.00735},
            "electrons": {"conv_thr": 7.35e-7, "mixing_beta": 0.7},
        },
        "scf:precise": {
            "control": {"calculation": "scf", "prefix": "shalom", "outdir": "./tmp", "tprnfor": True, "tstress": True},
            "system": {"occupations": "smearing", "smearing": "methfessel-paxton", "degauss": 0.00368},
            "electrons": {"conv_thr": 7.35e-9, "mixing_beta": 0.5},
        },
        "vc-relax:standard": {
            "control": {"calculation": "vc-relax", "prefix": "shalom", "outdir": "./tmp", "tprnfor": True, "tstress": True, "forc_conv_thr": 1.0e-3, "nstep": 200},
            "system": {"occupations": "smearing", "smearing": "methfessel-paxton", "degauss": 0.00735},
            "electrons": {"conv_thr": 7.35e-8, "mixing_beta": 0.7},
            "ions": {"ion_dynamics": "bfgs"},
            "cell": {"cell_dynamics": "bfgs", "press": 0.0},
        },
        "vc-relax:precise": {
            "control": {"calculation": "vc-relax", "prefix": "shalom", "outdir": "./tmp", "tprnfor": True, "tstress": True, "forc_conv_thr": 5.0e-4, "nstep": 200},
            "system": {"occupations": "smearing", "smearing": "methfessel-paxton", "degauss": 0.00368},
            "electrons": {"conv_thr": 7.35e-10, "mixing_beta": 0.5},
            "ions": {"ion_dynamics": "bfgs"},
            "cell": {"cell_dynamics": "bfgs", "press": 0.0},
        },
        "relax:standard": {
            "control": {"calculation": "relax", "prefix": "shalom", "outdir": "./tmp", "tprnfor": True, "forc_conv_thr": 1.0e-3, "nstep": 100},
            "system": {"occupations": "smearing", "smearing": "methfessel-paxton", "degauss": 0.00735},
            "electrons": {"conv_thr": 7.35e-8, "mixing_beta": 0.7},
            "ions": {"ion_dynamics": "bfgs"},
        },
        "relax:precise": {
            "control": {"calculation": "relax", "prefix": "shalom", "outdir": "./tmp", "tprnfor": True, "forc_conv_thr": 5.0e-4, "nstep": 100},
            "system": {"occupations": "smearing", "smearing": "methfessel-paxton", "degauss": 0.00368},
            "electrons": {"conv_thr": 7.35e-10, "mixing_beta": 0.5},
            "ions": {"ion_dynamics": "bfgs"},
        },
        "bands:standard": {
            "control": {"calculation": "bands", "prefix": "shalom", "outdir": "./tmp", "verbosity": "high"},
            "system": {"nbnd": 20},
            "electrons": {"conv_thr": 7.35e-7},
        },
        "bands:precise": {
            "control": {"calculation": "bands", "prefix": "shalom", "outdir": "./tmp", "verbosity": "high"},
            "system": {"nbnd": 30},
            "electrons": {"conv_thr": 7.35e-9},
        },
        "nscf:standard": {
            "control": {"calculation": "nscf", "prefix": "shalom", "outdir": "./tmp"},
            "system": {"occupations": "tetrahedra"},
            "electrons": {"conv_thr": 7.35e-7},
        },
        "nscf:precise": {
            "control": {"calculation": "nscf", "prefix": "shalom", "outdir": "./tmp"},
            "system": {"occupations": "tetrahedra"},
            "electrons": {"conv_thr": 7.35e-9},
        },
    },
    "qe_error_patterns": [
        {"pattern": "convergence NOT achieved", "type": "QE_SCF_UNCONVERGED", "severity": "correctable"},
        {"pattern": "S matrix not positive definite", "type": "QE_S_MATRIX", "severity": "correctable"},
        {"pattern": "bfgs failed after", "type": "QE_BFGS_FAILED", "severity": "correctable"},
        {"pattern": "The maximum number of steps has been reached.", "type": "QE_IONIC_NOT_CONVERGED", "severity": "correctable"},
        {"pattern": "Error in routine cdiaghg", "type": "QE_DIAG_FAILED", "severity": "correctable"},
        {"pattern": "Error in routine rdiaghg", "type": "QE_DIAG_FAILED", "severity": "correctable"},
        {"pattern": "eigenvalues not converged", "type": "QE_EIGVAL_NOT_CONVERGED", "severity": "correctable"},
        {"pattern": "too many bands are not converged", "type": "QE_TOO_MANY_BANDS", "severity": "correctable"},
        {"pattern": "negative or imaginary charge", "type": "QE_NEGATIVE_CHARGE", "severity": "correctable"},
        {"pattern": "charge is wrong", "type": "QE_CHARGE_WRONG", "severity": "correctable"},
        {"pattern": "angle between cell vectors is becoming too small", "type": "QE_CELL_DISTORTED", "severity": "correctable"},
        {"pattern": "Error in routine readpp", "type": "QE_PSEUDO_NOT_FOUND", "severity": "fatal"},
        {"pattern": "Not enough space allocated", "type": "QE_OUT_OF_MEMORY", "severity": "fatal"},
        {"pattern": "Error in routine davcio", "type": "QE_IO_ERROR", "severity": "fatal"},
    ],
    "qe_correction_strategies": {
        "QE_SCF_UNCONVERGED": [
            {"electrons.mixing_beta": 0.3, "electrons.electron_maxstep": 150},
            {"electrons.mixing_mode": "local-TF", "electrons.mixing_beta": 0.3, "_electron_maxstep_cap": 50},
            {"electrons.diagonalization": "cg", "electrons.mixing_beta": 0.2, "electrons.electron_maxstep": 200},
            {"electrons.mixing_beta": 0.1, "electrons.mixing_ndim": 12, "electrons.electron_maxstep": 300},
            {"electrons.diagonalization": "cg", "electrons.mixing_beta": 0.05, "electrons.mixing_ndim": 16, "electrons.electron_maxstep": 500},
        ],
        "QE_DIAG_FAILED": [
            {"electrons.diagonalization": "cg"},
            {"electrons.diagonalization": "cg", "electrons.diago_thr_init": 1.0e-2},
            {"electrons.diagonalization": "cg", "electrons.mixing_beta": 0.3},
        ],
        "QE_S_MATRIX": [
            {"ions.trust_radius_max": 0.3, "ions.trust_radius_ini": 0.2, "_rollback_geometry": True},
            {"ions.ion_dynamics": "damp", "ions.pot_extrapolation": "second_order", "_rollback_geometry": True, "_needs_atoms": True},
            {"electrons.diagonalization": "cg"},
            {"electrons.diago_david_ndim": 4},
        ],
        "QE_BFGS_FAILED": [
            {"control.forc_conv_thr": 5.0e-3, "control.nstep": 300},
            {"ions.ion_dynamics": "damp", "ions.pot_extrapolation": "second_order", "_needs_atoms": True},
            {"control.forc_conv_thr": 1.0e-2, "control.nstep": 400, "_quality_warning": "loosely_relaxed"},
        ],
        "QE_IONIC_NOT_CONVERGED": [
            {"control.nstep": 400},
            {"control.forc_conv_thr": 2.0e-3, "control.nstep": 500},
            {"ions.ion_dynamics": "damp", "_needs_atoms": True},
        ],
        "QE_CELL_DISTORTED": [
            {"cell.cell_factor": 2.0},
            {"cell.cell_factor": 3.0, "cell.cell_dofree": "shape"},
            {"control.calculation": "relax", "_quality_warning": "vc_relax_downgraded"},
        ],
        "QE_EIGVAL_NOT_CONVERGED": [
            {"electrons.diago_david_ndim": 8},
            {"electrons.diagonalization": "cg"},
        ],
        "QE_NEGATIVE_CHARGE": [
            {"electrons.mixing_beta": 0.3},
            {"electrons.mixing_beta": 0.1, "electrons.mixing_mode": "local-TF"},
            {"electrons.mixing_beta": 0.05, "electrons.mixing_ndim": 4},
        ],
        "QE_TOO_MANY_BANDS": [
            {"electrons.diago_david_ndim": 8},
            {"electrons.diagonalization": "cg"},
        ],
        "QE_CHARGE_WRONG": [
            {"electrons.mixing_beta": 0.3},
            {"system.ecutrho": 800, "electrons.mixing_beta": 0.2},
        ],
    },
}
