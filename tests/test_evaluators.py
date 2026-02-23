"""Tests for the multi-agent specialist evaluator system."""

import threading
from unittest.mock import MagicMock, patch

import pytest

from shalom.agents.design_layer import (
    CoarseSelector,
    MultiAgentFineSelector,
    get_fine_selector,
    FineSelector,
)
from shalom.agents.evaluators import (
    DEFAULT_WEIGHTS,
    SpecialistEvaluator,
    aggregate_scores,
    create_default_evaluators,
    evaluate_parallel,
)
from shalom.core.llm_provider import LLMProvider
from shalom.core.schemas import (
    CandidateScore,
    EvaluationResponse,
    MaterialCandidate,
    RankedMaterial,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_llm():
    return MagicMock(spec=LLMProvider)


@pytest.fixture
def sample_candidates():
    return [
        MaterialCandidate(
            material_name="MoS2",
            elements=["Mo", "S"],
            reasoning="Good d-band center for HER.",
            expected_properties={"bandgap": "1.8 eV"},
        ),
        MaterialCandidate(
            material_name="WS2",
            elements=["W", "S"],
            reasoning="Similar to MoS2 but more stable.",
            expected_properties={"bandgap": "2.0 eV"},
        ),
        MaterialCandidate(
            material_name="PbTe",
            elements=["Pb", "Te"],
            reasoning="High thermoelectric performance.",
            expected_properties={"ZT": "1.5"},
        ),
    ]


def _make_evaluation(perspective: str, scores_data: list) -> EvaluationResponse:
    """Helper to create an EvaluationResponse from simple data."""
    return EvaluationResponse(
        perspective=perspective,
        scores=[
            CandidateScore(
                material_name=name,
                score=score,
                confidence=conf,
                justification=reason,
            )
            for name, score, conf, reason in scores_data
        ],
    )


# ---------------------------------------------------------------------------
# SpecialistEvaluator tests
# ---------------------------------------------------------------------------


class TestSpecialistEvaluator:
    def test_returns_evaluation_response(self, mock_llm, sample_candidates):
        expected = EvaluationResponse(
            perspective="stability",
            scores=[
                CandidateScore(
                    material_name="MoS2", score=0.8, confidence=0.7,
                    justification="Thermodynamically stable.",
                ),
                CandidateScore(
                    material_name="WS2", score=0.9, confidence=0.8,
                    justification="Very stable.",
                ),
            ],
        )
        mock_llm.generate_structured_output.return_value = expected

        evaluator = SpecialistEvaluator(mock_llm, "stability", "You are stability evaluator.")
        result = evaluator.evaluate("Find HER catalyst", sample_candidates)

        assert result.perspective == "stability"
        assert len(result.scores) == 2
        mock_llm.generate_structured_output.assert_called_once()


# ---------------------------------------------------------------------------
# aggregate_scores tests
# ---------------------------------------------------------------------------


class TestAggregateScores:
    def test_picks_highest_weighted_score(self, sample_candidates):
        evaluations = [
            _make_evaluation("stability", [
                ("MoS2", 0.8, 0.9, "stable"), ("WS2", 0.7, 0.8, "ok"), ("PbTe", 0.6, 0.7, "ok"),
            ]),
            _make_evaluation("target_property", [
                ("MoS2", 0.9, 0.9, "great"), ("WS2", 0.5, 0.7, "ok"), ("PbTe", 0.4, 0.6, "bad"),
            ]),
            _make_evaluation("dft_feasibility", [
                ("MoS2", 0.7, 0.8, "ok"), ("WS2", 0.8, 0.9, "good"), ("PbTe", 0.6, 0.5, "ok"),
            ]),
            _make_evaluation("synthesizability", [
                ("MoS2", 0.8, 0.8, "known"), ("WS2", 0.7, 0.7, "known"), ("PbTe", 0.5, 0.6, "ok"),
            ]),
            _make_evaluation("novelty", [
                ("MoS2", 0.3, 0.5, "common"), ("WS2", 0.4, 0.5, "common"), ("PbTe", 0.8, 0.7, "new"),
            ]),
            _make_evaluation("environmental_cost", [
                ("MoS2", 0.9, 0.9, "earth-abundant"),
                ("WS2", 0.8, 0.8, "ok"),
                ("PbTe", 0.2, 0.9, "Pb is toxic"),
            ]),
        ]

        result, veto_reasons = aggregate_scores(evaluations, sample_candidates)
        assert result is not None
        assert result.candidate.material_name == "MoS2"

    def test_custom_weights(self, sample_candidates):
        evaluations = [
            _make_evaluation("stability", [
                ("MoS2", 0.5, 0.9, "ok"), ("WS2", 0.9, 0.9, "great"),
            ]),
            _make_evaluation("target_property", [
                ("MoS2", 0.9, 0.9, "great"), ("WS2", 0.3, 0.9, "bad"),
            ]),
        ]
        # Heavy weight on target_property → MoS2 wins
        weights = {"stability": 0.1, "target_property": 0.9}
        result, _ = aggregate_scores(
            evaluations, sample_candidates[:2], weights=weights
        )
        assert result is not None
        assert result.candidate.material_name == "MoS2"

    def test_veto_filters_low_stability(self, sample_candidates):
        evaluations = [
            _make_evaluation("stability", [
                ("MoS2", 0.1, 0.9, "unstable!"),  # below 0.3 threshold
                ("WS2", 0.8, 0.9, "stable"),
                ("PbTe", 0.5, 0.7, "ok"),
            ]),
            _make_evaluation("target_property", [
                ("MoS2", 0.95, 0.9, "excellent"),
                ("WS2", 0.6, 0.8, "ok"),
                ("PbTe", 0.5, 0.6, "mediocre"),
            ]),
        ]
        result, veto_reasons = aggregate_scores(evaluations, sample_candidates)
        assert result is not None
        # MoS2 should be vetoed despite high target_property
        assert result.candidate.material_name != "MoS2"
        assert any("MoS2" in r and "stability" in r for r in veto_reasons)

    def test_veto_relaxed_strict_false(self, sample_candidates):
        """strict_veto=False halves thresholds, allowing metastable materials."""
        evaluations = [
            _make_evaluation("stability", [
                ("MoS2", 0.2, 0.9, "metastable"),  # 0.2 > 0.3*0.5=0.15 → passes
                ("WS2", 0.8, 0.9, "stable"),
            ]),
            _make_evaluation("target_property", [
                ("MoS2", 0.95, 0.9, "excellent"),
                ("WS2", 0.4, 0.8, "ok"),
            ]),
        ]
        # With strict_veto=True, MoS2 (0.2) is vetoed (threshold 0.3)
        result_strict, _ = aggregate_scores(
            evaluations, sample_candidates[:2], strict_veto=True
        )
        assert result_strict is not None
        assert result_strict.candidate.material_name == "WS2"

        # With strict_veto=False, MoS2 (0.2) passes (threshold 0.15)
        result_relaxed, _ = aggregate_scores(
            evaluations, sample_candidates[:2], strict_veto=False
        )
        assert result_relaxed is not None
        assert result_relaxed.candidate.material_name == "MoS2"

    def test_veto_all_filtered_returns_none(self, sample_candidates):
        evaluations = [
            _make_evaluation("stability", [
                ("MoS2", 0.1, 0.9, "unstable"),
                ("WS2", 0.1, 0.9, "unstable"),
                ("PbTe", 0.1, 0.9, "unstable"),
            ]),
        ]
        result, veto_reasons = aggregate_scores(evaluations, sample_candidates)
        assert result is None
        assert len(veto_reasons) == 3

    def test_confidence_weighting(self, sample_candidates):
        """Low-confidence scores should have less impact than high-confidence ones."""
        evaluations = [
            _make_evaluation("target_property", [
                ("MoS2", 0.9, 0.2, "guess"),   # effective: 0.9 * 0.2 = 0.18
                ("WS2", 0.7, 0.9, "certain"),  # effective: 0.7 * 0.9 = 0.63
            ]),
        ]
        weights = {"target_property": 1.0}
        result, _ = aggregate_scores(
            evaluations, sample_candidates[:2], weights=weights
        )
        assert result is not None
        # WS2 should win because of higher confidence despite lower raw score
        assert result.candidate.material_name == "WS2"


# ---------------------------------------------------------------------------
# evaluate_parallel tests
# ---------------------------------------------------------------------------


class TestEvaluateParallel:
    def test_parallel_returns_all_results(self, mock_llm, sample_candidates):
        expected = EvaluationResponse(
            perspective="test",
            scores=[
                CandidateScore(
                    material_name="MoS2", score=0.8, confidence=0.7, justification="ok"
                ),
            ],
        )
        mock_llm.generate_structured_output.return_value = expected

        evaluators = [
            SpecialistEvaluator(mock_llm, f"perspective_{i}", "prompt")
            for i in range(3)
        ]
        results = evaluate_parallel(
            evaluators, "test", sample_candidates, max_concurrent_api_calls=2
        )
        assert len(results) == 3

    def test_semaphore_limits_concurrency(self, mock_llm, sample_candidates):
        max_concurrent = 2
        concurrent_count = {"value": 0, "peak": 0}
        lock = threading.Lock()

        def tracked_evaluate(self, target_objective, candidates):
            with lock:
                concurrent_count["value"] += 1
                concurrent_count["peak"] = max(
                    concurrent_count["peak"], concurrent_count["value"]
                )
            try:
                return EvaluationResponse(
                    perspective=self.perspective,
                    scores=[
                        CandidateScore(
                            material_name="MoS2", score=0.5, confidence=0.5,
                            justification="test",
                        )
                    ],
                )
            finally:
                import time
                time.sleep(0.05)  # simulate API latency
                with lock:
                    concurrent_count["value"] -= 1

        with patch.object(SpecialistEvaluator, "evaluate", tracked_evaluate):
            evaluators = [
                SpecialistEvaluator(mock_llm, f"p{i}", "prompt")
                for i in range(4)
            ]
            evaluate_parallel(
                evaluators, "test", sample_candidates,
                max_concurrent_api_calls=max_concurrent,
            )

        # Peak concurrency should not exceed semaphore limit
        assert concurrent_count["peak"] <= max_concurrent

    def test_retry_on_failure(self, mock_llm, sample_candidates):
        call_count = {"value": 0}
        expected = EvaluationResponse(
            perspective="stability",
            scores=[
                CandidateScore(
                    material_name="MoS2", score=0.8, confidence=0.7, justification="ok"
                ),
            ],
        )

        def failing_evaluate(self, target_objective, candidates):
            call_count["value"] += 1
            if call_count["value"] <= 1:
                raise RuntimeError("API rate limit")
            return expected

        with patch.object(SpecialistEvaluator, "evaluate", failing_evaluate):
            evaluators = [SpecialistEvaluator(mock_llm, "stability", "prompt")]
            results = evaluate_parallel(
                evaluators, "test", sample_candidates, api_max_retries=3
            )
            assert len(results) == 1
            assert call_count["value"] == 2  # 1 failure + 1 success


# ---------------------------------------------------------------------------
# create_default_evaluators tests
# ---------------------------------------------------------------------------


class TestCreateDefaultEvaluators:
    def test_creates_six_evaluators(self, mock_llm):
        evaluators = create_default_evaluators(mock_llm)
        assert len(evaluators) == 6

    def test_perspectives_match_weights(self, mock_llm):
        evaluators = create_default_evaluators(mock_llm)
        perspectives = {e.perspective for e in evaluators}
        assert perspectives == set(DEFAULT_WEIGHTS.keys())

    def test_prompts_contain_physics_rules(self, mock_llm):
        evaluators = create_default_evaluators(mock_llm)
        prompt_map = {e.perspective: e.system_prompt for e in evaluators}

        assert "Hume-Rothery" in prompt_map["stability"]
        assert "d-band center" in prompt_map["target_property"]
        assert "DFT+U" in prompt_map["dft_feasibility"]
        assert "Goldschmidt tolerance factor" in prompt_map["synthesizability"]
        assert "elemental substitution" in prompt_map["novelty"]
        assert "Pb" in prompt_map["environmental_cost"]

    def test_prompts_contain_confidence_rule(self, mock_llm):
        evaluators = create_default_evaluators(mock_llm)
        for e in evaluators:
            assert "CONFIDENCE RULE" in e.system_prompt


# ---------------------------------------------------------------------------
# MultiAgentFineSelector tests
# ---------------------------------------------------------------------------


class TestMultiAgentFineSelector:
    def test_end_to_end(self, mock_llm, sample_candidates):
        """Full flow: 6 evaluators → aggregate → RankedMaterial."""
        def make_eval_response(**kwargs):
            perspective = "unknown"
            user_prompt = kwargs.get("user_prompt", "")
            for p in DEFAULT_WEIGHTS:
                if f"'{p}'" in user_prompt:
                    perspective = p
                    break
            return EvaluationResponse(
                perspective=perspective,
                scores=[
                    CandidateScore(
                        material_name=c.material_name,
                        score=0.7,
                        confidence=0.8,
                        justification="Good candidate.",
                    )
                    for c in sample_candidates
                ],
            )

        mock_llm.generate_structured_output.side_effect = make_eval_response

        selector = MultiAgentFineSelector(mock_llm, parallel=False)
        result = selector.rank_and_select("Find HER catalyst", sample_candidates)

        assert isinstance(result, RankedMaterial)
        assert result.score > 0

    def test_micro_loop_veto_feedback(self, mock_llm, sample_candidates):
        """All vetoed → CoarseSelector re-invoked → new candidates evaluated."""
        call_count = {"value": 0}

        # First round: all vetoed (stability = 0.1)
        # Second round: one passes (stability = 0.8)
        def eval_side_effect(**kwargs):
            call_count["value"] += 1
            user_prompt = kwargs.get("user_prompt", "")
            # Determine if this is first or second batch
            if "Avoid these issues" in user_prompt or call_count["value"] > 6:
                # Second round candidates
                return EvaluationResponse(
                    perspective="stability",
                    scores=[
                        CandidateScore(
                            material_name="NiPS3", score=0.8, confidence=0.8,
                            justification="Stable.",
                        ),
                    ],
                )
            return EvaluationResponse(
                perspective="stability",
                scores=[
                    CandidateScore(
                        material_name=c.material_name, score=0.1, confidence=0.9,
                        justification="Unstable!",
                    )
                    for c in sample_candidates
                ],
            )

        mock_llm.generate_structured_output.side_effect = eval_side_effect

        coarse = MagicMock(spec=CoarseSelector)
        new_candidate = MaterialCandidate(
            material_name="NiPS3", elements=["Ni", "P", "S"],
            reasoning="Novel 2D material.", expected_properties={},
        )
        coarse.select.return_value = [new_candidate]

        # Use only 1 evaluator for simplicity
        evaluator = SpecialistEvaluator(mock_llm, "stability", "test")
        selector = MultiAgentFineSelector(
            mock_llm, evaluators=[evaluator], parallel=False,
            weights={"stability": 1.0}, veto_thresholds={"stability": 0.3},
        )

        result = selector.rank_and_select(
            "Find catalyst", sample_candidates,
            coarse_selector=coarse, max_design_retries=1,
        )

        coarse.select.assert_called_once()
        assert result.candidate.material_name == "NiPS3"


# ---------------------------------------------------------------------------
# get_fine_selector factory tests
# ---------------------------------------------------------------------------


class TestGetFineSelector:
    def test_simple_mode(self, mock_llm):
        selector = get_fine_selector(mock_llm, mode="simple")
        assert isinstance(selector, FineSelector)

    def test_multi_agent_mode(self, mock_llm):
        selector = get_fine_selector(mock_llm, mode="multi_agent")
        assert isinstance(selector, MultiAgentFineSelector)

    def test_unknown_mode_raises(self, mock_llm):
        with pytest.raises(ValueError, match="Unknown selector mode"):
            get_fine_selector(mock_llm, mode="invalid")

    def test_kwargs_passed_to_multi_agent(self, mock_llm):
        selector = get_fine_selector(
            mock_llm, mode="multi_agent", strict_veto=False, parallel=False
        )
        assert isinstance(selector, MultiAgentFineSelector)
        assert selector.strict_veto is False
        assert selector.parallel is False
