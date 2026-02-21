import os
from unittest.mock import MagicMock
import pytest

from agentmat.core.schemas import MaterialCandidate, RankedMaterial
from agentmat.core.llm_provider import LLMProvider
from agentmat.agents.design_layer import CoarseSelector, FineSelector

def test_coarse_selector_mock():
    """LLM API를 가짜(mock)로 응답하게 하여 CoarseSelector의 데이터 흐름을 테스트합니다."""
    # 1. Mock LLM Provider 생성
    mock_llm = MagicMock(spec=LLMProvider)
    
    # 2. 가짜 응답 데이터 정의
    mock_response = MagicMock()
    mock_response.candidates = [
        MaterialCandidate(
            material_name="Copper (111)",
            elements=["Cu"],
            reasoning="d-band center is appropriate for intermediate binding.",
            expected_properties={"surface_energy": "low"}
        ),
        MaterialCandidate(
            material_name="Platinum",
            elements=["Pt"],
            reasoning="High catalytic activity.",
            expected_properties={"cost": "high"}
        )
    ]
    
    # generate_structured_output 호출 시 가짜 응답 반환하도록 설정
    mock_llm.generate_structured_output.return_value = mock_response

    # 3. 에이전트 실행
    selector = CoarseSelector(llm_provider=mock_llm)
    candidates = selector.select("Find a good catalyst for hydrogen evolution")

    # 4. 검증
    assert len(candidates) == 2
    assert candidates[0].elements == ["Cu"]
    assert mock_llm.generate_structured_output.called

def test_fine_selector_mock():
    """FineSelector 파이프라인 테스트"""
    mock_llm = MagicMock(spec=LLMProvider)
    
    candidates = [
        MaterialCandidate(
            material_name="Copper (111)", elements=["Cu"], reasoning="Good", expected_properties={}
        ),
        MaterialCandidate(
            material_name="Platinum", elements=["Pt"], reasoning="Better", expected_properties={}
        )
    ]
    
    # 가짜 1등 후보 응답
    mock_llm.generate_structured_output.return_value = RankedMaterial(
        candidate=candidates[1],
        score=0.95,
        ranking_justification="Pt has the best theoretical performance."
    )
    
    selector = FineSelector(llm_provider=mock_llm)
    winner = selector.rank_and_select("Best catalyst", candidates)
    
    assert winner.score == 0.95
    assert winner.candidate.material_name == "Platinum"
