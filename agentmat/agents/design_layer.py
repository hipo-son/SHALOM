import json
from typing import List, Dict, Any
from pydantic import BaseModel, create_model

from agentmat.core.llm_provider import LLMProvider
from agentmat.core.schemas import MaterialCandidate, RankedMaterial

# 동적으로 List[MaterialCandidate]를 리턴받기 위한 Wrapper 스키마
CandidateListResponse = create_model(
    'CandidateListResponse',
    candidates=(List[MaterialCandidate], ...)
)

class CoarseSelector:
    """
    Design Layer의 1단계: 대략적 선별자(Coarse Selector)
    방대한 화학 공간에서 타겟 목표에 부합하는 3~5개의 유망 후보군을 1차 선별합니다.
    """
    def __init__(self, llm_provider: LLMProvider):
        self.llm = llm_provider
        self.system_prompt = """
        당신은 "Coarse Selector"입니다. 세계 최고 수준의 전산재료과학자로서 일합니다.
        주어진 자연어 목표(Target Objective)를 달성하기 위해 가장 유망한 3~5개의 소재 후보를 선별하세요.

        [선별 지침]
        1. 주기율표의 경향성, 전기음성도, d-band 중심 이론 등 확립된 물리적/화학적 직관을 사용하세요.
        2. 무작위가 아닌, 명확한 과학적 근거(reasoning)가 있어야 합니다.
        3. 후보군은 너무 뻔한 하나의 소재에 국한되지 않고, 약간의 다양성(합금, 도핑 등)을 포함하는 것이 좋습니다.
        4. 반드시 JSON 포맷의 Candidates 리스트 형태로 응답해야 합니다.
        """

    def select(self, target_objective: str, context: str = "") -> List[MaterialCandidate]:
        """
        주어진 목표에 대해 후보군 리스트를 생성하여 반환합니다.
        """
        user_prompt = f"Target Objective: {target_objective}\n\nAdditional Context:\n{context}\n\nPlease provide 3 to 5 candidate materials."
        
        response = self.llm.generate_structured_output(
            system_prompt=self.system_prompt,
            user_prompt=user_prompt,
            response_model=CandidateListResponse
        )
        return response.candidates


class FineSelector:
    """
    Design Layer의 2단계: 정밀 선별자(Fine Selector)
    Coarse Selector가 넘겨준 후보군 내부에서 예상 속성을 기준으로 순위를 매기고 최적의 소재를 선택합니다.
    """
    def __init__(self, llm_provider: LLMProvider):
        self.llm = llm_provider
        self.system_prompt = """
        당신은 "Fine Selector"입니다.
        이제 거친 선별을 통과한 소수의 소재 후보군(Candidates)이 주어집니다.
        당신의 임무는 주어진 목표(Target Objective)에 가장 부합하는 소재가 무엇인지 정밀하게 평가(점수 0.0 ~ 1.0)하고,
        이 중 단 1개의 '최종 승자(Winner)'를 선택하는 것입니다.

        [평가 지침]
        1. 각 후보의 'reasoning'과 'expected_properties'를 심층 분석하세요.
        2. 시뮬레이션(DFT) 비용 대비 성공 확률도 고려하세요 (너무 복잡하거나 큰 셀은 감점 요소가 될 수 있음).
        3. 가장 높은 점수(Score)를 받은 후보가 다음 시뮬레이션 단계로 넘어갑니다.
        """

    def rank_and_select(self, target_objective: str, candidates: List[MaterialCandidate]) -> RankedMaterial:
        """
        후보군들의 순위를 매기고 1등(최상위) 후보를 반환합니다.
        """
        # 후보군 정보를 JSON 문자열로 변환하여 프롬프트에 주입
        candidates_json = json.dumps([c.model_dump() for c in candidates], indent=2)
        
        user_prompt = (
            f"Target Objective: {target_objective}\n\n"
            f"Candidates Pool:\n{candidates_json}\n\n"
            "Please assign a score (0.0 to 1.0) and provide ranking justification. "
            "Return ONLY the best matching material as the RankedMaterial."
        )

        best_candidate = self.llm.generate_structured_output(
            system_prompt=self.system_prompt,
            user_prompt=user_prompt,
            response_model=RankedMaterial
        )
        return best_candidate
