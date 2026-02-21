from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

class MaterialCandidate(BaseModel):
    """
    Coarse Selector가 선별한 대략적인 소재 후보군 스키마
    """
    material_name: str = Field(description="소재의 일반적인 이름 또는 화학식 (예: 'Copper', 'MoS2')")
    elements: List[str] = Field(description="소재를 구성하는 원소 기호 리스트 (예: ['Cu'], ['Mo', 'S'])")
    reasoning: str = Field(description="이 소재를 후보군으로 선정한 과학적/물리적 이유 (예: d-band 이론, 전기음성도 차이 등)")
    expected_properties: Dict[str, Any] = Field(default_factory=dict, description="이 소재에서 기대되는 대략적인 물성 (예: {'bandgap': '1.5~2.0 eV'})")

class RankedMaterial(BaseModel):
    """
    Fine Selector가 평가하고 순위를 매긴 최종 후보 스키마
    """
    candidate: MaterialCandidate = Field(description="평가 대상 소재 후보")
    score: float = Field(description="타겟 목표에 부합하는 정도 (0.0 ~ 1.0)")
    ranking_justification: str = Field(description="이 점수와 순위를 부여한 상세 이유")

class StructureReviewForm(BaseModel):
    """
    Simulation Layer에서 생성된 초기 구조(POSCAR 대용)를 검증하기 위한 양식
    """
    file_path: Optional[str] = Field(default=None, description="분석된 구조 파일의 상대/절대 경로")
    num_atoms: int = Field(description="구조 내 총 원자 수")
    cell_volume: float = Field(description="단위 셀의 부피 (Angstrom^3)")
    minimum_distance: float = Field(description="가장 가까운 두 원자 사이의 거리 (Angstrom). 원자 겹침 확인용")
    vacuum_thickness: Optional[float] = Field(default=None, description="표면/2D 물질의 경우 십입된 진공층의 두께 (Angstrom)")
    is_valid: bool = Field(description="이 구조가 VASP 계산을 돌리기에 물리적으로 타당한가?")
    feedback: str = Field(description="구조에 문제가 있다면 어떤 부분을 수정해야 하는지 구체적인 피드백")

class AgentMessage(BaseModel):
    """
    에이전트 간 통신에 사용되는 기본 메시지 스키마
    """
    sender: str = Field(description="메시지를 보내는 에이전트 이름")
    receiver: str = Field(description="메시지를 받는 에이전트 이름")
    content: str = Field(description="자연어 메시지 본문")
    payload: Optional[Dict[str, Any]] = Field(default=None, description="전달할 추가적인 구조화된 데이터 (JSON)")
