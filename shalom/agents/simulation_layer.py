import traceback
from typing import Optional, Dict, Any, Tuple
from mse import Atoms
from pydantic import BaseModel

from agentmat.core.llm_provider import LLMProvider
from agentmat.core.schemas import StructureReviewForm, RankedMaterial
from agentmat.tools.ase_builder import ASEBuilder

class GeneratorResponse(BaseModel):
    """Geometry Generator가 생성한 파이썬 스크립트 스키마"""
    python_code: str
    explanation: str

class GeometryGenerator:
    """
    Simulation Layer의 1단계: 구조 생성기
    자연어 목표와 타겟 소재(RankedMaterial)를 바탕으로 ASE Python 스크립트를 작성하여
    초기 구조(POSCAR)를 생성합니다.
    """
    def __init__(self, llm_provider: LLMProvider):
        self.llm = llm_provider
        self.system_prompt = """
        당신은 "Geometry Generator"입니다.
        주어진 소재(Winner Candidate) 정보와 사용자의 시뮬레이션 목표를 읽고,
        ASE (Atomic Simulation Environment) 라이브러리를 사용하여 해당 물리적 구조를 생성하는
        Python 코드를 작성하세요.

        [작성 지침]
        1. `from ase.build import bulk, surface` 등을 활용하세요.
        2. 최종적으로 생성된 Atoms 객체는 `atoms` 라는 이름의 변수에 할당되어야 합니다.
        3. 반환하는 `python_code` 내부에 `atoms = bulk(...)` 와 같은 로직이 반드시 포함되어야 합니다.
        4. (주의) 코드 스니펫 외에 markdown 기호(```python)를 포함하지 말고 순수 파이썬 코드만 반환하세요.
        """

    def generate_code(self, target_objective: str, ranked_material: RankedMaterial, previous_feedback: str = "") -> GeneratorResponse:
        user_prompt = f"Target Objective: {target_objective}\nWinner Material:\n{ranked_material.model_dump_json(indent=2)}\n\n"
        if previous_feedback:
            user_prompt += f"[CRITICAL FEEDBACK FROM REVIEWER]\n{previous_feedback}\n이 피드백을 반영하여 코드를 수정하세요.\n\n"
        
        user_prompt += "Please provide the Python code to generate the `atoms` object."
        
        return self.llm.generate_structured_output(
            system_prompt=self.system_prompt,
            user_prompt=user_prompt,
            response_model=GeneratorResponse
        )

class FormFiller:
    """
    Simulation Layer의 2단계: 양식 작성기
    Geometry Generator가 만든 Atoms 객체를 분석하여 규격화된 질문 양식(StructureReviewForm)을 채웁니다.
    이 단계는 LLM이 아닌 규칙 기반(Rule-based)으로 빠르고 정확하게 수행됩니다.
    """
    @staticmethod
    def evaluate_atoms(atoms: Atoms, filepath: Optional[str] = None) -> StructureReviewForm:
        # ASEBuilder의 분석 툴 사용
        analysis = ASEBuilder.analyze_structure(atoms)
        
        num_atoms = analysis["num_atoms"]
        min_dist = analysis["minimum_distance"]
        vol = analysis["cell_volume"]
        
        is_valid = True
        feedback = "구조가 물리적으로 타당해 보입니다."
        
        # 간단한 규칙 검사 (예시)
        if min_dist < 0.5:
            is_valid = False
            feedback = f"물리적 오류: 원자 간 거리가 너무 가깝습니다 ({min_dist:.2f} Å). 원자가 겹쳐있을 수 있습니다."
        elif num_atoms == 0:
            is_valid = False
            feedback = "오류: 원자가 하나도 생성되지 않았습니다."
        elif vol > 10000:
            # HPC 자원 보호를 위한 셀 크기 제한
            is_valid = False
            feedback = f"경고: 단위 셀이 너무 큽니다 (Volume: {vol:.1f}). 시뮬레이션 비용이 과도할 수 있습니다."

        # 진공층 검사 (대략적)
        vacuum_thickness = None
        cell = atoms.get_cell()
        positions = atoms.positions
        if cell[2][2] > 10.0:
            z_positions = positions[:, 2]
            slab_thickness = z_positions.max() - z_positions.min()
            vacuum_thickness = cell[2][2] - slab_thickness
            
            if vacuum_thickness > 0 and vacuum_thickness < 8.0:
                is_valid = False
                feedback = f"경고: 진공층 두께({vacuum_thickness:.1f} Å)가 너무 얇아 주기율 경계 조건에 의한 상호작용 우려가 있습니다 (권장 10~15 Å)."

        return StructureReviewForm(
            file_path=filepath,
            num_atoms=num_atoms,
            cell_volume=vol,
            minimum_distance=min_dist,
            vacuum_thickness=vacuum_thickness,
            is_valid=is_valid,
            feedback=feedback
        )

class GeometryReviewer:
    """
    Simulation Layer의 3단계: 구조 최종 승인자 (통제 루프 관리)
    """
    def __init__(self, generator: GeometryGenerator, max_retries: int = 3):
        self.generator = generator
        self.max_retries = max_retries

    def run_creation_loop(self, target_objective: str, ranked_material: RankedMaterial) -> Tuple[bool, Optional[Atoms], str]:
        """
        성공적인 구조가 만들어질 때까지 Generator -> exec() -> FormFiller -> Review 형태의 루프를 실행합니다.
        """
        feedback = ""
        
        for attempt in range(self.max_retries):
            # 1. 코드 생성 요청
            gen_response = self.generator.generate_code(target_objective, ranked_material, feedback)
            python_code = gen_response.python_code
            
            # 2. 실행 환경 구성 및 실행
            local_vars = {"bulk": ASEBuilder.construct_bulk, "surface": ASEBuilder.construct_surface}
            
            try:
                # 보안 위험이 있지만, 연구용 로컬 실행을 가정함.
                # 실제 운영 시 WASM/Docker 샌드박스에서 실행해야 함.
                exec(python_code, globals(), local_vars)
                
                if "atoms" not in local_vars:
                    raise ValueError("'atoms' 변수가 생성되지 않았습니다. 코드를 수정하세요.")
                
                atoms = local_vars["atoms"]
                
                # 3. Form Filler 평가
                form_result = FormFiller.evaluate_atoms(atoms)
                
                if form_result.is_valid:
                    # 성골! 구조를 파일로 저장
                    filename = f"POSCAR_{ranked_material.candidate.material_name.replace(' ', '_')}"
                    filepath = ASEBuilder.save_poscar(atoms, filename=filename, directory="generated_structures")
                    return True, atoms, filepath
                else:
                    # 규칙에 어긋남 -> 피드백 갱신 후 재시도
                    feedback = form_result.feedback
                    print(f"[Attempt {attempt+1}] 구조 검증 실패: {feedback}")
                    
            except Exception as e:
                # 코드 실행 오류 파싱
                error_trace = traceback.format_exc()
                feedback = f"Python 실행 중 예외가 발생했습니다:\n{str(e)}\n\nTraceback:\n{error_trace}\nASE 문법을 확인하고 올바른 atoms 객체를 생성하세요."
                print(f"[Attempt {attempt+1}] 코드 실행 실패.")
                
        return False, None, f"최대 시도 횟수({self.max_retries}) 초과. 최종 에러: {feedback}"
