import os
import sys

# 프로젝트 루트 디렉토리를 PYTHONPATH에 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agentmat.core.llm_provider import LLMProvider
from agentmat.agents.design_layer import CoarseSelector, FineSelector

def main():
    print("=== AgentMat Design Layer API Integration Test ===")
    
    # 환경변수 체크
    if not os.environ.get("OPENAI_API_KEY") and not os.environ.get("ANTHROPIC_API_KEY"):
        print("\n[오류] OPENAI_API_KEY 또는 ANTHROPIC_API_KEY 환경 변수가 설정되지 않았습니다.")
        print("이 스크립트를 실행하려면 터미널에서 API 키를 먼저 설정해야 합니다.")
        print("예시 (Windows PowerShell): $env:OPENAI_API_KEY='your-key-here'")
        return

    # 제공자 선택 (우선순위: OpenAI -> Anthropic)
    provider_type = "openai" if os.environ.get("OPENAI_API_KEY") else "anthropic"
    model_name = "gpt-4o" if provider_type == "openai" else "claude-3-5-sonnet-20240620"
    
    print(f"\n[1] LLM Provider 초기화: {provider_type} (모델: {model_name})")
    llm = LLMProvider(provider_type=provider_type, model_name=model_name)

    # 타겟 목표 설정
    target_objective = "A highly efficient, low-cost transition metal alloy surface for Hydrogen Evolution Reaction (HER)."
    print(f"\n[2] Target Objective: {target_objective}")

    # 1단계: Coarse Selector 실행
    print("\n[3] Running Coarse Selector (대략적 선별)...")
    coarse_selector = CoarseSelector(llm_provider=llm)
    try:
        candidates = coarse_selector.select(target_objective)
        print(f"-> {len(candidates)}개의 후보군이 선별되었습니다:")
        for i, c in enumerate(candidates):
            print(f"   [{i+1}] {c.material_name} (원소: {c.elements})")
            print(f"       근거: {c.reasoning}")
    except Exception as e:
        print(f"Coarse Selector 실행 중 오류 발생: {e}")
        return

    # 2단계: Fine Selector 실행
    if not candidates:
        print("선별된 후보군이 없어 종료합니다.")
        return

    print("\n[4] Running Fine Selector (정밀 순위 지정)...")
    fine_selector = FineSelector(llm_provider=llm)
    try:
        winner = fine_selector.rank_and_select(target_objective, candidates)
        print("\n=== [최종 승자 (Winner)] ===")
        print(f"소재 이름: {winner.candidate.material_name}")
        print(f"물리적 근거: {winner.candidate.reasoning}")
        print(f"부여된 점수: {winner.score}")
        print(f"선정 사유 (Justification):\n{winner.ranking_justification}")
    except Exception as e:
        print(f"Fine Selector 실행 중 오류 발생: {e}")

if __name__ == "__main__":
    main()
