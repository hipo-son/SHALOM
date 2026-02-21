# SHALOM: System of Hierarchical Agents for Logical Orchestration of Materials

**SHALOM**은 거대 언어 모델(LLM)과 다중 에이전트 프레임워크를 기반으로 한 범용 **계층적 워크플로우 오케스트레이션 시스템(Systematic Workflow Orchestration System)**입니다. 

논문 및 프로그램의 대표적인 활용 예시(Example Use Case)로 **자율 소재 탐색(Autonomous Materials Discovery)** 시스템을 구현하여, 추론 기반의 문제 해결 및 복잡한 워크플로우를 자동화하는 프레임워크의 범용성을 입증합니다.

## 프로젝트 개요
"Autonomous multi-agent framework for reasoning-driven materials discovery and systematic workflow orchestration."

SHALOM은 독립적인 역할을 가진 에이전트들이 상호 작용하며, 외부 환경(HPC, DFT 등)과 안전하게 연동하여 복잡한 전체 프로세스를 자율적으로 기획, 실행, 평가합니다.

## 핵심 기능 (Core Features)
- **다중 에이전트 오케스트레이션 (Multi-Agent Orchestration)**: 기획(Design), 실행(Simulation), 평가(Review) 등 독립적 계층으로 분리된 에이전트 협업.
- **추론 기반 워크플로우 자동화 (Reasoning-Driven Workflow)**: 자연어 목표를 바탕으로 스스로 계획을 수립하고, 중간 결과를 평가하여 피드백 루프를 형성.
- **MCP (Model Context Protocol) 환경 연동**: 외부 시스템(예: Slurm 기반 HPC 자원, VASP 등)에 대한 보안이 유지된 작업 제출 및 모니터링 체계.
- **계층적 로직 구조**: Triage-Ranking(선별 및 순위 지정) 기법을 통한 효율적인 의사결정 방식 지원.

## 적용 예시: 자율 소재 탐색 시스템
SHALOM 프레임워크의 대표적인 데모로서 분자 구조 생성(ASE), 양자 화학 계산(VASP) 시스템과의 연동을 통한 **자율 소재 탐색** 파이프라인이 포함되어 있습니다. 이는 본 프레임워크가 고비용의 시뮬레이션을 어떻게 논리적이고 효율적으로 자동화할 수 있는지 보여주는 강력한 예시입니다.

---

자세한 설계 및 로드맵은 `docs/master_design_document.md`를 참고하세요.
