# AgentMat: 멀티 에이전트 자율 소재 탐색 프레임워크

이 프로젝트는 거대 언어 모델(LLM)과 다중 에이전트 프레임워크를 기반으로 한 **자율 소재 탐색 시스템(Autonomous Materials Discovery System)**입니다. DFT(VASP 등) 계산과 HPC(Slurm) 환경을 MCP(Model Context Protocol)로 연동하여, 소재 탐색의 전 과정을 자동화하고 최적화하는 것을 목표로 합니다.

## 핵심 기능
- **계층적 탐색 로직 (Triage-Ranking)**: Coarse Selector와 Fine Selector를 통한 효율적인 후보군 선별.
- **자동화된 시뮬레이션 파이프라인**: ASE를 활용한 구조 생성 및 사전 검증 루프.
- **MCP 기반 HPC 연동**: 보안이 유지된 형태의 Slurm/VASP 작업 제출 및 모니터링.
- **다중 에이전트 오케스트레이션**: 기획(Design), 실행(Simulation), 평가(Review) 계층의 분리.

자세한 설계 및 로드맵은 `docs/master_design_document.md`를 참고하세요.
