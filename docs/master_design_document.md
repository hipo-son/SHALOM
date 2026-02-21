# AgentMat (가칭) 마스터 설계 문서 (Master Design Document)

본 프로젝트는 확장성 높은 Python 라이브러리 및 MCP(Model Context Protocol) 기반 아키텍처를 결합한 범용 다중 에이전트 소재 탐색 프레임워크입니다.

## 1. 시스템 핵심 아키텍처 (System Architecture)
전체 시스템은 역할이 명확히 분리된 3개의 독립적인 계층으로 구성됩니다.
- **Design Layer (기획/추론 계층)**: 자연어 목표를 바탕으로 다음에 어떤 소재를 시뮬레이션할지 결정하는 상위 에이전트 그룹입니다.
- **Simulation Layer (실행/검증 계층)**: 결정된 소재를 실제 DFT 입력 파일로 변환하고, 구조적 결함을 시뮬레이션 전에 검증합니다.
- **Review Layer (평가 계층)**: 시뮬레이션 결과(예: VASP OUTCAR)를 분석하여 목표 달성 여부를 판단하고, 피드백 루프를 닫습니다.

## 2. Design Layer: 계층적 탐색 로직 (Triage-Ranking)
무작위 탐색을 방지하고 계산 비용을 줄이기 위해 선별-순위 지정(Triage-ranking) 아키텍처를 도입합니다.
- **Coarse Selector (대략적 선별자)**: 방대한 전체 화학 공간에서 3~5개의 유망한 후보군(Pool)을 1차 선별합니다. (주기율표 경향성, d-band 중심 이론 등 활용)
- **Fine Selector (정밀 선별자)**: 소규모 후보군 내부에서 예상 속성을 기준으로 순위를 매기고, 타겟에 가장 근접한 단 1개의 소재를 최종 선택합니다.

## 3. Simulation Layer: 사전 검증 및 자동화 로직
HPC 자원(Slurm)에 작업 제출 전, POSCAR의 물리적 타당성을 검증하는 3단계 자가 수정 루프를 구축합니다.
- **Geometry Generator (구조 생성기)**: 자연어 요건과 ASE 기반 Python 스크립트를 작성해 초기 POSCAR 생성.
- **Form Filler (양식 작성기)**: 생성된 구조를 분석해 층수, 진공 두께, 원자 겹침 등을 규격화된 양식으로 평가.
- **Geometry Reviewer (구조 최종 승인자)**: 평가를 바탕으로 합격/불합격 결정 및 수정 지시(최대 3~5회 재시도).

## 4. MCP 기반 HPC 연동 모듈
- **VASP-Slurm MCP 서버**: LLM이 사전 정의된 도구로만 HPC에 접근하게 하여 보안 유지. (`submit_slurm_job`, `check_job_status`, `read_vasp_outcar` 등)
- **자가 복구 알고리즘 (Self-Correction)**: SCF 실패 시 Review Agent가 로그를 파싱해 INCAR 파라미터를 조정하고 재제출.

## 5. 단계별 개발 마일스톤 (Milestones)
- **Phase 1: Core Library & MCP 환경 세팅**: Python 라이브러리 구조화, ASE 기반 Geometry Generator 프롬프트 및 Form Filler 셋업.
- **Phase 2: Triage-Ranking 에이전트 루프 구현**: Coarse/Fine Selector 프롬프트 파이프라인 구축.
- **Phase 3: HPC VASP 통합 및 자가 복구 로직**: Slurm 연동, Bulk 소재 대상 E2E 테스트.
- **Phase 4: 심화 시스템 및 오픈소스 배포**: 2D/TMD 시스템 확장, 탐색 성능 메트릭 측정 도구 탑재.

---

# Multi-Agent Autonomous Materials Discovery Framework
*(Research Proposal & System Design Details)*

## Research Objective
A multi-agent orchestration framework for autonomous first-principles materials discovery that can plan/execute workflows, manage resources, evaluate candidates, and adapt search strategies.

## System Architecture Roles
- **Planner Agent**: Defines objectives, allocates budget, manages strategies.
- **Executor Agent**: Submits Slurm jobs, monitors execution, handles failures.
- **Evaluator Agent**: Parses results, computes metrics, ranks candidates.
- **Critic/Auditor Agent**: Validates tool usage, ensures safety/reproducibility.

## Software Architecture
Library-Centric Design implemented in Python for reproducibility and HPC integration. Includes Agent Framework, Tool System, Backend Layer (Slurm), and Provider Interface (LLM APIs).
