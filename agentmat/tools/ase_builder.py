from typing import List, Optional, Tuple
from ase import Atoms
from ase.build import bulk, surface, add_vacuum
from ase.io import write
import os

class ASEBuilder:
    """
    Simulation Layer의 Geometry Generator 에이전트가 호출할 ASE 관련 헬퍼 도구 모음입니다.
    LLM이 직접 ASE 전체 문법을 작성하기보다, 이 도구들을 조립하여 쓰도록 유도합니다.
    """
    
    @staticmethod
    def construct_bulk(symbol: str, crystalstructure: str, a: Optional[float] = None) -> Atoms:
        """
        기본적인 Bulk 구조를 생성합니다.
        
        Args:
            symbol (str): 원소 기호 (예: 'Cu')
            crystalstructure (str): 결정구조 ('fcc', 'bcc', 'hcp', 'sc' 등)
            a (float, optional): 격자 상수. 미제공 시 ASE 기본값 사용
            
        Returns:
            Atoms: 생성된 구조체
        """
        return bulk(symbol, crystalstructure=crystalstructure, a=a)

    @staticmethod
    def construct_surface(
        atoms: Atoms, 
        indices: Tuple[int, int, int], 
        layers: int, 
        vacuum: float = 10.0
    ) -> Atoms:
        """
        주어진 격자에서 표면(Surface) 슬래브를 생성합니다.
        
        Args:
            atoms (Atoms): 베이스가 되는 Bulk 구조체
            indices (Tuple[int,int,int]): 밀러 지수 (예: (1, 1, 1))
            layers (int): 슬래브의 층 수
            vacuum (float): 슬래브 위아래에 추가할 진공 두께 (Angstrom)
            
        Returns:
            Atoms: 표면 구조체
        """
        # 주기성 유지를 위해 표본 만들기
        slab = surface(atoms, indices, layers)
        slab.center(vacuum=vacuum, axis=2) # z축 기준 중앙 정렬 및 진공 추가
        return slab

    @staticmethod
    def save_poscar(atoms: Atoms, filename: str = "POSCAR", directory: str = ".") -> str:
        """
        생성된 구조체를 VASP POSCAR 포맷으로 저장합니다.
        
        Args:
            atoms (Atoms): 저장할 구조체
            filename (str): 파일 이름 (기본값: POSCAR)
            directory (str): 저장할 디렉토리 (기본값: 현재 디렉토리)
            
        Returns:
            str: 저장된 파일의 전체 경로
        """
        os.makedirs(directory, exist_ok=True)
        filepath = os.path.join(directory, filename)
        
        # VASP 버전 명시를 피하기 위해 포맷 지정 및 write
        write(filepath, atoms, format="vasp")
        return filepath
        
    @staticmethod
    def analyze_structure(atoms: Atoms) -> dict:
        """
        Form Filler 에이전트가 사용할 기본적인 구조 정보를 반환합니다.
        
        Args:
            atoms (Atoms): 분석할 구조체
            
        Returns:
            dict: 구조 정보 (원자수, 부피, 가장 가까운 거리 등)
        """
        num_atoms = len(atoms)
        volume = atoms.get_volume()
        
        # 가장 가까운 원자 간 거리 (단순 O(N^2) 계산. 대규모 셀에선 최적화 필요)
        min_dist = float('inf')
        distances = atoms.get_all_distances()
        for i in range(num_atoms):
            for j in range(i+1, num_atoms):
                if distances[i][j] < min_dist:
                    min_dist = distances[i][j]
                    
        return {
            "num_atoms": num_atoms,
            "cell_volume": volume,
            "minimum_distance": min_dist if num_atoms > 1 else 0.0,
            "cell_parameters": atoms.get_cell().tolist()
        }
