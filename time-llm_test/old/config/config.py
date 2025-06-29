# config/config.py
import os
from dataclasses import dataclass
from typing import Dict, List, Optional
from datetime import datetime, timedelta

@dataclass
class DataCollectionConfig:
    """데이터 수집 설정"""
    # API 설정
    naver_api_endpoint: str = "https://finance.naver.com/api/sise/etfItemList.nhn"
    
    # 퇴직연금 투자 제한 설정
    max_derivative_risk_ratio: float = 40.0  # 파생상품 위험평가액 40% 초과 금지
    exclude_leverage_inverse: bool = True    # 레버리지/인버스 ETF 제외
    exclude_overseas_direct: bool = True     # 해외 직접투자 ETF 제외
    max_risk_asset_ratio: float = 70.0      # 위험자산 투자한도 70%
    
    # Time-LLM 설정
    patch_length: int = 30                   # 입력 시퀀스 길이
    prediction_days: int = 10               # 예측 일수
    min_data_points: int = 252              # 최소 데이터 포인트 (1년)
    
    # 데이터 수집 설정
    start_date: str = (datetime.now() - timedelta(days=1095)).strftime('%Y-%m-%d')  # 3년
    end_date: str = datetime.now().strftime('%Y-%m-%d')
    max_etfs: int = 50                      # 테스트용 ETF 수 제한
    
    # 저장 경로
    data_dir: str = "data"
    raw_data_dir: str = f"{data_dir}/raw"
    processed_data_dir: str = f"{data_dir}/processed"
    time_llm_data_dir: str = f"{data_dir}/time_llm_format"

config = DataCollectionConfig()
