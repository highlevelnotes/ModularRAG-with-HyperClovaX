# src/collector/etf_collector.py
import requests
import pandas as pd
import json
import logging
import time
from typing import List, Dict, Tuple, Optional
from urllib.parse import urlencode
import FinanceDataReader as fdr
from bs4 import BeautifulSoup

class KoreanETFCollector:
    """국내 ETF 데이터 수집기"""
    
    def __init__(self, config):
        self.config = config
        self.logger = self._setup_logger()
        
        # 퇴직연금 투자 제한 키워드
        self.leverage_keywords = [
            '레버리지', 'LEVERAGE', '인버스', 'INVERSE', 
            '2X', '3X', '-1X', '곱버스', 'LEVERAGED'
        ]
        
        self.overseas_keywords = [
            '미국', 'US', 'USA', '중국', 'CHINA', '일본', 'JAPAN',
            '유럽', 'EUROPE', '신흥국', 'EMERGING', '글로벌', 'GLOBAL',
            '해외', 'OVERSEAS', '선진국', 'DEVELOPED'
        ]
        
        self.derivative_keywords = [
            '선물', 'FUTURES', '파생', 'DERIVATIVE', '옵션', 'OPTION',
            '스왑', 'SWAP', '골드선물', '원유선물', '통화선물'
        ]
    
    def _setup_logger(self) -> logging.Logger:
        """로거 설정"""
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def collect_naver_etf_list(self) -> pd.DataFrame:
        """네이버 금융에서 ETF 목록 수집"""
        try:
            self.logger.info("네이버 금융에서 ETF 목록 수집 중...")
            
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            
            response = requests.get(self.config.naver_api_endpoint, headers=headers)
            response.encoding = 'cp949'
            
            if response.status_code == 200:
                data = json.loads(response.text)
                etf_list = data['result']['etfItemList']
                
                df = pd.DataFrame(etf_list)
                df = df.rename(columns={
                    'itemcode': 'code',
                    'itemname': 'name',
                    'nowVal': 'current_price',
                    'risefall': 'change_direction',
                    'changeVal': 'change_amount',
                    'changeRate': 'change_rate'
                })
                
                # 기본 컬럼 추가 (FDR 대신)
                df['market'] = 'KRX'
                df['sector'] = 'ETF'
                
                self.logger.info(f"네이버에서 {len(df)}개 ETF 수집 완료")
                return df
            else:
                self.logger.error(f"네이버 API 호출 실패: {response.status_code}")
                return pd.DataFrame()
                
        except Exception as e:
            self.logger.error(f"네이버 ETF 목록 수집 오류: {str(e)}")
            return pd.DataFrame()
    
    def is_pension_eligible_etf(self, etf_info: Dict) -> Tuple[bool, str]:
        """퇴직연금 투자 가능 ETF 판별"""
        etf_name = etf_info.get('name', '').upper()
        
        # 1. 레버리지/인버스 ETF 제외
        for keyword in self.leverage_keywords:
            if keyword in etf_name:
                return False, f"레버리지/인버스 ETF ({keyword})"
        
        # 2. 해외 직접투자 ETF 제외 (국내 자본시장 투자만)
        if not self._is_domestic_investment_etf(etf_info):
            return False, "해외 직접투자 ETF"
        
        # 3. 파생상품 위험평가액 40% 초과 우려 ETF 제외
        for keyword in self.derivative_keywords:
            if keyword in etf_name:
                return False, f"파생상품 위험평가액 초과 우려 ({keyword})"
        
        return True, "투자가능"
    
    def _is_domestic_investment_etf(self, etf_info: Dict) -> bool:
        """국내 투자 ETF 여부 확인"""
        etf_name = etf_info.get('name', '').upper()
        
        # 해외 투자 키워드 확인
        for keyword in self.overseas_keywords:
            if keyword in etf_name:
                return False
        
        # 국내 투자 키워드 확인
        domestic_keywords = [
            'KOSPI', 'KOSDAQ', 'KRX', '코스피', '코스닥',
            '한국', 'KOREA', 'KR', '국내', 'DOMESTIC',
            '삼성', 'SK', 'LG', '현대', '포스코',
            '은행', 'BANK', '증권', '보험', '건설', '조선',
            'K-', '원화', 'KRW'
        ]
        
        for keyword in domestic_keywords:
            if keyword in etf_name:
                return True
        
        # 기본적으로 명확하지 않은 경우 해외 투자로 간주
        return False
    
    def filter_pension_eligible_etfs(self, etf_df: pd.DataFrame) -> pd.DataFrame:
        """퇴직연금 투자 가능 ETF 필터링"""
        self.logger.info("퇴직연금 투자 가능 ETF 필터링 중...")
        
        eligible_etfs = []
        
        for _, row in etf_df.iterrows():
            etf_info = row.to_dict()
            is_eligible, reason = self.is_pension_eligible_etf(etf_info)
            
            if is_eligible:
                etf_info['pension_eligible'] = True
                etf_info['eligibility_reason'] = reason
                eligible_etfs.append(etf_info)
            else:
                self.logger.debug(f"제외: {etf_info.get('name', 'Unknown')} - {reason}")
        
        result_df = pd.DataFrame(eligible_etfs)
        self.logger.info(f"퇴직연금 투자 가능 ETF: {len(result_df)}개")
        
        return result_df
    
    def get_comprehensive_etf_list(self) -> pd.DataFrame:
        """종합 ETF 목록 수집 (네이버만 사용)"""
        self.logger.info("종합 ETF 목록 수집 시작...")
        
        # 네이버에서만 ETF 목록 수집 (더 안정적)
        naver_df = self.collect_naver_etf_list()
        
        if naver_df.empty:
            self.logger.error("ETF 목록 수집 실패")
            return pd.DataFrame()
        
        # 퇴직연금 투자 가능 ETF 필터링
        pension_eligible_df = self.filter_pension_eligible_etfs(naver_df)
        
        return pension_eligible_df
