# 필요한 라이브러리 설치 및 임포트
import os
import json
import pandas as pd
import numpy as np
import requests
import warnings
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import time
import re
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import logging
import yfinance as yf
from dataclasses import dataclass
import traceback
from pykrx import stock as pkstock

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')

# 데이터 클래스 정의
@dataclass
class ETFInfo:
    symbol: str
    name: str
    sector: str
    asset_type: str
    is_domestic: bool = True

@dataclass
class TrainingExample:
    etf_symbol: str
    etf_name: str
    sector: str
    time_period: str
    input_sequence: Dict[str, List[float]]
    target_prediction: List[float]
    market_context: Dict[str, Any]
    instruction: str
    response: str

class DomesticETFCollector:
    """국내 자본시장 활성화를 위한 국내 전용 ETF 수집기"""
    
    def __init__(self):
        # 국내 ETF 식별 키워드
        self.domestic_keywords = [
            '코스피', 'KOSPI', '코스닥', 'KOSDAQ', 'KRX', '한국', 'Korea',
            '삼성', '반도체', 'K-', '국내', '지배구조', 'ESG', '바이오',
            '게임', 'IT', '금융', '부동산', '에너지', '소재', '산업재',
            '헬스케어', '통신', '유틸리티', '필수소비재', '경기소비재',
            '조선', '자동차', '화학', '철강', '건설', '기계', '전자'
        ]
        
        # 해외 투자 제외 키워드 (자금 유출 방지)
        self.exclude_keywords = [
            '미국', 'US', 'USA', '중국', 'China', '일본', 'Japan', '유럽', 'Europe',
            'S&P', 'NASDAQ', 'NASDAQ100', 'CSI', 'STOXX', '글로벌', 'Global', 
            '선진국', '신흥국', '베트남', '인도', '브라질', '러시아', 'MSCI',
            '원자재', '금', '은', '구리', '원유', 'WTI', '천연가스', '달러', 'USD',
            '항셍', 'Hang Seng', '닛케이', '다우존스', '나스닥', '스파이',
            '해외', '외국', '국제', '월드', 'World', '아시아', 'Asia'
        ]
        
        # 주요 국내 ETF 목록
        self.domestic_etf_list = [
            ETFInfo('069500', 'KODEX 200', 'Large Cap', 'Index'),
            ETFInfo('102110', 'TIGER 200', 'Large Cap', 'Index'),
            ETFInfo('114800', 'KODEX 인버스', 'Large Cap', 'Inverse'),
            ETFInfo('122630', 'KODEX 레버리지', 'Large Cap', 'Leverage'),
            ETFInfo('233740', 'KODEX 코스닥150', 'Small Cap', 'Index'),
            ETFInfo('091160', 'KODEX 반도체', 'Technology', 'Sector'),
            ETFInfo('091170', 'KODEX 은행', 'Financial', 'Sector'),
            ETFInfo('305720', 'KODEX 2차전지산업', 'Technology', 'Theme'),
            ETFInfo('275290', 'KODEX 200 고배당', 'Large Cap', 'Dividend'),
            ETFInfo('332620', 'KODEX 200 ESG', 'Large Cap', 'ESG')
        ]
    
    def get_domestic_etf_list(self) -> List[ETFInfo]:
        """국내 전용 ETF 목록 반환"""
        logger.info(f"국내 전용 ETF {len(self.domestic_etf_list)}개 로드 완료")
        return self.domestic_etf_list
    
    def is_domestic_etf(self, etf_name: str) -> bool:
        """ETF가 국내 전용인지 판단"""
        # 해외 투자 ETF 제외
        if any(keyword in etf_name for keyword in self.exclude_keywords):
            return False
        
        # 국내 관련 키워드 포함 여부 확인
        if any(keyword in etf_name for keyword in self.domestic_keywords):
            return True
        
        # 한글이 포함되어 있으면서 해외 지역명이 없는 경우
        has_korean = bool(re.search(r'[가-힣]', etf_name))
        return has_korean

class TimeLLMDataProcessor:
    """Time-LLM 학습용 시계열 데이터 처리기"""
    
    def __init__(self):
        self.time_periods = [1, 3, 7, 15, 30]  # 다중 기간 분석
        self.scaler = MinMaxScaler()
        
    def collect_etf_historical_data(self, etf_list: List[ETFInfo], years: int = 20) -> Dict[str, Dict]:
        """ETF 과거 20년 데이터 수집"""
        logger.info(f"과거 {years}년간 ETF 데이터 수집 시작...")
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=years * 365)
        etf_data = {}
        
        for etf_info in etf_list:
            symbol = etf_info.symbol
            name = etf_info.name
            
            try:
                # Yahoo Finance에서 데이터 수집
                yahoo_symbol = f"{symbol}.KS"
                ticker = yf.Ticker(yahoo_symbol)
                df = ticker.history(start=start_date.strftime('%Y-%m-%d'), 
                                  end=end_date.strftime('%Y-%m-%d'))
                
                if not df.empty and len(df) > 365:
                    df = self._process_dataframe(df)
                    etf_data[symbol] = {
                        'name': name,
                        'data': df,
                        'sector': etf_info.sector,
                        'asset_type': etf_info.asset_type
                    }
                    logger.info(f"{name} ({symbol}) 데이터 수집 완료: {len(df)}일")
                
                time.sleep(0.1)  # API 호출 제한 대응
                
            except Exception as e:
                logger.warning(f"{symbol} 데이터 수집 실패: {e}")
                # 샘플 데이터 생성으로 대체
                df = self._generate_sample_data(start_date, end_date)
                df = self._process_dataframe(df)
                etf_data[symbol] = {
                    'name': name,
                    'data': df,
                    'sector': etf_info.sector,
                    'asset_type': etf_info.asset_type
                }
        
        logger.info(f"총 {len(etf_data)}개 ETF 데이터 수집 완료")
        return etf_data
    
    def _process_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """데이터프레임 처리 및 기술적 지표 계산"""
        # 컬럼명 표준화
        if 'Close' in df.columns:
            df = df.rename(columns={
                'Open': 'open', 'High': 'high', 'Low': 'low', 
                'Close': 'close', 'Volume': 'volume'
            })
        
        # 기술적 지표 계산
        df['returns'] = df['close'].pct_change()
        df['volatility'] = df['returns'].rolling(window=20).std() * np.sqrt(252)
        df['sma_20'] = df['close'].rolling(window=20).mean()
        df['sma_60'] = df['close'].rolling(window=60).mean()
        df['rsi'] = self._calculate_rsi(df['close'])
        df['bollinger_upper'], df['bollinger_lower'] = self._calculate_bollinger_bands(df['close'])
        df['macd'], df['macd_signal'] = self._calculate_macd(df['close'])
        
        # NaN 값 처리
        df = df.fillna(method='ffill').fillna(method='bfill')
        return df
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """RSI 계산"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_bollinger_bands(self, prices: pd.Series, period: int = 20, std_dev: int = 2):
        """볼린저 밴드 계산"""
        sma = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        upper_band = sma + (std * std_dev)
        lower_band = sma - (std * std_dev)
        return upper_band, lower_band
    
    def _calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
        """MACD 계산"""
        exp1 = prices.ewm(span=fast).mean()
        exp2 = prices.ewm(span=slow).mean()
        macd = exp1 - exp2
        macd_signal = macd.ewm(span=signal).mean()
        return macd, macd_signal
    
    def create_time_llm_training_data(self, etf_data: Dict) -> List[Dict]:
        """Time-LLM 학습용 JSON 데이터 생성"""
        logger.info("Time-LLM 학습 데이터 생성 시작...")
        training_data = []
        
        for symbol, info in etf_data.items():
            df = info['data']
            name = info['name']
            sector = info['sector']
            asset_type = info['asset_type']
            
            # 각 기간별로 시계열 패턴 생성
            for period in self.time_periods:
                sequences = self._create_sequences(df, period)
                
                for seq_data in sequences:
                    training_sample = {
                        "etf_symbol": symbol,
                        "etf_name": name,
                        "sector": sector,
                        "asset_type": asset_type,
                        "time_period": f"{period}day",
                        "input_sequence": seq_data['input'],
                        "target_prediction": seq_data['target'],
                        "market_context": seq_data['context'],
                        "instruction": self._generate_instruction(period, name, asset_type),
                        "response": self._generate_response(seq_data, period, name, sector)
                    }
                    training_data.append(training_sample)
        
        logger.info(f"총 {len(training_data)}개 학습 샘플 생성")
        return training_data

class HyperCLOVASkillTrainer:
    """HyperCLOVA 스킬트레이너 학습 관리자"""
    
    def __init__(self, api_key: str, app_id: str):
        self.api_key = api_key
        self.app_id = app_id
        self.base_url = "https://clovastudio.stream.ntruss.com"
        self.headers = {
            "X-NCP-CLOVASTUDIO-API-KEY": api_key,
            "X-NCP-APIGW-API-KEY": api_key,
            "Content-Type": "application/json",
            "Accept": "application/json"
        }
    
    def create_skillset(self, skillset_name: str) -> Optional[str]:
        """스킬셋 생성"""
        logger.info(f"스킬셋 생성: {skillset_name}")
        
        payload = {
            "name": skillset_name,
            "description": "국내 자본시장 활성화를 위한 퇴직연금 ETF Time-LLM 분석 스킬셋",
            "serviceField": "금융/투자",
            "responseFormat": "구조화된 퇴직연금 ETF 투자 분석 및 추천 결과"
        }
        
        try:
            response = requests.post(
                f"{self.base_url}/testapp/v1/api-tools/skillsets",
                headers=self.headers,
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                skillset_id = result.get('skillsetId')
                logger.info(f"스킬셋 생성 완료: {skillset_id}")
                return skillset_id
            else:
                logger.error(f"스킬셋 생성 실패: {response.status_code}")
                return None
                
        except Exception as e:
            logger.error(f"스킬셋 생성 오류: {e}")
            return None
    
    def create_skill(self, skillset_id: str, skill_name: str) -> Optional[str]:
        """ETF Time-LLM 분석 스킬 생성"""
        api_spec = {
            "openapi": "3.0.0",
            "info": {
                "title": "퇴직연금 ETF Time-LLM 분석 API",
                "version": "1.0.0",
                "description": "국내 ETF 시계열 데이터 기반 Time-LLM 분석 및 퇴직연금 투자 추천"
            },
            "paths": {
                "/analyze": {
                    "post": {
                        "summary": "ETF 시계열 분석 및 퇴직연금 투자 추천",
                        "requestBody": {
                            "required": True,
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "object",
                                        "properties": {
                                            "etf_symbol": {"type": "string"},
                                            "analysis_period": {"type": "integer", "enum": [1, 3, 7, 15, 30]},
                                            "investment_amount": {"type": "number"},
                                            "risk_tolerance": {"type": "string", "enum": ["낮음", "중간", "높음"]}
                                        }
                                    }
                                }
                            }
                        },
                        "responses": {
                            "200": {
                                "description": "성공적인 분석 결과",
                                "content": {
                                    "application/json": {
                                        "schema": {
                                            "type": "object",
                                            "properties": {
                                                "recommendation": {"type": "string"},
                                                "expected_return": {"type": "number"},
                                                "risk_level": {"type": "string"},
                                                "analysis": {"type": "string"}
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        
        payload = {
            "skillsetId": skillset_id,
            "name": skill_name,
            "description": "Time-LLM 기반 퇴직연금 ETF 투자 분석 및 추천 스킬",
            "apiSpec": json.dumps(api_spec)
        }
        
        try:
            response = requests.post(
                f"{self.base_url}/testapp/v1/api-tools/skills",
                headers=self.headers,
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                skill_id = result.get('skillId')
                logger.info(f"스킬 생성 완료: {skill_id}")
                return skill_id
            else:
                logger.error(f"스킬 생성 실패: {response.status_code}")
                return None
                
        except Exception as e:
            logger.error(f"스킬 생성 오류: {e}")
            return None
    
    def upload_training_data(self, skill_id: str, training_data: List[Dict]) -> bool:
        """학습 데이터 업로드"""
        logger.info(f"학습 데이터 업로드: {len(training_data)}개 샘플")
        
        batch_size = 50
        total_batches = (len(training_data) + batch_size - 1) // batch_size
        
        for i in range(0, len(training_data), batch_size):
            batch = training_data[i:i+batch_size]
            batch_num = i // batch_size + 1
            
            # 학습 데이터 형식 변환
            formatted_batch = []
            for sample in batch:
                formatted_sample = {
                    "input": sample["instruction"],
                    "output": sample["response"],
                    "metadata": {
                        "etf_symbol": sample["etf_symbol"],
                        "time_period": sample["time_period"],
                        "market_context": sample["market_context"]
                    }
                }
                formatted_batch.append(formatted_sample)
            
            payload = {
                "skillId": skill_id,
                "trainingData": formatted_batch,
                "batchNumber": batch_num,
                "totalBatches": total_batches
            }
            
            try:
                response = requests.post(
                    f"{self.base_url}/testapp/v1/api-tools/skills/{skill_id}/training-data",
                    headers=self.headers,
                    json=payload,
                    timeout=60
                )
                
                if response.status_code not in [200, 201]:
                    logger.error(f"배치 {batch_num} 업로드 실패: {response.status_code}")
                    return False
                
                logger.info(f"배치 {batch_num}/{total_batches} 업로드 완료")
                time.sleep(1)
                
            except Exception as e:
                logger.error(f"배치 {batch_num} 업로드 오류: {e}")
                return False
        
        return True
    
    def train_model(self, skillset_id: str) -> Optional[str]:
        """HyperCLOVA X 모델 학습"""
        logger.info("HyperCLOVA X 모델 학습 시작...")
        
        payload = {
            "skillsetId": skillset_id,
            "modelType": "HyperCLOVA-X",  # 최고 성능 모델
            "trainingConfig": {
                "epochs": 15,
                "learningRate": 0.0001,
                "batchSize": 16,
                "warmupSteps": 100,
                "weightDecay": 0.01,
                "maxSequenceLength": 2048
            },
            "optimizationConfig": {
                "enableMixedPrecision": True,
                "gradientAccumulation": 4,
                "scheduler": "cosine",
                "earlyStopping": True
            },
            "validationSplit": 0.2
        }
        
        try:
            response = requests.post(
                f"{self.base_url}/testapp/v1/api-tools/skillsets/{skillset_id}/train",
                headers=self.headers,
                json=payload,
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                training_job_id = result.get('jobId')
                logger.info(f"학습 작업 시작: {training_job_id}")
                return training_job_id
            else:
                logger.error(f"학습 시작 실패: {response.status_code}")
                return None
                
        except Exception as e:
            logger.error(f"학습 시작 오류: {e}")
            return None
    
    def wait_for_training_completion(self, job_id: str, max_wait_time: int = 3600) -> Dict:
        """학습 완료 대기"""
        logger.info(f"학습 완료 대기 (최대 {max_wait_time//60}분)")
        
        start_time = time.time()
        check_interval = 30
        
        while time.time() - start_time < max_wait_time:
            try:
                response = requests.get(
                    f"{self.base_url}/testapp/v1/api-tools/training-jobs/{job_id}",
                    headers=self.headers,
                    timeout=30
                )
                
                if response.status_code == 200:
                    status = response.json()
                    
                    if status.get('status') == 'completed':
                        logger.info("모델 학습 완료!")
                        return status
                    elif status.get('status') == 'failed':
                        logger.error("모델 학습 실패")
                        return status
                    else:
                        progress = status.get('progress', 0)
                        logger.info(f"학습 진행률: {progress}%")
                        time.sleep(check_interval)
                
            except Exception as e:
                logger.error(f"상태 확인 오류: {e}")
                time.sleep(check_interval)
        
        return {"status": "timeout"}

class RetirementPensionAgent:
    """퇴직연금 ETF 투자 에이전트"""
    
    def __init__(self, trained_model_id: str, api_key: str):
        self.model_id = trained_model_id
        self.api_key = api_key
        self.base_url = "https://clovastudio.stream.ntruss.com"
        self.headers = {
            "X-NCP-CLOVASTUDIO-API-KEY": api_key,
            "X-NCP-APIGW-API-KEY": api_key,
            "Content-Type": "application/json"
        }
        
        # 퇴직연금 투자 규정
        self.investment_rules = {
            "max_risk_asset_ratio": 0.7,  # 위험자산 최대 70%
            "min_safe_asset_ratio": 0.3,  # 안전자산 최소 30%
            "max_single_asset_ratio": 0.1,  # 동일 종목 최대 10%
            "rebalancing_threshold": 0.05  # 리밸런싱 임계값 5%
        }
    
    def analyze_portfolio(self, etf_symbols: List[str], 
                         investment_amount: float,
                         risk_tolerance: str = "중간") -> Dict:
        """포트폴리오 분석 및 최적화"""
        logger.info(f"포트폴리오 분석: {etf_symbols}")
        
        recommendations = []
        for symbol in etf_symbols:
            analysis = self._analyze_single_etf(symbol, risk_tolerance)
            recommendations.append(analysis)
        
        # 포트폴리오 최적화
        optimized_weights = self._optimize_portfolio_weights(
            recommendations, investment_amount, risk_tolerance
        )
        
        # 퇴직연금 규정 준수 확인
        compliance_check = self._check_retirement_compliance(optimized_weights, recommendations)
        
        return {
            "analysis_summary": {
                "total_investment": investment_amount,
                "risk_tolerance": risk_tolerance,
                "analysis_date": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            },
            "individual_analysis": recommendations,
            "optimized_allocation": optimized_weights,
            "compliance_check": compliance_check,
            "recommendation_summary": self._generate_portfolio_summary(recommendations, optimized_weights)
        }
    
    def _analyze_single_etf(self, symbol: str, risk_tolerance: str) -> Dict:
        """개별 ETF Time-LLM 분석"""
        prompt = f"""
퇴직연금 투자 관점에서 {symbol} ETF를 분석해주세요.
위험 허용도: {risk_tolerance}
분석 기준: Time-LLM 다중 기간 분석 (1일, 3일, 7일, 15일, 30일)

분석 요청사항:
1. 시계열 패턴 분석 및 예상 수익률
2. 위험도 평가 및 변동성 분석
3. 퇴직연금 포트폴리오 적합성 (100점 만점)
4. 권장 투자 비중 및 근거
5. 국내 자본시장 기여도 평가
        """
        
        payload = {
            "messages": [
                {
                    "role": "system",
                    "content": "당신은 Time-LLM 전문가이며 퇴직연금 투자 어드바이저입니다. 국내 자본시장 활성화를 목표로 하며, 해외 자금 유출을 방지하는 투자 전략을 제시합니다."
                },
                {"role": "user", "content": prompt}
            ],
            "topP": 0.8,
            "maxTokens": 1500,
            "temperature": 0.3,
            "repeatPenalty": 1.1
        }
        
        try:
            response = requests.post(
                f"{self.base_url}/testapp/v1/chat-completions/{self.model_id}",
                headers=self.headers,
                json=payload,
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                analysis_text = result['result']['message']['content']
                return self._parse_analysis_result(symbol, analysis_text)
            else:
                return self._generate_fallback_analysis(symbol)
                
        except Exception as e:
            logger.error(f"ETF 분석 오류: {e}")
            return self._generate_fallback_analysis(symbol)

def main():
    """전체 시스템 실행"""
    logger.info("Time-LLM 퇴직연금 에이전트 구축 시작")
    
    # 설정값
    CLOVA_API_KEY = "your_clova_api_key_here"
    CLOVA_APP_ID = "your_app_id_here"
    
    try:
        # 1단계: 국내 ETF 데이터 수집
        etf_collector = DomesticETFCollector()
        domestic_etfs = etf_collector.get_domestic_etf_list()
        
        # 2단계: 시계열 데이터 처리
        data_processor = TimeLLMDataProcessor()
        etf_historical_data = data_processor.collect_etf_historical_data(domestic_etfs, years=20)
        
        # 3단계: Time-LLM 학습 데이터 생성
        training_data = data_processor.create_time_llm_training_data(etf_historical_data)
        
        # JSON 파일로 저장
        with open('time_llm_retirement_pension_training_data.json', 'w', encoding='utf-8') as f:
            json.dump(training_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"학습 데이터 저장 완료: {len(training_data)}개 샘플")
        
        # 4단계: HyperCLOVA 스킬트레이너 학습
        skill_trainer = HyperCLOVASkillTrainer(CLOVA_API_KEY, CLOVA_APP_ID)
        
        # 스킬셋 및 스킬 생성
        skillset_id = skill_trainer.create_skillset("retirement_pension_time_llm")
        skill_id = skill_trainer.create_skill(skillset_id, "etf_time_llm_analysis")
        
        # 학습 데이터 업로드 및 모델 학습
        skill_trainer.upload_training_data(skill_id, training_data)
        training_job_id = skill_trainer.train_model(skillset_id)
        
        # 학습 완료 대기
        training_result = skill_trainer.wait_for_training_completion(training_job_id)
        
        if training_result.get('status') == 'completed':
            trained_model_id = training_result.get('modelId')
            
            # 5단계: 퇴직연금 에이전트 구축
            pension_agent = RetirementPensionAgent(trained_model_id, CLOVA_API_KEY)
            
            # 샘플 포트폴리오 분석
            sample_etfs = [etf.symbol for etf in domestic_etfs[:5]]
            analysis_result = pension_agent.analyze_portfolio(
                etf_symbols=sample_etfs,
                investment_amount=10000000,  # 1천만원
                risk_tolerance="중간"
            )
            
            logger.info("퇴직연금 에이전트 구축 완료")
            return analysis_result
        
    except Exception as e:
        logger.error(f"시스템 실행 오류: {e}")
        return None

# 시스템 실행
if __name__ == "__main__":
    result = main()
