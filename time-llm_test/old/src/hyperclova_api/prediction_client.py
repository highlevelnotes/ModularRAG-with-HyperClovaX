# src/hyperclova_api/prediction_client.py
import os
import json
import requests
import logging
import numpy as np
from typing import List, Dict, Any
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

class HyperClovaXPredictionClient:
    """순수 HyperClova X API 기반 예측 클라이언트"""
    
    def __init__(self, model_id: str = None):
        self.api_key = os.getenv("CLOVASTUDIO_API_KEY")
        
        if not self.api_key:
            raise ValueError("CLOVASTUDIO_API_KEY가 설정되지 않았습니다.")
        
        self.model_id = model_id or "HCX-003"  # 기본 모델 또는 튜닝된 모델
        self.base_url = "https://clovastudio.stream.ntruss.com"
        self.logger = self._setup_logger()
        
        self.logger.info(f"HyperClova X 예측 클라이언트 초기화 완료 (모델: {self.model_id})")
    
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
    
    def _get_headers(self, request_id: str = None) -> Dict[str, str]:
        """API 요청 헤더 생성"""
        if request_id is None:
            request_id = f"predict-{int(time.time())}"
        
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "X-NCP-CLOVASTUDIO-REQUEST-ID": request_id
        }
    
    def predict_etf_prices(self, etf_name: str, price_sequence: List[float]) -> Dict[str, Any]:
        """ETF 가격 예측"""
        try:
            # 입력 데이터 정규화
            normalized_sequence, scaler_info = self._normalize_sequence(price_sequence)
            
            # 텍스트 형식으로 변환
            input_text = ' '.join([f"{x:.6f}" for x in normalized_sequence])
            
            # 프롬프트 구성
            prompt = f"""다음은 {etf_name} ETF의 정규화된 가격 시계열 데이터입니다.
과거 30일간의 데이터를 바탕으로 향후 10일의 가격을 예측하세요.

입력 데이터: {input_text}

예상 출력:"""
            
            # API 요청 데이터 구성[4]
            request_data = {
                "messages": [
                    {
                        "role": "system",
                        "content": "당신은 한국 ETF 가격 예측 전문가입니다. 주어진 정규화된 가격 시계열 데이터를 분석하여 미래 가격을 정확히 예측하세요. 출력은 반드시 공백으로 구분된 10개의 숫자만 제공하세요."
                    },
                    {
                        "role": "user", 
                        "content": prompt
                    }
                ],
                "topP": 0.8,
                "topK": 0,
                "maxTokens": 256,
                "temperature": 0.3,
                "repeatPenalty": 1.2,
                "stopBefore": [],
                "includeAiFilters": False
            }
            
            # API 호출
            api_url = f"{self.base_url}/testapp/v1/chat-completions/{self.model_id}"
            
            response = requests.post(
                api_url,
                headers=self._get_headers(),
                json=request_data,
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                prediction_text = result["result"]["message"]["content"]
                
                # 예측 결과 파싱
                predicted_normalized = self._parse_prediction(prediction_text)
                predicted_prices = self._denormalize_sequence(predicted_normalized, scaler_info)
                
                return {
                    "etf_name": etf_name,
                    "input_sequence": price_sequence,
                    "predicted_prices": predicted_prices,
                    "raw_prediction": prediction_text,
                    "scaler_info": scaler_info,
                    "timestamp": datetime.now().isoformat()
                }
            else:
                raise Exception(f"예측 요청 실패: {response.status_code} - {response.text}")
                
        except Exception as e:
            self.logger.error(f"예측 오류: {str(e)}")
            raise
    
    def _normalize_sequence(self, sequence: List[float]) -> tuple:
        """시퀀스 정규화"""
        sequence = np.array(sequence)
        min_val = sequence.min()
        max_val = sequence.max()
        
        if max_val > min_val:
            normalized = (sequence - min_val) / (max_val - min_val)
        else:
            normalized = np.zeros_like(sequence)
        
        scaler_info = {'min_val': float(min_val), 'max_val': float(max_val)}
        return normalized, scaler_info
    
    def _denormalize_sequence(self, normalized_sequence: List[float], scaler_info: Dict) -> List[float]:
        """시퀀스 역정규화"""
        normalized = np.array(normalized_sequence)
        min_val = scaler_info['min_val']
        max_val = scaler_info['max_val']
        
        denormalized = normalized * (max_val - min_val) + min_val
        return denormalized.tolist()
    
    def _parse_prediction(self, prediction_text: str) -> List[float]:
        """예측 텍스트에서 숫자 추출"""
        try:
            import re
            
            # 숫자 패턴 찾기
            number_pattern = r'[-+]?(?:\d*\.\d+|\d+)'
            matches = re.findall(number_pattern, prediction_text)
            
            numbers = []
            for match in matches:
                try:
                    number = float(match)
                    if 0 <= number <= 1:  # 정규화된 값 범위
                        numbers.append(number)
                except ValueError:
                    continue
            
            # 정확히 10개의 예측값이 필요
            if len(numbers) >= 10:
                return numbers[:10]
            else:
                # 부족한 경우 마지막 값으로 채움
                while len(numbers) < 10:
                    numbers.append(numbers[-1] if numbers else 0.5)
                return numbers
                
        except Exception:
            return [0.5] * 10  # 기본값 반환
