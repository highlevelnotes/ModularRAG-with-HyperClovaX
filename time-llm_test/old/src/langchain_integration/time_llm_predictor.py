# src/langchain_integration/time_llm_predictor.py
import os
import json
import logging
import numpy as np
import time
from typing import List, Dict, Any, Optional
from datetime import datetime
from dotenv import load_dotenv

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import BaseOutputParser
from langchain_community.chat_models import ChatClovaX
from langchain.chains import LLMChain

load_dotenv()

class TimeSeriesOutputParser(BaseOutputParser):
    """시계열 예측 결과 파싱"""
    
    def parse(self, text: str) -> List[float]:
        """텍스트에서 숫자 시퀀스 추출"""
        try:
            numbers = []
            for token in text.split():
                try:
                    number = float(token)
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

class LangChainTimeLLMPredictor:
    """LangChain 기반 Time-LLM 예측기"""
    
    def __init__(self, model_id: str = None, task_id: str = None):
        self.api_key = os.getenv("NCP_CLOVASTUDIO_API_KEY")
        self.apigw_api_key = os.getenv("NCP_APIGW_API_KEY")
        
        if not self.api_key:
            raise ValueError("NCP_CLOVASTUDIO_API_KEY가 설정되지 않았습니다. .env 파일을 확인하세요.")
        if not self.apigw_api_key:
            raise ValueError("NCP_APIGW_API_KEY가 설정되지 않았습니다. .env 파일을 확인하세요.")
        
        self.model_id = model_id
        self.task_id = task_id
        self.logger = self._setup_logger()
        
        # ChatClovaX 모델 초기화
        self._initialize_model()
        
        # 프롬프트 템플릿 설정
        self._setup_prompt_template()
        
        # 출력 파서 설정
        self.output_parser = TimeSeriesOutputParser()
        
        # LangChain 체인 구성
        self._setup_chain()
    
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
    
    def _initialize_model(self):
        """ChatClovaX 모델 초기화 (새로운 Bearer 토큰 방식)"""
        try:
            model_params = {
                "clovastudio_api_key": self.api_key,
                # apigw_api_key 제거 (더 이상 필요 없음)
                "max_tokens": 512,
                "temperature": 0.3,
                "top_p": 0.6,
                "repeat_penalty": 1.2,
                "include_ai_filters": False,
                "service_app": False
            }
            
            # 튜닝된 모델이 있는 경우
            if self.model_id:
                model_params["model"] = self.model_id
                self.logger.info(f"튜닝된 모델 사용: {self.model_id}")
            elif self.task_id:
                model_params["task_id"] = self.task_id
                self.logger.info(f"튜닝 작업 ID 사용: {self.task_id}")
            else:
                model_params["model"] = "HCX-003"  # 기본 모델
                self.logger.info("기본 HCX-003 모델 사용")
            
            self.chat_model = ChatClovaX(**model_params)
            
        except Exception as e:
            self.logger.error(f"모델 초기화 오류: {str(e)}")
            raise
    
    def _setup_prompt_template(self):
        """프롬프트 템플릿 설정"""
        self.prompt_template = ChatPromptTemplate.from_messages([
            ("system", """당신은 한국 ETF 가격 예측 전문가입니다. 
주어진 정규화된 가격 시계열 데이터를 분석하여 미래 가격을 정확히 예측하세요.
출력은 반드시 공백으로 구분된 10개의 숫자만 제공하세요."""),
            ("human", "{etf_prediction_prompt}")
        ])
    
    def _setup_chain(self):
        """LangChain 체인 구성"""
        self.prediction_chain = self.prompt_template | self.chat_model | self.output_parser
    
    def normalize_sequence(self, sequence: List[float]) -> tuple:
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
    
    def denormalize_sequence(self, normalized_sequence: List[float], scaler_info: Dict) -> List[float]:
        """시퀀스 역정규화"""
        normalized = np.array(normalized_sequence)
        min_val = scaler_info['min_val']
        max_val = scaler_info['max_val']
        
        denormalized = normalized * (max_val - min_val) + min_val
        return denormalized.tolist()
    
    def predict_etf_prices(self, etf_name: str, price_sequence: List[float]) -> Dict[str, Any]:
        """ETF 가격 예측"""
        try:
            # 입력 데이터 정규화
            normalized_sequence, scaler_info = self.normalize_sequence(price_sequence)
            
            # 텍스트 형식으로 변환
            input_text = ' '.join([f"{x:.6f}" for x in normalized_sequence])
            
            # 프롬프트 구성
            prediction_prompt = f"""다음은 {etf_name} ETF의 정규화된 가격 시계열 데이터입니다.
과거 30일간의 데이터를 바탕으로 향후 10일의 가격을 예측하세요.

입력 데이터: {input_text}

예상 출력:"""
            
            # LangChain 체인 실행
            predicted_normalized = self.prediction_chain.invoke({
                "etf_prediction_prompt": prediction_prompt
            })
            
            # 역정규화
            predicted_prices = self.denormalize_sequence(predicted_normalized, scaler_info)
            
            result = {
                "etf_name": etf_name,
                "input_sequence": price_sequence,
                "normalized_input": normalized_sequence.tolist(),
                "predicted_normalized": predicted_normalized,
                "predicted_prices": predicted_prices,
                "scaler_info": scaler_info,
                "prediction_timestamp": datetime.now().isoformat()
            }
            
            self.logger.info(f"{etf_name} 예측 완료")
            return result
            
        except Exception as e:
            self.logger.error(f"{etf_name} 예측 오류: {str(e)}")
            raise
    
    def batch_predict(self, test_data_path: str, num_samples: int = 10) -> Dict[str, Any]:
        """배치 예측 실행"""
        try:
            self.logger.info(f"배치 예측 시작: {num_samples}개 샘플")
            
            with open(test_data_path, 'r', encoding='utf-8') as f:
                test_data = json.load(f)
            
            results = []
            
            for i, sample in enumerate(test_data[:num_samples]):
                try:
                    etf_code = sample.get('etf_code', 'Unknown')
                    
                    # 입력 시퀀스 추출
                    input_content = sample['input']
                    input_sequence_text = input_content.split('입력 데이터: ')[1].split('\n')[0]
                    input_sequence = [float(x) for x in input_sequence_text.split()]
                    
                    # 실제 출력
                    actual_output = [float(x) for x in sample['output'].split()]
                    
                    # 예측 실행
                    prediction_result = self.predict_etf_prices(etf_code, input_sequence)
                    predicted_output = prediction_result['predicted_normalized']
                    
                    # 결과 저장
                    result = {
                        'sample_id': i,
                        'etf_code': etf_code,
                        'actual': actual_output,
                        'predicted': predicted_output,
                        'mse': self._calculate_mse(actual_output, predicted_output),
                        'full_prediction': prediction_result
                    }
                    results.append(result)
                    
                    self.logger.info(f"샘플 {i+1}/{num_samples} 완료 - MSE: {result['mse']:.6f}")
                    
                    # API 제한을 위한 대기
                    time.sleep(1)
                    
                except Exception as e:
                    self.logger.error(f"샘플 {i} 예측 실패: {str(e)}")
                    continue
            
            # 전체 성능 계산
            avg_mse = np.mean([r['mse'] for r in results])
            
            batch_summary = {
                'total_samples': len(results),
                'average_mse': avg_mse,
                'model_info': {
                    'model_id': self.model_id,
                    'task_id': self.task_id
                },
                'results': results,
                'timestamp': datetime.now().isoformat()
            }
            
            self.logger.info(f"배치 예측 완료 - 평균 MSE: {avg_mse:.6f}")
            
            return batch_summary
            
        except Exception as e:
            self.logger.error(f"배치 예측 오류: {str(e)}")
            raise
    
    def _calculate_mse(self, actual: List[float], predicted: List[float]) -> float:
        """평균 제곱 오차 계산"""
        actual = np.array(actual)
        predicted = np.array(predicted[:len(actual)])
        return float(np.mean((actual - predicted) ** 2))
