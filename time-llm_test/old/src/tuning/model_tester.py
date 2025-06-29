# src/tuning/model_tester.py
import json
import requests
import logging
from typing import Dict, List
import numpy as np
import time

class TimeLLMTester:
    """튜닝된 Time-LLM 모델 테스트"""
    
    def __init__(self, api_key: str, api_key_primary: str, model_id: str):
        self.api_key = api_key
        self.api_key_primary = api_key_primary
        self.model_id = model_id
        self.base_url = "https://clovastudio.stream.ntruss.com"
        self.logger = logging.getLogger(__name__)
    
    def _get_headers(self) -> Dict[str, str]:
        """API 요청 헤더 생성"""
        return {
            "Content-Type": "application/json; charset=utf-8",
            "X-NCP-CLOVASTUDIO-API-KEY": self.api_key,
            "X-NCP-APIGW-API-KEY": self.api_key_primary,
            "X-NCP-CLOVASTUDIO-REQUEST-ID": f"test-{int(time.time())}"
        }
    
    def predict_etf_prices(self, etf_name: str, price_sequence: List[float]) -> List[float]:
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
            
            # API 요청
            request_data = {
                "messages": [
                    {"role": "user", "content": prompt}
                ],
                "topP": 0.6,
                "topK": 0,
                "maxTokens": 512,
                "temperature": 0.3,
                "repeatPenalty": 1.2,
                "stopBefore": [],
                "includeAiFilters": False
            }
            
            response = requests.post(
                f"{self.base_url}/testapp/v1/chat-completions/{self.model_id}",
                headers=self._get_headers(),
                json=request_data
            )
            
            if response.status_code == 200:
                result = response.json()
                prediction_text = result["result"]["message"]["content"]
                
                # 예측 결과 파싱 및 역정규화
                predicted_normalized = self._parse_prediction(prediction_text)
                predicted_prices = self._denormalize_sequence(predicted_normalized, scaler_info)
                
                return predicted_prices
            else:
                raise Exception(f"예측 요청 실패: {response.status_code} - {response.text}")
                
        except Exception as e:
            self.logger.error(f"가격 예측 오류: {str(e)}")
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
            # 공백으로 구분된 숫자 추출
            numbers = []
            for token in prediction_text.split():
                try:
                    number = float(token)
                    if 0 <= number <= 1:  # 정규화된 값 범위 확인
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
                
        except Exception as e:
            self.logger.warning(f"예측 파싱 오류: {str(e)}")
            return [0.5] * 10  # 기본값 반환
    
    def batch_test(self, test_data_path: str, num_samples: int = 10) -> Dict:
        """배치 테스트 실행"""
        try:
            self.logger.info(f"배치 테스트 시작: {num_samples}개 샘플")
            
            with open(test_data_path, 'r', encoding='utf-8') as f:
                test_data = json.load(f)
            
            results = []
            
            for i, sample in enumerate(test_data[:num_samples]):
                try:
                    etf_code = sample.get('etf_code', 'Unknown')
                    
                    # 입력 시퀀스 추출 (프롬프트에서)
                    input_content = sample['input']
                    input_sequence_text = input_content.split('입력 데이터: ')[1].split('\n')[0]
                    input_sequence = [float(x) for x in input_sequence_text.split()]
                    
                    # 실제 출력
                    actual_output = [float(x) for x in sample['output'].split()]
                    
                    # 예측 실행
                    predicted_output = self.predict_etf_prices(etf_code, input_sequence)
                    
                    # 결과 저장
                    result = {
                        'sample_id': i,
                        'etf_code': etf_code,
                        'actual': actual_output,
                        'predicted': predicted_output,
                        'mse': self._calculate_mse(actual_output, predicted_output)
                    }
                    results.append(result)
                    
                    self.logger.info(f"샘플 {i+1}/{num_samples} 완료 - MSE: {result['mse']:.6f}")
                    
                except Exception as e:
                    self.logger.error(f"샘플 {i} 테스트 실패: {str(e)}")
                    continue
            
            # 전체 성능 계산
            avg_mse = np.mean([r['mse'] for r in results])
            
            test_summary = {
                'total_samples': len(results),
                'average_mse': avg_mse,
                'results': results
            }
            
            self.logger.info(f"배치 테스트 완료 - 평균 MSE: {avg_mse:.6f}")
            
            return test_summary
            
        except Exception as e:
            self.logger.error(f"배치 테스트 오류: {str(e)}")
            raise
    
    def _calculate_mse(self, actual: List[float], predicted: List[float]) -> float:
        """평균 제곱 오차 계산"""
        actual = np.array(actual)
        predicted = np.array(predicted[:len(actual)])  # 길이 맞추기
        return float(np.mean((actual - predicted) ** 2))
