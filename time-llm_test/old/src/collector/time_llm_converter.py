# src/collector/time_llm_converter.py
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
import json
import logging
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime

class TimeLLMConverter:
    """Time-LLM 형식 변환기"""
    
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def normalize_price_series(self, prices: np.ndarray, method: str = 'minmax') -> Tuple[np.ndarray, Dict]:
        """가격 시계열 정규화"""
        if method == 'minmax':
            scaler = MinMaxScaler()
            normalized = scaler.fit_transform(prices.reshape(-1, 1)).flatten()
            scaler_info = {
                'method': 'minmax',
                'min_val': float(prices.min()),
                'max_val': float(prices.max())
            }
        elif method == 'zscore':
            mean_val = prices.mean()
            std_val = prices.std()
            normalized = (prices - mean_val) / std_val
            scaler_info = {
                'method': 'zscore',
                'mean_val': float(mean_val),
                'std_val': float(std_val)
            }
        else:
            raise ValueError(f"지원하지 않는 정규화 방법: {method}")
        
        return normalized, scaler_info
    
    def create_patches(self, prices: np.ndarray, patch_length: int, prediction_days: int) -> List[Dict]:
        """시계열을 패치로 분할하여 Time-LLM 형식 생성"""
        patches = []
        
        for i in range(len(prices) - patch_length - prediction_days + 1):
            # 입력 시퀀스 (과거 데이터)
            input_sequence = prices[i:i+patch_length]
            # 출력 시퀀스 (미래 데이터)
            output_sequence = prices[i+patch_length:i+patch_length+prediction_days]
            
            # 시퀀스 정규화
            combined_sequence = np.concatenate([input_sequence, output_sequence])
            normalized_combined, scaler_info = self.normalize_price_series(combined_sequence)
            
            norm_input = normalized_combined[:patch_length]
            norm_output = normalized_combined[patch_length:]
            
            # 텍스트 형식으로 변환
            input_text = ' '.join([f"{x:.6f}" for x in norm_input])
            output_text = ' '.join([f"{x:.6f}" for x in norm_output])
            
            patch = {
                'input_sequence': input_text,
                'output_sequence': output_text,
                'scaler_info': scaler_info,
                'sequence_index': i,
                'input_length': patch_length,
                'output_length': prediction_days
            }
            
            patches.append(patch)
        
        return patches
    
    def create_prompt_as_prefix(self, etf_name: str, patch_data: Dict) -> str:
        """Prompt-as-Prefix (PaP) 방식으로 프롬프트 생성"""
        base_prompt = f"""다음은 {etf_name} ETF의 정규화된 가격 시계열 데이터입니다.
과거 {patch_data['input_length']}일간의 데이터를 바탕으로 향후 {patch_data['output_length']}일의 가격을 예측하세요.

입력 데이터: {patch_data['input_sequence']}

예상 출력:"""
        
        return base_prompt
    
    def convert_to_time_llm_format(self, price_data: Dict[str, pd.DataFrame]) -> List[Dict]:
        """전체 가격 데이터를 Time-LLM 형식으로 변환"""
        self.logger.info("Time-LLM 형식으로 데이터 변환 시작...")
        
        time_llm_data = []
        
        for etf_code, df in price_data.items():
            try:
                etf_name = etf_code  # 실제로는 ETF 이름을 가져와야 함
                
                # 종가 데이터 추출
                prices = df['close_price'].values
                dates = df['date'].values
                
                if len(prices) < self.config.patch_length + self.config.prediction_days:
                    self.logger.warning(f"{etf_code}: 데이터 길이 부족")
                    continue
                
                # 패치 생성
                patches = self.create_patches(
                    prices, 
                    self.config.patch_length, 
                    self.config.prediction_days
                )
                
                # Time-LLM 형식으로 변환
                for patch in patches:
                    # HyperClova X 파인튜닝 형식
                    prompt = self.create_prompt_as_prefix(etf_name, patch)
                    
                    time_llm_sample = {
                        'input': prompt,
                        'output': patch['output_sequence'],
                        'etf_code': etf_code,
                        'etf_name': etf_name,
                        'sequence_index': patch['sequence_index'],
                        'scaler_info': patch['scaler_info'],
                        'metadata': {
                            'input_length': patch['input_length'],
                            'output_length': patch['output_length'],
                            'start_date': str(dates[patch['sequence_index']]),
                            'end_date': str(dates[patch['sequence_index'] + patch['input_length'] - 1])
                        }
                    }
                    
                    time_llm_data.append(time_llm_sample)
                
                self.logger.info(f"{etf_code}: {len(patches)}개 샘플 생성")
                
            except Exception as e:
                self.logger.error(f"{etf_code} 변환 실패: {str(e)}")
                continue
        
        self.logger.info(f"Time-LLM 형식 변환 완료: 총 {len(time_llm_data)}개 샘플")
        
        return time_llm_data
    
    def create_hyperclova_x_format(self, time_llm_data: List[Dict]) -> List[Dict]:
        """HyperClova X 파인튜닝 형식으로 변환"""
        self.logger.info("HyperClova X 파인튜닝 형식으로 변환 중...")
        
        hyperclova_data = []
        
        for sample in time_llm_data:
            hyperclova_sample = {
                'input': sample['input'],
                'output': sample['output']
            }
            hyperclova_data.append(hyperclova_sample)
        
        self.logger.info(f"HyperClova X 형식 변환 완료: {len(hyperclova_data)}개 샘플")
        
        return hyperclova_data
