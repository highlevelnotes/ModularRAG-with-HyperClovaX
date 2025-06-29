# src/langchain_integration/time_llm_data_processor.py
import json
import logging
import os
from typing import List, Dict, Any
from datetime import datetime
import pandas as pd
import numpy as np
from dotenv import load_dotenv

load_dotenv()

class TimeLLMDataProcessor:
    """LangChain용 Time-LLM 데이터 처리기"""
    
    def __init__(self):
        self.logger = self._setup_logger()
    
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
    
    def prepare_langchain_tuning_data(self, json_file_path: str) -> List[Dict[str, Any]]:
        """JSON 파일을 LangChain 튜닝 형식으로 변환"""
        try:
            self.logger.info(f"LangChain 튜닝 데이터 준비 중: {json_file_path}")
            
            with open(json_file_path, 'r', encoding='utf-8') as f:
                time_llm_data = json.load(f)
            
            # LangChain ChatClovaX 형식으로 변환
            langchain_data = []
            for sample in time_llm_data:
                # 메시지 형식으로 구성
                formatted_sample = {
                    "messages": [
                        {
                            "role": "user",
                            "content": sample["input"]
                        },
                        {
                            "role": "assistant", 
                            "content": sample["output"]
                        }
                    ],
                    "metadata": {
                        "etf_code": sample.get("etf_code", ""),
                        "etf_name": sample.get("etf_name", ""),
                        "sequence_index": sample.get("sequence_index", 0),
                        "scaler_info": sample.get("scaler_info", {}),
                        "sample_metadata": sample.get("metadata", {})
                    }
                }
                langchain_data.append(formatted_sample)
            
            self.logger.info(f"LangChain 튜닝 데이터 준비 완료: {len(langchain_data)}개 샘플")
            return langchain_data
            
        except Exception as e:
            self.logger.error(f"LangChain 튜닝 데이터 준비 오류: {str(e)}")
            raise
    
    def save_langchain_tuning_file(self, langchain_data: List[Dict], output_path: str):
        """LangChain 튜닝 데이터를 JSONL 형식으로 저장"""
        try:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                for sample in langchain_data:
                    f.write(json.dumps(sample, ensure_ascii=False) + '\n')
            
            self.logger.info(f"LangChain 튜닝 파일 저장 완료: {output_path}")
            
        except Exception as e:
            self.logger.error(f"LangChain 튜닝 파일 저장 오류: {str(e)}")
            raise
