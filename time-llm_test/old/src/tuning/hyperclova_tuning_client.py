# src/tuning/hyperclova_tuning_client.py
import json
import requests
import time
import logging
from typing import Dict, List, Optional
from datetime import datetime
import os

class HyperClovaXTuningClient:
    """HyperClova X 파인튜닝 클라이언트"""
    
    def __init__(self, api_key: str, api_key_primary: str, request_id: str = None):
        self.api_key = api_key
        self.api_key_primary = api_key_primary
        self.request_id = request_id or f"time-llm-{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.base_url = "https://clovastudio.apigw.ntruss.com"
        self.logger = self._setup_logger()
        
        # HyperClova X 튜닝 설정[4]
        self.tuning_config = {
            "model_engine": "hyperclova-x",
            "max_tokens_per_batch": 4096,
            "temperature": 0.5,
            "top_p": 0.6,
            "repeat_penalty": 1.2
        }
    
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
    
    def _get_headers(self) -> Dict[str, str]:
        """API 요청 헤더 생성"""
        return {
            "Content-Type": "application/json; charset=utf-8",
            "X-NCP-CLOVASTUDIO-API-KEY": self.api_key,
            "X-NCP-APIGW-API-KEY": self.api_key_primary,
            "X-NCP-CLOVASTUDIO-REQUEST-ID": self.request_id
        }
    
    def prepare_tuning_data(self, json_file_path: str) -> List[Dict]:
        """JSON 파일을 HyperClova X 튜닝 형식으로 준비[2]"""
        try:
            self.logger.info(f"튜닝 데이터 준비 중: {json_file_path}")
            
            with open(json_file_path, 'r', encoding='utf-8') as f:
                time_llm_data = json.load(f)
            
            # HyperClova X SFT 형식으로 변환[4]
            tuning_data = []
            for sample in time_llm_data:
                # 특수 토큰을 사용한 대화 형식 구성
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
                    ]
                }
                tuning_data.append(formatted_sample)
            
            self.logger.info(f"튜닝 데이터 준비 완료: {len(tuning_data)}개 샘플")
            return tuning_data
            
        except Exception as e:
            self.logger.error(f"튜닝 데이터 준비 오류: {str(e)}")
            raise
    
    def upload_training_file(self, tuning_data: List[Dict], file_name: str = None) -> str:
        """튜닝 데이터 파일 업로드"""
        try:
            if file_name is None:
                file_name = f"time_llm_tuning_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
            
            # JSONL 형식으로 저장[2]
            jsonl_path = f"data/tuning/{file_name}"
            os.makedirs(os.path.dirname(jsonl_path), exist_ok=True)
            
            with open(jsonl_path, 'w', encoding='utf-8') as f:
                for sample in tuning_data:
                    f.write(json.dumps(sample, ensure_ascii=False) + '\n')
            
            self.logger.info(f"튜닝 파일 생성 완료: {jsonl_path}")
            
            # 파일 업로드 API 호출
            upload_url = f"{self.base_url}/tuning/v2/files"
            
            with open(jsonl_path, 'rb') as f:
                files = {'file': (file_name, f, 'application/json')}
                headers = {
                    "X-NCP-CLOVASTUDIO-API-KEY": self.api_key,
                    "X-NCP-APIGW-API-KEY": self.api_key_primary,
                    "X-NCP-CLOVASTUDIO-REQUEST-ID": self.request_id
                }
                
                response = requests.post(upload_url, files=files, headers=headers)
                
                if response.status_code == 200:
                    file_id = response.json()["result"]["fileId"]
                    self.logger.info(f"파일 업로드 성공: {file_id}")
                    return file_id
                else:
                    raise Exception(f"파일 업로드 실패: {response.status_code} - {response.text}")
                    
        except Exception as e:
            self.logger.error(f"파일 업로드 오류: {str(e)}")
            raise
    
    def create_tuning_job(self, file_id: str, task_name: str = None) -> str:
        """튜닝 작업 생성[3]"""
        try:
            if task_name is None:
                task_name = f"time-llm-etf-prediction-{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            self.logger.info(f"튜닝 작업 생성 중: {task_name}")
            
            # 튜닝 작업 생성 요청
            tuning_url = f"{self.base_url}/tuning/v2/tasks"
            
            request_data = {
                "taskName": task_name,
                "modelEngine": self.tuning_config["model_engine"],
                "trainingFileId": file_id,
                "hyperParameters": {
                    "learningRate": 0.0001,
                    "batchSize": 8,
                    "epochs": 3,
                    "warmupSteps": 100
                },
                "description": "Time-LLM for Korean ETF price prediction and retirement portfolio optimization"
            }
            
            response = requests.post(
                tuning_url,
                headers=self._get_headers(),
                json=request_data
            )
            
            if response.status_code == 200:
                task_id = response.json()["result"]["taskId"]
                self.logger.info(f"튜닝 작업 생성 성공: {task_id}")
                return task_id
            else:
                raise Exception(f"튜닝 작업 생성 실패: {response.status_code} - {response.text}")
                
        except Exception as e:
            self.logger.error(f"튜닝 작업 생성 오류: {str(e)}")
            raise
    
    def check_tuning_status(self, task_id: str) -> Dict:
        """튜닝 작업 상태 확인[3]"""
        try:
            status_url = f"{self.base_url}/tuning/v2/tasks/{task_id}"
            
            response = requests.get(status_url, headers=self._get_headers())
            
            if response.status_code == 200:
                result = response.json()["result"]
                status = result["status"]
                
                self.logger.info(f"튜닝 상태: {status}")
                
                if status == "COMPLETED":
                    model_id = result.get("tunedModelId")
                    self.logger.info(f"튜닝 완료! 모델 ID: {model_id}")
                elif status == "FAILED":
                    error_message = result.get("errorMessage", "Unknown error")
                    self.logger.error(f"튜닝 실패: {error_message}")
                
                return result
            else:
                raise Exception(f"상태 확인 실패: {response.status_code} - {response.text}")
                
        except Exception as e:
            self.logger.error(f"상태 확인 오류: {str(e)}")
            raise
    
    def wait_for_completion(self, task_id: str, check_interval: int = 60) -> str:
        """튜닝 완료까지 대기"""
        self.logger.info(f"튜닝 완료 대기 중... (체크 간격: {check_interval}초)")
        
        while True:
            try:
                status_result = self.check_tuning_status(task_id)
                status = status_result["status"]
                
                if status == "COMPLETED":
                    model_id = status_result["tunedModelId"]
                    self.logger.info(f"튜닝 완료! 최종 모델 ID: {model_id}")
                    return model_id
                elif status == "FAILED":
                    error_message = status_result.get("errorMessage", "Unknown error")
                    raise Exception(f"튜닝 실패: {error_message}")
                elif status in ["RUNNING", "PENDING", "QUEUED"]:
                    self.logger.info(f"튜닝 진행 중... 상태: {status}")
                    time.sleep(check_interval)
                else:
                    self.logger.warning(f"알 수 없는 상태: {status}")
                    time.sleep(check_interval)
                    
            except KeyboardInterrupt:
                self.logger.info("사용자에 의해 대기 중단됨")
                break
            except Exception as e:
                self.logger.error(f"상태 확인 중 오류: {str(e)}")
                time.sleep(check_interval)
