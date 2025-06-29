# src/langchain_integration/hyperclova_tuning_client.py
import os
import json
import requests
import time
import logging
from typing import Dict, List, Optional
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

class LangChainHyperClovaTuningClient:
    """LangChain 기반 HyperClova X 튜닝 클라이언트 (새로운 Bearer 토큰 방식)"""
    
    def __init__(self):
        self.api_key = os.getenv("NCP_CLOVASTUDIO_API_KEY")
        
        if not self.api_key:
            raise ValueError("NCP_CLOVASTUDIO_API_KEY가 설정되지 않았습니다. .env 파일을 확인하세요.")
        
        # 새로운 API 경로 (공식 문서 기준)
        self.base_url = "https://clovastudio.stream.ntruss.com"
        self.logger = self._setup_logger()
        
        # 튜닝 설정
        self.tuning_config = {
            "model_engine": "HCX-003",
            "hyperparameters": {
                "learning_rate": 0.0001,
                "batch_size": 8,
                "epochs": 3,
                "warmup_steps": 100
            }
        }
        
        self.logger.info("LangChain HyperClova X 튜닝 클라이언트 초기화 완료 (Bearer 토큰 방식)")
    
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
    
    def _get_headers(self, request_id: str = None, content_type: str = "application/json") -> Dict[str, str]:
        """API 요청 헤더 생성 (새로운 Bearer 토큰 방식)"""
        if request_id is None:
            request_id = f"langchain-tuning-{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": content_type,
            "X-NCP-CLOVASTUDIO-REQUEST-ID": request_id
        }
    
    def _get_file_upload_headers(self, request_id: str = None) -> Dict[str, str]:
        """파일 업로드용 헤더 (Content-Type 제외)"""
        if request_id is None:
            request_id = f"upload-{int(time.time())}"
        
        return {
            "Authorization": f"Bearer {self.api_key}",
            "X-NCP-CLOVASTUDIO-REQUEST-ID": request_id
        }
    
    def upload_training_file(self, jsonl_file_path: str) -> str:
        """튜닝 데이터 파일 업로드 (새로운 API 방식)"""
        try:
            self.logger.info(f"튜닝 파일 업로드 시작: {jsonl_file_path}")
            
            upload_url = f"{self.base_url}/tuning/v2/files"
            
            with open(jsonl_file_path, 'rb') as f:
                files = {
                    'file': (os.path.basename(jsonl_file_path), f, 'application/json')
                }
                
                # 파일 업로드용 헤더 (Content-Type은 requests가 자동 설정)
                headers = self._get_file_upload_headers()
                
                response = requests.post(upload_url, files=files, headers=headers, timeout=60)
                
                if response.status_code == 200:
                    result = response.json()
                    file_id = result["result"]["fileId"]
                    self.logger.info(f"파일 업로드 성공: {file_id}")
                    return file_id
                else:
                    raise Exception(f"파일 업로드 실패: {response.status_code} - {response.text}")
                    
        except Exception as e:
            self.logger.error(f"파일 업로드 오류: {str(e)}")
            raise
    
    def create_tuning_job(self, file_id: str, task_name: str = None) -> str:
        """튜닝 작업 생성"""
        try:
            if task_name is None:
                task_name = f"langchain-time-llm-{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            self.logger.info(f"튜닝 작업 생성 중: {task_name}")
            
            tuning_url = f"{self.base_url}/tuning/v2/tasks"
            
            request_data = {
                "taskName": task_name,
                "modelEngine": self.tuning_config["model_engine"],
                "trainingFileId": file_id,
                "hyperParameters": self.tuning_config["hyperparameters"],
                "description": "LangChain-based Time-LLM for Korean ETF price prediction"
            }
            
            response = requests.post(
                tuning_url,
                headers=self._get_headers(),
                json=request_data,
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                task_id = result["result"]["taskId"]
                self.logger.info(f"튜닝 작업 생성 성공: {task_id}")
                return task_id
            else:
                raise Exception(f"튜닝 작업 생성 실패: {response.status_code} - {response.text}")
                
        except Exception as e:
            self.logger.error(f"튜닝 작업 생성 오류: {str(e)}")
            raise
    
    def check_tuning_status(self, task_id: str) -> Dict:
        """튜닝 작업 상태 확인"""
        try:
            status_url = f"{self.base_url}/tuning/v2/tasks/{task_id}"
            
            response = requests.get(status_url, headers=self._get_headers(), timeout=30)
            
            if response.status_code == 200:
                result = response.json()["result"]
                status = result["status"]
                
                self.logger.info(f"튜닝 상태: {status}")
                
                if status == "SUCCEEDED":  # 새로운 API에서는 SUCCEEDED 사용
                    model_id = result.get("id")  # 새로운 API에서는 id 필드 사용
                    self.logger.info(f"튜닝 완료! 모델 ID: {model_id}")
                elif status == "FAILED":
                    status_info = result.get("statusInfo", {})
                    error_message = status_info.get("message", "Unknown error")
                    failure_reason = status_info.get("failureReason", "Unknown reason")
                    self.logger.error(f"튜닝 실패: {failure_reason} - {error_message}")
                
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
                
                if status == "SUCCEEDED":  # 새로운 API 상태값
                    model_id = status_result["id"]
                    self.logger.info(f"튜닝 완료! 최종 모델 ID: {model_id}")
                    return model_id
                elif status == "FAILED":
                    status_info = status_result.get("statusInfo", {})
                    error_message = status_info.get("message", "Unknown error")
                    failure_reason = status_info.get("failureReason", "Unknown reason")
                    raise Exception(f"튜닝 실패: {failure_reason} - {error_message}")
                elif status in ["RUNNING", "WAIT"]:  # 새로운 API 상태값
                    # 진행 상황 표시
                    status_info = status_result.get("statusInfo", {})
                    curr_step = status_info.get("currStep")
                    total_steps = status_info.get("totalTrainSteps")
                    curr_epoch = status_info.get("currEpoch")
                    total_epochs = status_info.get("totalTrainEpochs")
                    
                    progress_info = f"상태: {status}"
                    if curr_step and total_steps:
                        progress_info += f", 스텝: {curr_step}/{total_steps}"
                    if curr_epoch and total_epochs:
                        progress_info += f", 에폭: {curr_epoch}/{total_epochs}"
                    
                    self.logger.info(f"튜닝 진행 중... {progress_info}")
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
