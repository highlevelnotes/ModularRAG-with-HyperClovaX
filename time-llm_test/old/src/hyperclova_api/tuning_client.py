# src/hyperclova_api/tuning_client.py
import os
import json
import requests
import time
import logging
from typing import Dict, List, Optional
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

class HyperClovaXTuningClient:
    """순수 HyperClova X API 기반 튜닝 클라이언트"""
    
    def __init__(self):
        self.api_key = os.getenv("CLOVASTUDIO_API_KEY")
        
        if not self.api_key:
            raise ValueError("CLOVASTUDIO_API_KEY가 설정되지 않았습니다. .env 파일을 확인하세요.")
        
        # 검색 결과에서 확인된 새로운 API 경로[2]
        self.base_url = "https://clovastudio.stream.ntruss.com"
        self.logger = self._setup_logger()
        
        # 튜닝 설정
        self.tuning_config = {
            "model": "HCX-003",
            "method": "LoRA",  # 검색 결과에서 확인된 튜닝 방법[3]
            "taskType": "GENERATION",
            "trainEpochs": 3,
            "learningRate": 0.0001
        }
        
        self.logger.info("HyperClova X 순수 API 튜닝 클라이언트 초기화 완료")
    
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
        """API 요청 헤더 생성 (Bearer 토큰 방식)[2]"""
        if request_id is None:
            request_id = f"tuning-{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": content_type,
            "X-NCP-CLOVASTUDIO-REQUEST-ID": request_id
        }
    
    def _get_file_upload_headers(self, request_id: str = None) -> Dict[str, str]:
        """파일 업로드용 헤더"""
        if request_id is None:
            request_id = f"upload-{int(time.time())}"
        
        return {
            "Authorization": f"Bearer {self.api_key}",
            "X-NCP-CLOVASTUDIO-REQUEST-ID": request_id
        }
    
    def prepare_tuning_data(self, json_file_path: str) -> str:
        """JSON 파일을 HyperClova X 튜닝 형식으로 변환"""
        try:
            self.logger.info(f"튜닝 데이터 준비 중: {json_file_path}")
            
            with open(json_file_path, 'r', encoding='utf-8') as f:
                time_llm_data = json.load(f)
            
            # HyperClova X SFT 형식으로 변환[3]
            tuning_samples = []
            for sample in time_llm_data:
                # 특수 토큰을 사용한 대화 형식 구성[3]
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
                tuning_samples.append(formatted_sample)
            
            # JSONL 파일로 저장
            output_path = f"data/hyperclova_tuning/time_llm_tuning_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                for sample in tuning_samples:
                    f.write(json.dumps(sample, ensure_ascii=False) + '\n')
            
            self.logger.info(f"튜닝 데이터 준비 완료: {len(tuning_samples)}개 샘플 -> {output_path}")
            return output_path
            
        except Exception as e:
            self.logger.error(f"튜닝 데이터 준비 오류: {str(e)}")
            raise
    
    def upload_training_file(self, jsonl_file_path: str) -> str:
        """튜닝 데이터 파일 업로드"""
        try:
            self.logger.info(f"튜닝 파일 업로드 시작: {jsonl_file_path}")
            
            # 여러 가능한 업로드 경로 시도
            possible_paths = [
                "/tuning/v2/files",
                "/tuning/files", 
                "/api/tuning/files",
                "/testapp/v1/tuning/files"
            ]
            
            for path in possible_paths:
                upload_url = f"{self.base_url}{path}"
                
                try:
                    self.logger.info(f"업로드 시도: {upload_url}")
                    
                    with open(jsonl_file_path, 'rb') as f:
                        files = {
                            'file': (os.path.basename(jsonl_file_path), f, 'application/json')
                        }
                        
                        headers = self._get_file_upload_headers()
                        
                        response = requests.post(
                            upload_url, 
                            files=files, 
                            headers=headers, 
                            timeout=120
                        )
                        
                        self.logger.info(f"응답 상태: {response.status_code}")
                        self.logger.info(f"응답 내용: {response.text}")
                        
                        if response.status_code == 200:
                            result = response.json()
                            if "result" in result and "fileId" in result["result"]:
                                file_id = result["result"]["fileId"]
                                self.logger.info(f"파일 업로드 성공: {file_id}")
                                return file_id
                        elif response.status_code != 404:
                            # 404가 아닌 다른 오류는 로깅하고 계속 시도
                            self.logger.warning(f"업로드 실패 ({upload_url}): {response.status_code} - {response.text}")
                
                except Exception as e:
                    self.logger.debug(f"업로드 시도 실패 ({upload_url}): {str(e)}")
                    continue
            
            raise Exception("모든 업로드 경로에서 파일 업로드 실패")
                    
        except Exception as e:
            self.logger.error(f"파일 업로드 오류: {str(e)}")
            raise
    
    def create_tuning_job(self, file_id: str, task_name: str = None) -> str:
        """튜닝 작업 생성"""
        try:
            if task_name is None:
                task_name = f"time-llm-etf-{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            self.logger.info(f"튜닝 작업 생성 중: {task_name}")
            
            # 여러 가능한 튜닝 작업 생성 경로 시도
            possible_paths = [
                "/tuning/v2/tasks",
                "/tuning/tasks",
                "/api/tuning/tasks",
                "/testapp/v1/tuning/tasks"
            ]
            
            request_data = {
                "taskName": task_name,
                "model": self.tuning_config["model"],
                "method": self.tuning_config["method"],
                "taskType": self.tuning_config["taskType"],
                "trainingFileId": file_id,
                "trainEpochs": self.tuning_config["trainEpochs"],
                "learningRate": self.tuning_config["learningRate"],
                "description": "Time-LLM for Korean ETF price prediction"
            }
            
            for path in possible_paths:
                tuning_url = f"{self.base_url}{path}"
                
                try:
                    self.logger.info(f"튜닝 작업 생성 시도: {tuning_url}")
                    
                    response = requests.post(
                        tuning_url,
                        headers=self._get_headers(),
                        json=request_data,
                        timeout=60
                    )
                    
                    self.logger.info(f"응답 상태: {response.status_code}")
                    self.logger.info(f"응답 내용: {response.text}")
                    
                    if response.status_code == 200:
                        result = response.json()
                        if "result" in result and "id" in result["result"]:
                            task_id = result["result"]["id"]
                            self.logger.info(f"튜닝 작업 생성 성공: {task_id}")
                            return task_id
                        elif "result" in result and "taskId" in result["result"]:
                            task_id = result["result"]["taskId"]
                            self.logger.info(f"튜닝 작업 생성 성공: {task_id}")
                            return task_id
                    elif response.status_code != 404:
                        self.logger.warning(f"튜닝 작업 생성 실패 ({tuning_url}): {response.status_code} - {response.text}")
                
                except Exception as e:
                    self.logger.debug(f"튜닝 작업 생성 시도 실패 ({tuning_url}): {str(e)}")
                    continue
            
            raise Exception("모든 경로에서 튜닝 작업 생성 실패")
                
        except Exception as e:
            self.logger.error(f"튜닝 작업 생성 오류: {str(e)}")
            raise
    
    def check_tuning_status(self, task_id: str) -> Dict:
        """튜닝 작업 상태 확인"""
        try:
            # 검색 결과에서 확인된 상태 확인 API[2]
            status_url = f"{self.base_url}/tuning/v2/tasks/{task_id}"
            
            response = requests.get(status_url, headers=self._get_headers(), timeout=30)
            
            if response.status_code == 200:
                result = response.json()["result"]
                status = result["status"]
                
                # 검색 결과에서 확인된 상태값들[2]
                status_mapping = {
                    "WAIT": "대기 중",
                    "RUNNING": "실행 중", 
                    "SUCCEEDED": "완료",
                    "FAILED": "실패"
                }
                
                self.logger.info(f"튜닝 상태: {status} ({status_mapping.get(status, status)})")
                
                # 진행 상황 표시
                if "statusInfo" in result:
                    status_info = result["statusInfo"]
                    curr_step = status_info.get("currStep")
                    total_steps = status_info.get("totalTrainSteps")
                    curr_epoch = status_info.get("currEpoch")
                    total_epochs = status_info.get("totalTrainEpochs")
                    
                    if curr_step and total_steps:
                        progress = (curr_step / total_steps) * 100
                        self.logger.info(f"진행률: {progress:.1f}% ({curr_step}/{total_steps} 스텝)")
                    
                    if curr_epoch and total_epochs:
                        self.logger.info(f"에폭: {curr_epoch}/{total_epochs}")
                
                if status == "SUCCEEDED":
                    model_id = result.get("id")
                    self.logger.info(f"튜닝 완료! 모델 ID: {model_id}")
                elif status == "FAILED":
                    status_info = result.get("statusInfo", {})
                    failure_reason = status_info.get("failureReason", "Unknown reason")
                    message = status_info.get("message", "Unknown error")
                    self.logger.error(f"튜닝 실패: {failure_reason} - {message}")
                
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
                
                if status == "SUCCEEDED":
                    model_id = status_result["id"]
                    self.logger.info(f"튜닝 완료! 최종 모델 ID: {model_id}")
                    return model_id
                elif status == "FAILED":
                    status_info = status_result.get("statusInfo", {})
                    failure_reason = status_info.get("failureReason", "Unknown reason")
                    message = status_info.get("message", "Unknown error")
                    raise Exception(f"튜닝 실패: {failure_reason} - {message}")
                elif status in ["RUNNING", "WAIT"]:
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
