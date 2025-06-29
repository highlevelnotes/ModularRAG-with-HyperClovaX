import os
import json
import time
import requests
from datetime import datetime
from dotenv import load_dotenv

class HyperCLOVAXTuningClient:
    def __init__(self):
        load_dotenv()
        self.api_key = os.getenv('CLOVA_API_KEY')
        self.access_key = os.getenv('NCP_ACCESS_KEY')
        self.secret_key = os.getenv('NCP_SECRET_KEY')
        
        if not all([self.api_key, self.access_key, self.secret_key]):
            raise ValueError("모든 API 키가 필요합니다: CLOVA_API_KEY, NCP_ACCESS_KEY, NCP_SECRET_KEY")
        
        self.base_url = "https://clovastudio.stream.ntruss.com"
        self.headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json'
        }

    def test_api_connection(self):
        """API 연결 테스트"""
        url = f"{self.base_url}/testapp/v1/chat-completions/HCX-003"
        
        payload = {
            'messages': [
                {'role': 'user', 'content': '연결 테스트'}
            ],
            'maxTokens': 5,
            'temperature': 0.1
        }
        
        try:
            response = requests.post(url, headers=self.headers, json=payload)
            
            if response.status_code == 200:
                print("✅ API 연결 성공")
                return True
            else:
                print(f"❌ API 연결 실패: {response.status_code} - {response.text}")
                return False
                
        except Exception as e:
            print(f"❌ API 연결 오류: {e}")
            return False

    def start_tuning_job(self, bucket_name, file_path, job_name=None):
        """Object Storage 방식으로 튜닝 작업 시작"""
        if job_name is None:
            job_name = f"etf_tuning_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        url = f"{self.base_url}/tuning/v2/tasks"
        
        payload = {
            "name": job_name,
            "model": "HCX-003",
            "tuningType": "PEFT",
            "taskType": "GENERATION",
            "trainEpochs": 3,
            "learningRate": 1e-5,
            "trainingDatasetBucket": bucket_name,
            "trainingDatasetFilePath": file_path,
            "trainingDatasetAccessKey": self.access_key,
            "trainingDatasetSecretKey": self.secret_key
        }
        
        try:
            print(f"튜닝 작업 요청 중...")
            print(f"URL: {url}")
            print(f"작업명: {job_name}")
            print(f"버킷: {bucket_name}")
            print(f"파일 경로: {file_path}")
            
            response = requests.post(url, headers=self.headers, json=payload)
            
            print(f"응답 상태 코드: {response.status_code}")
            
            if response.status_code in [200, 201]:
                result = response.json()
                
                # 수정된 작업 ID 파싱 로직
                task_id = None
                if 'result' in result:
                    task_id = result['result'].get('id')  # 'id' 필드가 작업 ID
                
                if task_id:
                    print(f"✅ 튜닝 작업 시작 성공!")
                    print(f"작업 ID: {task_id}")
                    print(f"상태: {result['result'].get('status', 'UNKNOWN')}")
                    print(f"방법: {result['result'].get('method', 'UNKNOWN')}")
                    return task_id
                else:
                    print(f"❌ 작업 ID를 찾을 수 없음")
                    return None
            else:
                print(f"❌ 튜닝 작업 시작 실패: {response.text}")
                return None
                
        except Exception as e:
            print(f"❌ 튜닝 요청 오류: {e}")
            return None


    def check_tuning_status(self, task_id):
        """튜닝 상태 확인"""
        url = f"{self.base_url}/tuning/v2/tasks/{task_id}"
        
        try:
            response = requests.get(url, headers=self.headers)
            
            if response.status_code == 200:
                return response.json()
            else:
                print(f"상태 확인 실패: {response.status_code} - {response.text}")
                return {}
                
        except Exception as e:
            print(f"❌ 상태 확인 실패: {e}")
            return {}

    def wait_for_completion(self, task_id, interval=300, max_wait_time=7200):
        """튜닝 완료 대기 (최대 2시간)"""
        print(f"튜닝 완료 대기 중... (5분마다 상태 확인, 최대 2시간)")
        
        start_time = time.time()
        
        while time.time() - start_time < max_wait_time:
            status_info = self.check_tuning_status(task_id)
            
            if not status_info:
                print("상태 정보를 가져올 수 없습니다. 5분 후 재시도...")
                time.sleep(interval)
                continue
            
            status = status_info.get('status', 'UNKNOWN')
            progress = status_info.get('progress', 0)
            
            print(f"튜닝 상태: {status} 진행률: {progress}%")
            
            if status in ['COMPLETED', 'SUCCESS', 'FINISHED']:
                model_id = (status_info.get('tunedModelId') or 
                           status_info.get('modelId') or
                           status_info.get('model_id'))
                print(f"✅ 튜닝 완료! 모델 ID: {model_id}")
                return model_id
                
            elif status in ['FAILED', 'ERROR', 'CANCELLED']:
                error_msg = status_info.get('errorMessage', '알 수 없는 오류')
                print(f"❌ 튜닝 실패: {error_msg}")
                return None
            
            time.sleep(interval)
        
        print("❌ 최대 대기 시간 초과")
        return None

    def test_tuned_model(self, model_id, prompt):
        """튜닝된 모델 테스트"""
        url = f"{self.base_url}/testapp/v1/chat-completions/{model_id}"
        
        payload = {
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "topP": 0.8,
            "topK": 0,
            "maxTokens": 500,
            "temperature": 0.3,
            "repeatPenalty": 1.0,
            "stopBefore": [],
            "includeAiFilters": True
        }
        
        try:
            response = requests.post(url, headers=self.headers, json=payload)
            
            if response.status_code == 200:
                result = response.json()
                if 'result' in result and 'message' in result['result']:
                    return result['result']['message']['content']
                else:
                    return f"응답 구조 오류: {result}"
            else:
                return f"API 오류 ({response.status_code}): {response.text}"
                
        except Exception as e:
            return f"네트워크 오류: {e}"


def main():
    """메인 실행 함수"""
    print("=== HyperCLOVA X 튜닝 (Object Storage 방식) ===\n")
    
    try:
        # 클라이언트 초기화
        client = HyperCLOVAXTuningClient()
        
        # 1단계: API 연결 테스트
        print("1단계: API 연결 테스트 중...")
        if not client.test_api_connection():
            print("❌ API 연결 실패. 프로그램을 종료합니다.")
            return
        
        # 2단계: Object Storage 정보 입력
        print("\n2단계: Object Storage 정보 입력...")
        
        # 사용자 입력 또는 직접 설정
        bucket_name = input("버킷명을 입력하세요: ").strip()
        if not bucket_name:
            bucket_name = "hyperclova-tuning-data"  # 기본값
            print(f"기본 버킷명 사용: {bucket_name}")
        
        file_path = "hyperclova_instruction_dataset.jsonl"
        print(f"파일 경로: {file_path}")
        
        # 3단계: 튜닝 작업 시작
        print("\n3단계: 튜닝 작업 시작 중...")
        task_id = client.start_tuning_job(bucket_name, file_path)
        
        if not task_id:
            print("\n❌ 튜닝 작업 시작 실패.")
            print("확인 사항:")
            print("1. 버킷명이 정확한지 확인")
            print("2. 파일이 버킷에 업로드되었는지 확인")
            print("3. Access Key와 Secret Key가 올바른지 확인")
            print("4. Object Storage 권한이 있는지 확인")
            return
        
        # 4단계: 튜닝 완료 대기
        print(f"\n4단계: 튜닝 완료 대기 중... (작업 ID: {task_id})")
        model_id = client.wait_for_completion(task_id)
        
        if not model_id:
            print("❌ 튜닝이 완료되지 않았습니다.")
            return
        
        # 5단계: 튜닝된 모델 테스트
        print(f"\n5단계: 튜닝된 모델 테스트 중... (모델 ID: {model_id})")
        
        test_prompt = """다음은 310970 ETF의 정규화된 가격 시계열 데이터입니다.
과거 30일간의 데이터를 바탕으로 향후 10일의 가격을 예측하세요.

입력 데이터: 0.647059 0.500000 0.536765 0.463235 0.647059 0.794118 0.808824 0.838235 0.713235 1.000000 0.823529 0.757353 0.485294 0.544118 0.397059 0.338235 0.536765 0.654412 0.360294 0.397059 0.308824 0.448529 0.235294 0.191176 0.301471 0.058824 0.279412 0.426471 0.139706 0.000000

예상 출력:"""
        
        response = client.test_tuned_model(model_id, test_prompt)
        print(f"튜닝된 모델 응답:\n{response}")
        
        # 결과 저장
        result_info = {
            "task_id": task_id,
            "model_id": model_id,
            "tuning_date": datetime.now().isoformat(),
            "test_result": response,
            "bucket_name": bucket_name,
            "file_path": file_path,
            "api_method": "Object Storage"
        }
        
        with open('hcx003_tuning_result.json', 'w', encoding='utf-8') as f:
            json.dump(result_info, f, ensure_ascii=False, indent=2)
        
        print(f"\n=== 튜닝 완료 ===")
        print(f"튜닝된 모델 ID: {model_id}")
        print(f"결과 저장: hcx003_tuning_result.json")
        
    except Exception as e:
        print(f"❌ 예상치 못한 오류 발생: {e}")
        import traceback
        traceback.print_exc()

def check_existing_task():
    """기존 작업 상태 확인"""
    load_dotenv()
    api_key = os.getenv('CLOVA_API_KEY')
    
    headers = {
        'Authorization': f'Bearer {api_key}',
        'Content-Type': 'application/json'
    }
    
    task_id = "vpcxuy1b"  # 응답에서 받은 작업 ID
    url = f"https://clovastudio.stream.ntruss.com/tuning/v2/tasks/{task_id}"
    
    try:
        response = requests.get(url, headers=headers)
        
        if response.status_code == 200:
            result = response.json()
            status = result.get('result', {}).get('status', 'UNKNOWN')
            print(f"작업 ID: {task_id}")
            print(f"현재 상태: {status}")
            print(f"전체 응답: {result}")
            return result
        else:
            print(f"상태 확인 실패: {response.status_code} - {response.text}")
            return None
            
    except Exception as e:
        print(f"오류: {e}")
        return None


if __name__ == '__main__':
    # main()
    check_existing_task()

