# -*- coding: utf-8 -*-

import requests
import json
from datetime import datetime
import os
from dotenv import load_dotenv

class HyperCLOVAXTuningTester:
    def __init__(self, host, api_key, request_id, task_id):
        """
        HyperCLOVA X 튜닝 모델 테스터
        
        Args:
            host (str): API 호스트 URL
            api_key (str): Bearer 토큰 형식의 API 키
            request_id (str): 요청 ID
            task_id (str): 튜닝 작업 ID
        """
        self._host = host
        self._api_key = api_key
        self._request_id = request_id
        self._task_id = task_id
        
        # API 키 형식 확인
        if not self._api_key.startswith('Bearer nv-'):
            raise ValueError("API 키는 'Bearer nv-'로 시작해야 합니다.")

    def execute(self, completion_request):
        """
        튜닝된 모델로 텍스트 생성 실행
        
        Args:
            completion_request (dict): 요청 데이터
            
        Returns:
            str: 모델 응답 또는 오류 메시지
        """
        headers = {
            'Authorization': self._api_key,
            'X-NCP-CLOVASTUDIO-REQUEST-ID': self._request_id,
            'Content-Type': 'application/json; charset=utf-8',
            'Accept': 'text/event-stream'
        }

        url = f"{self._host}/testapp/v2/tasks/{self._task_id}/chat-completions"
        
        try:
            print(f"요청 URL: {url}")
            print(f"작업 ID: {self._task_id}")
            print(f"API 키: {self._api_key[:20]}...")
            
            with requests.post(url, headers=headers, json=completion_request, stream=True) as response:
                if response.status_code == 401:
                    return "❌ 인증 실패: API 키를 확인하세요."
                elif response.status_code == 404:
                    return f"❌ 모델을 찾을 수 없습니다: {self._task_id}"
                elif response.status_code != 200:
                    return f"❌ API 오류 ({response.status_code}): {response.text}"
                
                # 스트리밍 응답 처리
                full_response = ""
                for line in response.iter_lines():
                    if line:
                        line_text = line.decode("utf-8")
                        print(line_text)  # 실시간 출력
                        
                        # JSON 파싱하여 실제 응답 추출
                        if line_text.startswith('data: '):
                            try:
                                data = json.loads(line_text[6:])  # 'data: ' 제거
                                if 'message' in data and 'content' in data['message']:
                                    full_response += data['message']['content']
                            except json.JSONDecodeError:
                                continue
                
                return full_response if full_response else "응답을 파싱할 수 없습니다."
                
        except requests.exceptions.RequestException as e:
            return f"❌ 네트워크 오류: {e}"

    def test_etf_prediction(self, etf_data):
        """
        ETF 시계열 예측 테스트
        
        Args:
            etf_data (str): 정규화된 ETF 가격 데이터
            
        Returns:
            str: 예측 결과
        """
        preset_text = [
            {
                "role": "user", 
                "content": f"""다음은 310970 ETF의 정규화된 가격 시계열 데이터입니다.
과거 30일간의 데이터를 바탕으로 향후 10일의 가격을 예측하세요.

입력 데이터: {etf_data}

예상 출력:"""
            }
        ]

        request_data = {
            'messages': preset_text,
            'topP': 0.8,
            'topK': 0,
            'maxTokens': 200,
            'temperature': 0.3,
            'repeatPenalty': 1.0,
            'stopBefore': [],
            'includeAiFilters': True
        }

        print("=== ETF 시계열 예측 테스트 ===")
        print(f"입력 데이터: {etf_data}")
        print("\n응답:")
        
        return self.execute(request_data)

def create_test_cases():
    """ETF 테스트 케이스 생성"""
    return [
        # 테스트 케이스 1
        "0.647059 0.500000 0.536765 0.463235 0.647059 0.794118 0.808824 0.838235 0.713235 1.000000 0.823529 0.757353 0.485294 0.544118 0.397059 0.338235 0.536765 0.654412 0.360294 0.397059 0.308824 0.448529 0.235294 0.191176 0.301471 0.058824 0.279412 0.426471 0.139706 0.000000",
        
        # 테스트 케이스 2 (상승 트렌드)
        "0.100000 0.150000 0.200000 0.250000 0.300000 0.350000 0.400000 0.450000 0.500000 0.550000 0.600000 0.650000 0.700000 0.750000 0.800000 0.850000 0.900000 0.920000 0.940000 0.960000 0.970000 0.980000 0.985000 0.990000 0.992000 0.994000 0.996000 0.998000 0.999000 1.000000",
        
        # 테스트 케이스 3 (하락 트렌드)
        "1.000000 0.950000 0.900000 0.850000 0.800000 0.750000 0.700000 0.650000 0.600000 0.550000 0.500000 0.450000 0.400000 0.350000 0.300000 0.250000 0.200000 0.150000 0.100000 0.080000 0.060000 0.040000 0.030000 0.020000 0.015000 0.010000 0.008000 0.005000 0.002000 0.000000"
    ]

def main():
    """메인 실행 함수"""
    print("=== HyperCLOVA X 튜닝 모델 테스트 ===\n")
    
    # 환경변수 로드
    load_dotenv()
    
    # API 설정
    host = 'https://clovastudio.stream.ntruss.com'
    api_key = os.getenv('CLOVA_API_KEY')  # nv-로 시작하는 새 API 키
    request_id = '8fe68c8dd0434060a6ed19e0a34f6829'  # 고유 요청 ID
    task_id = 'vpcxuy1b'  # 튜닝 작업 ID
    
    # API 키 형식 확인
    if not api_key:
        api_key = input("CLOVA API 키를 입력하세요 (nv-로 시작): ").strip()
    
    if not api_key.startswith('nv-'):
        print("❌ API 키는 'nv-'로 시작해야 합니다.")
        return
    
    # Bearer 토큰 형식으로 변환
    bearer_token = f"Bearer {api_key}"
    
    try:
        # 테스터 초기화
        tester = HyperCLOVAXTuningTester(host, bearer_token, request_id, task_id)
        
        # 테스트 케이스 실행
        test_cases = create_test_cases()
        results = []
        
        for i, etf_data in enumerate(test_cases, 1):
            print(f"\n{'='*50}")
            print(f"테스트 {i}/{len(test_cases)}")
            print(f"{'='*50}")
            
            result = tester.test_etf_prediction(etf_data)
            
            results.append({
                'test_number': i,
                'input_data': etf_data,
                'prediction': result,
                'timestamp': datetime.now().isoformat()
            })
            
            print(f"\n최종 예측 결과: {result}")
        
        # 결과 저장
        output_file = f'tuning_test_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        print(f"\n=== 테스트 완료 ===")
        print(f"결과 저장: {output_file}")
        print(f"총 테스트: {len(results)}개")
        
    except Exception as e:
        print(f"❌ 오류 발생: {e}")

def quick_test():
    """빠른 단일 테스트"""
    load_dotenv()
    
    host = 'https://clovastudio.stream.ntruss.com'
    api_key = os.getenv('CLOVA_API_KEY')
    
    if not api_key:
        api_key = input("CLOVA API 키를 입력하세요 (nv-로 시작): ").strip()
    
    bearer_token = f"Bearer {api_key}"
    request_id = '8fe68c8dd0434060a6ed19e0a34f6829'
    task_id = 'vpcxuy1b'
    
    tester = HyperCLOVAXTuningTester(host, bearer_token, request_id, task_id)
    
    # 간단한 테스트
    test_data = "0.647059 0.500000 0.536765 0.463235 0.647059 0.794118 0.808824 0.838235 0.713235 1.000000 0.823529 0.757353 0.485294 0.544118 0.397059 0.338235 0.536765 0.654412 0.360294 0.397059 0.308824 0.448529 0.235294 0.191176 0.301471 0.058824 0.279412 0.426471 0.139706 0.000000"
    
    result = tester.test_etf_prediction(test_data)
    print(f"\n최종 결과: {result}")

if __name__ == '__main__':
    # 전체 테스트 실행
    main()
    
    # 또는 빠른 테스트만 실행
    # quick_test()
