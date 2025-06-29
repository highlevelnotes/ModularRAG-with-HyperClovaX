import requests
import os
from dotenv import load_dotenv

def test_clova_api_new_format():
    """새로운 API 키 형식으로 CLOVA Studio API 테스트"""
    load_dotenv()
    
    # 환경변수에서 API 키 가져오기
    api_key = os.getenv('CLOVA_API_KEY')
    
    print("=== CLOVA Studio API 키 테스트 (신규 형식) ===")
    print(f"CLOVA_API_KEY 존재: {'✓' if api_key else '❌'}")
    
    if not api_key:
        print("\n❌ CLOVA_API_KEY가 설정되지 않았습니다.")
        return False
    
    print(f"CLOVA_API_KEY: {api_key[:10]}...")
    
    # 새로운 형식의 헤더 (Legacy auth 방식 제거)
    headers = {
        'Authorization': f'Bearer {api_key}',
        'Content-Type': 'application/json'
    }
    
    # 기본 Chat Completions API 테스트
    url = 'https://clovastudio.stream.ntruss.com/testapp/v1/chat-completions/HCX-003'
    
    payload = {
        'messages': [
            {'role': 'user', 'content': '안녕하세요'}
        ],
        'maxTokens': 10,
        'temperature': 0.1
    }
    
    try:
        print(f"\nAPI 호출 중 (신규 형식)...")
        print(f"URL: {url}")
        print(f"헤더: Authorization: Bearer {api_key[:10]}...")
        
        response = requests.post(url, headers=headers, json=payload)
        
        print(f"응답 상태 코드: {response.status_code}")
        print(f"응답 내용: {response.text}")
        
        if response.status_code == 200:
            print("\n✅ API 키가 정상적으로 작동합니다!")
            result = response.json()
            if 'result' in result and 'message' in result['result']:
                print(f"AI 응답: {result['result']['message']['content']}")
            return True
            
        elif response.status_code == 401:
            print("\n❌ 여전히 인증 실패")
            print("추가 해결 방법:")
            print("1. CLOVA Studio에서 기존 API 키 삭제")
            print("2. 완전히 새로운 API 키 발급")
            print("3. 테스트 API 키가 아닌 서비스 API 키 시도")
            return False
            
        else:
            print(f"\n❌ 다른 오류 ({response.status_code})")
            return False
            
    except Exception as e:
        print(f"\n❌ 네트워크 오류: {e}")
        return False

def test_alternative_endpoint():
    """대안 엔드포인트로 테스트"""
    load_dotenv()
    api_key = os.getenv('CLOVA_API_KEY')
    
    if not api_key:
        return False
    
    # 대안 엔드포인트들
    endpoints = [
        'https://clovastudio.stream.ntruss.com/testapp/v1/chat-completions/HCX-DASH-001',
        'https://clovastudio.stream.ntruss.com/testapp/v1/completions/HCX-003',
        'https://clovastudio.apigw.ntruss.com/testapp/v1/chat-completions/HCX-003'
    ]
    
    headers = {
        'Authorization': f'Bearer {api_key}',
        'Content-Type': 'application/json'
    }
    
    payload = {
        'messages': [{'role': 'user', 'content': '테스트'}],
        'maxTokens': 5
    }
    
    for i, url in enumerate(endpoints, 1):
        print(f"\n대안 {i} 테스트: {url}")
        try:
            response = requests.post(url, headers=headers, json=payload)
            print(f"상태 코드: {response.status_code}")
            if response.status_code == 200:
                print("✅ 성공!")
                return True
        except Exception as e:
            print(f"오류: {e}")
    
    return False

if __name__ == '__main__':
    # 신규 형식으로 테스트
    success = test_clova_api_new_format()
    
    # 실패시 대안 엔드포인트 테스트
    if not success:
        print("\n대안 엔드포인트 테스트 중...")
        test_alternative_endpoint()
