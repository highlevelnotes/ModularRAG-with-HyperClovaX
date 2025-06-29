import json
import time

def comprehensive_data_check(file_path):
    """튜닝 데이터 종합 점검"""
    print("=== 튜닝 데이터 종합 점검 ===\n")
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = [json.loads(line) for line in f if line.strip()]
        
        print(f"총 데이터 수: {len(data)}개")
        
        # 최소 요구량 확인
        if len(data) < 10:
            print("❌ 데이터 부족: 최소 10개 이상 필요")
        elif len(data) < 400:
            print("⚠️ 권장량 미달: 400개 이상 권장")
        else:
            print("✅ 데이터 양 충족")
        
        # 형식 검증
        valid_count = 0
        for i, item in enumerate(data):
            if 'messages' in item and isinstance(item['messages'], list):
                messages = item['messages']
                if len(messages) >= 2:
                    user_msg = any(msg.get('role') == 'user' for msg in messages)
                    assistant_msg = any(msg.get('role') == 'assistant' for msg in messages)
                    if user_msg and assistant_msg:
                        valid_count += 1
        
        print(f"유효한 대화 쌍: {valid_count}/{len(data)}개")
        
        if valid_count < len(data) * 0.9:
            print("❌ 데이터 형식 문제: 90% 이상이 유효해야 함")
        else:
            print("✅ 데이터 형식 양호")
            
        return valid_count >= 10
        
    except Exception as e:
        print(f"❌ 데이터 점검 실패: {e}")
        return False

# 실행
data_ok = comprehensive_data_check('data/hyperclova_tuning/fixed_tuning_data_20250628_122248.jsonl')

def check_account_permissions():
    """계정 권한 및 서비스 이용 가능 여부 확인"""
    from dotenv import load_dotenv
    import os
    import requests
    
    load_dotenv()
    api_key = os.getenv('CLOVA_API_KEY')
    
    # 다양한 모델로 접근 권한 테스트
    models_to_test = ['HCX-003']
    
    headers = {
        'Authorization': f'Bearer {api_key}',
        'Content-Type': 'application/json'
    }
    
    print("=== 계정 권한 점검 ===\n")
    
    for model in models_to_test:
        url = f'https://clovastudio.stream.ntruss.com/testapp/v1/chat-completions/{model}'
        payload = {
            'messages': [{'role': 'user', 'content': '테스트'}],
            'maxTokens': 5
        }
        
        try:
            response = requests.post(url, headers=headers, json=payload)
            if response.status_code == 200:
                print(f"✅ {model}: 접근 가능")
            elif response.status_code == 403:
                print(f"❌ {model}: 권한 없음")
            elif response.status_code == 404:
                print(f"⚠️ {model}: 모델 없음")
            else:
                print(f"⚠️ {model}: {response.status_code}")
        except Exception as e:
            print(f"❌ {model}: 네트워크 오류")

check_account_permissions()

def try_alternative_models(tuning_data):
    """대안 모델들로 튜닝 시도"""
    from dotenv import load_dotenv
    import os
    import requests
    
    load_dotenv()
    api_key = os.getenv('CLOVA_API_KEY')
    
    headers = {
        'Authorization': f'Bearer {api_key}',
        'Content-Type': 'application/json'
    }
    
    # 튜닝 가능한 대안 모델들
    alternative_models = ['HCX-003']
    
    for model in alternative_models:
        print(f"\n{model} 모델로 튜닝 시도 중...")
        
        url = 'https://clovastudio.stream.ntruss.com/testapp/v1/api-tools/tuning/tasks'
        
        payload = {
            "name": f"{model.lower()}_tuning_{int(time.time())}",
            "baseModel": model,
            "description": f"{model} ETF 시계열 예측 모델 튜닝",
            "dataset": tuning_data,
            "hyperParameters": {
                "epochs": 3,
                "learningRate": 0.00001,
                "batchSize": 4
            }
        }
        
        try:
            response = requests.post(url, headers=headers, json=payload)
            print(f"응답: {response.status_code} - {response.text}")
            
            if response.status_code in [200, 201]:
                result = response.json()
                task_id = result.get('taskId') or result.get('id')
                if task_id:
                    print(f"✅ {model} 튜닝 시작 성공: {task_id}")
                    return task_id, model
        except Exception as e:
            print(f"❌ {model} 오류: {e}")
    
    return None, None

def create_minimal_test_dataset():
    """최소한의 테스트 데이터셋 생성"""
    import json
    
    # 50개의 간단한 테스트 데이터 생성
    test_data = []
    for i in range(50):
        sample = {
            "messages": [
                {
                    "role": "user",
                    "content": f"숫자 {i}에 1을 더하면?"
                },
                {
                    "role": "assistant", 
                    "content": f"{i + 1}"
                }
            ]
        }
        test_data.append(sample)
    
    # 파일 저장
    with open('minimal_test_data.jsonl', 'w', encoding='utf-8') as f:
        for item in test_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print("✅ 최소 테스트 데이터셋 생성 완료: minimal_test_data.jsonl")
    return test_data

# 최소 데이터로 테스트
minimal_data = create_minimal_test_dataset()

def comprehensive_troubleshooting():
    """종합적인 문제 해결 점검"""
    print("=== HyperCLOVA X 튜닝 문제 해결 종합 점검 ===\n")
    
    # 1. 데이터 점검
    print("1. 데이터 점검 중...")
    data_ok = comprehensive_data_check('fixed_tuning_data_20250628_122248.jsonl')
    
    # 2. 계정 권한 점검  
    print("\n2. 계정 권한 점검 중...")
    check_account_permissions()
    
    # 3. 최소 데이터로 테스트
    if not data_ok:
        print("\n3. 최소 데이터셋으로 테스트...")
        minimal_data = create_minimal_test_dataset()
        task_id, model = try_alternative_models(minimal_data)
        if task_id:
            print(f"✅ 최소 데이터로 {model} 튜닝 성공!")
            return task_id, model
    
    # 4. 대안 모델 시도
    print("\n4. 대안 모델로 튜닝 시도...")
    with open('fixed_tuning_data_20250628_122248.jsonl', 'r', encoding='utf-8') as f:
        original_data = [json.loads(line) for line in f if line.strip()]
    
    task_id, model = try_alternative_models(original_data[:50])  # 처음 50개만
    
    if task_id:
        print(f"✅ {model} 모델로 튜닝 성공!")
        return task_id, model
    else:
        print("\n❌ 모든 방법 실패")
        print("권장 해결책:")
        print("1. CLOVA Studio 웹 인터페이스 사용")
        print("2. 네이버 클라우드 고객센터 문의")
        print("3. 계정 권한 재검토")
        return None, None

# 실행
comprehensive_troubleshooting()
