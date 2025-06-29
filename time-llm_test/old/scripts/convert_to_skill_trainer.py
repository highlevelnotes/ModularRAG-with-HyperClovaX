# scripts/convert_to_skill_trainer.py
import json
import os
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

def convert_existing_data_to_skill_trainer():
    """기존 JSON 데이터를 스킬 트레이너 형식으로 변환"""
    
    print("🔄 기존 Time-LLM 데이터를 스킬 트레이너 형식으로 변환 중...")
    
    # 기존 데이터 로드
    input_file = "data/time_llm_format/hyperclova_x_tuning_data.json"
    
    with open(input_file, 'r', encoding='utf-8') as f:
        time_llm_data = json.load(f)
    
    print(f"📊 로드된 데이터: {len(time_llm_data)}개 샘플")
    
    # 스킬 트레이너용 데이터 변환
    skill_trainer_data = []
    
    for i, sample in enumerate(time_llm_data[:100]):  # 처음 100개만 사용
        etf_code = sample.get('etf_code', 'Unknown')
        
        # 입력 시퀀스에서 가격 데이터 추출
        input_content = sample['input']
        try:
            input_sequence_text = input_content.split('입력 데이터: ')[1].split('\n')[0]
            # 정규화된 값을 실제 가격으로 변환 (예시)
            normalized_values = [float(x) for x in input_sequence_text.split()]
            # 25000~30000 범위로 스케일링
            sample_prices = [25000 + (x * 5000) for x in normalized_values]
        except:
            # 파싱 실패시 샘플 데이터 사용
            sample_prices = [25000 + i * 50 for i in range(30)]
        
        # 다양한 사용자 쿼리 패턴 생성
        query_patterns = [
            f"{etf_code} ETF의 향후 10일 가격을 예측해주세요",
            f"{etf_code} ETF 다음 주 전망은 어떤가요?",
            f"{etf_code} 투자 타이밍을 알려주세요",
            f"{etf_code} ETF가 오를까요?",
            f"{etf_code} 가격 예측해주세요"
        ]
        
        user_query = query_patterns[i % len(query_patterns)]
        
        skill_sample = {
            "user_query": user_query,
            "api_call": {
                "endpoint": "/predict_etf",
                "method": "POST", 
                "parameters": {
                    "etf_code": etf_code,
                    "price_sequence": sample_prices,
                    "prediction_days": 10
                }
            },
            "expected_output": sample['output'],
            "metadata": {
                "etf_name": sample.get('etf_name', etf_code),
                "original_sample_index": i
            }
        }
        
        skill_trainer_data.append(skill_sample)
    
    # 변환된 데이터 저장
    output_dir = "data/skill_trainer"
    os.makedirs(output_dir, exist_ok=True)
    
    output_file = f"{output_dir}/skill_trainer_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(skill_trainer_data, f, ensure_ascii=False, indent=2)
    
    print(f"✅ 스킬 트레이너 데이터 변환 완료: {len(skill_trainer_data)}개 샘플")
    print(f"📁 저장 위치: {output_file}")
    
    # 샘플 데이터 미리보기
    print("\n📋 변환된 데이터 샘플:")
    for i, sample in enumerate(skill_trainer_data[:3]):
        print(f"\n{i+1}. 사용자 쿼리: {sample['user_query']}")
        print(f"   ETF 코드: {sample['api_call']['parameters']['etf_code']}")
        print(f"   예상 출력: {sample['expected_output'][:50]}...")
    
    return output_file

if __name__ == "__main__":
    print("="*80)
    print("🚀 기존 데이터를 스킬 트레이너 형식으로 변환")
    print("="*80)
    
    output_file = convert_existing_data_to_skill_trainer()
    
    print("\n" + "="*80)
    print("📋 다음 단계: CLOVA Studio 스킬 트레이너 설정")
    print("="*80)
    print("1. CLOVA Studio 콘솔 → 스킬 트레이너 접속")
    print("2. 스킬셋 생성: 'etf-time-llm-predictor'")
    print("3. 스킬 생성 및 API 스펙 등록")
    print(f"4. 변환된 데이터 업로드: {output_file}")
    print("5. 학습 시작 → 테스트 앱 발행")
    print("="*80)
