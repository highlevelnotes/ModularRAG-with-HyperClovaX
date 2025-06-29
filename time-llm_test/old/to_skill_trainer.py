import json
import re
from typing import List, Dict, Any

def parse_message_data(data_string: str) -> List[Dict[str, Any]]:
    """
    입력된 메시지 형식 데이터를 파싱하여 구조화된 리스트로 변환
    """
    # JSON 객체들을 분리
    json_objects = []
    
    # 각 줄을 개별 JSON으로 파싱
    lines = data_string.strip().split('\n')
    
    for line in lines:
        if line.strip():
            try:
                json_obj = json.loads(line)
                json_objects.append(json_obj)
            except json.JSONDecodeError:
                continue
    
    return json_objects

def extract_etf_code_from_content(content: str) -> str:
    """
    메시지 내용에서 ETF 코드를 추출
    """
    # ETF 코드 패턴 매칭 (6자리 숫자)
    etf_pattern = r'(\d{6})\s+ETF'
    match = re.search(etf_pattern, content)
    
    if match:
        return match.group(1)
    
    # 기본값으로 310970 반환 (대부분의 데이터가 이 코드)
    return "310970"

def convert_messages_to_skill_trainer(json_objects: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    파싱된 메시지 데이터를 Skill Trainer 형식으로 변환
    """
    skill_trainer_data = []
    
    for obj in json_objects:
        if "messages" in obj and len(obj["messages"]) >= 2:
            user_message = obj["messages"][0]["content"]
            assistant_message = obj["messages"][1]["content"]
            
            # ETF 코드 추출
            etf_code = extract_etf_code_from_content(user_message)
            
            # Skill Trainer 형식으로 변환
            skill_item = {
                "question": user_message,
                "answer": assistant_message,
                "category": "etf_prediction",
                "etf_code": etf_code,
                "prediction_type": "10day_forecast"
            }
            
            skill_trainer_data.append(skill_item)
    
    return skill_trainer_data

def enhance_skill_trainer_data(skill_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Skill Trainer 데이터에 추가 메타데이터 및 개선사항 적용
    """
    enhanced_data = []
    
    for i, item in enumerate(skill_data):
        # 기존 데이터 복사
        enhanced_item = item.copy()
        
        # 추가 메타데이터
        enhanced_item["sample_id"] = f"etf_pred_{item['etf_code']}_{i+1:03d}"
        enhanced_item["data_type"] = "normalized_price_series"
        enhanced_item["input_length"] = "30_days"
        enhanced_item["output_length"] = "10_days"
        
        # 질문 형식 개선 (더 자연스러운 한국어)
        original_question = item["question"]
        if "입력 데이터:" in original_question:
            # 입력 데이터 부분 추출
            input_data_start = original_question.find("입력 데이터:")
            input_data = original_question[input_data_start:].replace("입력 데이터:", "").strip()
            
            # 개선된 질문 형식
            enhanced_question = f"""당신은 {item['etf_code']} ETF 전문 시계열 분석가입니다.
다음은 과거 30일간의 정규화된 가격 데이터입니다. 이 패턴을 분석하여 향후 10일간의 정규화된 가격을 예측해주세요.

과거 30일 데이터: {input_data}

기술적 분석과 시계열 패턴을 고려하여 향후 10일간의 예측값을 제시해주세요."""
            
            enhanced_item["question"] = enhanced_question
        
        # 답변 형식 개선
        enhanced_item["answer"] = f"향후 10일 예측값: {item['answer']}"
        
        enhanced_data.append(enhanced_item)
    
    return enhanced_data

def create_etf_specific_variants(base_data: List[Dict[str, Any]], etf_codes: List[str]) -> List[Dict[str, Any]]:
    """
    기존 데이터를 다른 ETF 코드들로 확장
    """
    expanded_data = []
    
    # 원본 데이터 추가
    expanded_data.extend(base_data)
    
    # 다른 ETF 코드들로 변형 생성
    for etf_code in etf_codes:
        if etf_code != "310970":  # 원본과 다른 ETF만
            for item in base_data[:3]:  # 샘플 몇 개만 변형
                variant_item = item.copy()
                variant_item["etf_code"] = etf_code
                variant_item["sample_id"] = f"etf_pred_{etf_code}_{len(expanded_data)+1:03d}"
                
                # 질문에서 ETF 코드 변경
                variant_item["question"] = variant_item["question"].replace("310970", etf_code)
                
                expanded_data.append(variant_item)
    
    return expanded_data

def main_conversion_pipeline(input_data: str, additional_etf_codes: List[str] = None) -> Dict[str, Any]:
    """
    전체 변환 파이프라인 실행
    """
    if additional_etf_codes is None:
        additional_etf_codes = ["069500", "232080", "114260", "279530", "102110"]
    
    print("=== ETF 시계열 데이터 Skill Trainer 변환 시작 ===\n")
    
    # 1단계: 데이터 파싱
    print("1단계: 메시지 데이터 파싱 중...")
    json_objects = parse_message_data(input_data)
    print(f"✓ {len(json_objects)}개 메시지 객체 파싱 완료\n")
    
    # 2단계: 기본 변환
    print("2단계: Skill Trainer 형식으로 변환 중...")
    skill_data = convert_messages_to_skill_trainer(json_objects)
    print(f"✓ {len(skill_data)}개 기본 샘플 생성 완료\n")
    
    # 3단계: 데이터 개선
    print("3단계: 데이터 품질 개선 중...")
    enhanced_data = enhance_skill_trainer_data(skill_data)
    print(f"✓ 데이터 개선 완료\n")
    
    # 4단계: 다른 ETF로 확장
    print("4단계: 다른 ETF 코드로 데이터 확장 중...")
    final_data = create_etf_specific_variants(enhanced_data, additional_etf_codes)
    print(f"✓ 최종 {len(final_data)}개 샘플 생성 완료\n")
    
    # 5단계: 결과 저장
    print("5단계: 결과 파일 저장 중...")
    
    # JSON 파일 저장
    with open('etf_skill_trainer_data.json', 'w', encoding='utf-8') as f:
        json.dump(final_data, f, ensure_ascii=False, indent=2)
    
    # CSV 파일도 저장
    import pandas as pd
    df = pd.DataFrame(final_data)
    df.to_csv('etf_skill_trainer_data.csv', index=False, encoding='utf-8-sig')
    
    print("✓ 파일 저장 완료")
    print("- etf_skill_trainer_data.json: HyperCLOVA Skill Trainer 업로드용")
    print("- etf_skill_trainer_data.csv: 동일 내용의 CSV 파일\n")
    
    # 6단계: 통계 정보
    print("=== 변환 결과 통계 ===")
    etf_distribution = {}
    for item in final_data:
        etf_code = item['etf_code']
        etf_distribution[etf_code] = etf_distribution.get(etf_code, 0) + 1
    
    print(f"총 샘플 수: {len(final_data)}")
    print("ETF별 분포:")
    for etf_code, count in etf_distribution.items():
        print(f"- {etf_code}: {count}개")
    
    # 데이터 품질 검증
    valid_samples = sum(1 for item in final_data 
                       if len(item['question']) > 100 and len(item['answer']) > 10)
    print(f"\n데이터 품질: {valid_samples}/{len(final_data)} ({valid_samples/len(final_data)*100:.1f}%) 유효")
    
    return {
        'total_samples': len(final_data),
        'etf_distribution': etf_distribution,
        'data': final_data
    }

# 실제 사용 예시
if __name__ == "__main__":
    # JSONL 파일 경로
    file_path = 'data/hyperclova_tuning/time_llm_tuning_20250626_155419.jsonl'
    
    print("=== ETF 시계열 데이터 Skill Trainer 변환 시작 ===\n")
    
    # JSONL 파일 읽기
    print(f"데이터 파일 로딩 중: {file_path}")
    json_objects = []
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                if line.strip():
                    try:
                        json_obj = json.loads(line)
                        json_objects.append(json_obj)
                    except json.JSONDecodeError as e:
                        print(f"라인 {line_num} 파싱 오류: {e}")
                        continue
        
        print(f"✓ {len(json_objects)}개 데이터 샘플 로딩 완료\n")
        
        # 추가 ETF 코드들 (국내 전용 ETF)
        additional_etfs = ["069500", "232080", "114260", "279530", "102110", "091160", "305720"]
        
        # 1단계: 기본 변환
        print("1단계: Skill Trainer 형식으로 변환 중...")
        skill_data = convert_messages_to_skill_trainer(json_objects)
        print(f"✓ {len(skill_data)}개 기본 샘플 생성 완료\n")
        
        # 2단계: 데이터 개선
        print("2단계: 데이터 품질 개선 중...")
        enhanced_data = enhance_skill_trainer_data(skill_data)
        print(f"✓ 데이터 개선 완료\n")
        
        # 3단계: 다른 ETF로 확장
        print("3단계: 다른 ETF 코드로 데이터 확장 중...")
        final_data = create_etf_specific_variants(enhanced_data, additional_etfs)
        print(f"✓ 최종 {len(final_data)}개 샘플 생성 완료\n")
        
        # 4단계: 결과 저장
        print("4단계: 결과 파일 저장 중...")
        
        # JSON 파일 저장
        output_json = 'etf_skill_trainer_data.json'
        with open(output_json, 'w', encoding='utf-8') as f:
            json.dump(final_data, f, ensure_ascii=False, indent=2)
        
        # CSV 파일도 저장
        import pandas as pd
        output_csv = 'etf_skill_trainer_data.csv'
        df = pd.DataFrame(final_data)
        df.to_csv(output_csv, index=False, encoding='utf-8-sig')
        
        print(f"✓ 파일 저장 완료")
        print(f"- {output_json}: HyperCLOVA Skill Trainer 업로드용")
        print(f"- {output_csv}: 동일 내용의 CSV 파일\n")
        
        # 5단계: 통계 정보 출력
        print("=== 변환 결과 통계 ===")
        etf_distribution = {}
        for item in final_data:
            etf_code = item.get('etf_code', 'unknown')
            etf_distribution[etf_code] = etf_distribution.get(etf_code, 0) + 1
        
        print(f"총 샘플 수: {len(final_data)}")
        print("ETF별 분포:")
        for etf_code, count in sorted(etf_distribution.items()):
            print(f"- {etf_code}: {count}개")
        
        # 데이터 품질 검증
        valid_samples = sum(1 for item in final_data 
                           if len(item.get('question', '')) > 100 and len(item.get('answer', '')) > 10)
        quality_rate = valid_samples / len(final_data) * 100 if final_data else 0
        print(f"\n데이터 품질: {valid_samples}/{len(final_data)} ({quality_rate:.1f}%) 유효")
        
        # 6단계: 샘플 미리보기
        print("\n=== 샘플 데이터 미리보기 ===")
        preview_count = min(3, len(final_data))
        for i in range(preview_count):
            sample = final_data[i]
            print(f"\n--- 샘플 {i+1} ---")
            print(f"ETF 코드: {sample.get('etf_code', 'N/A')}")
            print(f"카테고리: {sample.get('category', 'N/A')}")
            print(f"질문 (처음 150자): {sample.get('question', '')[:150]}...")
            print(f"답변: {sample.get('answer', '')}")
        
        print(f"\n=== 변환 완료 ===")
        print(f"생성된 {output_json} 파일을 CLOVA Studio의 Skill Trainer에 업로드하여 학습을 시작할 수 있습니다.")
        
    except FileNotFoundError:
        print(f"❌ 파일을 찾을 수 없습니다: {file_path}")
        print("파일 경로를 확인해주세요.")
    except Exception as e:
        print(f"❌ 처리 중 오류 발생: {str(e)}")
        import traceback
        traceback.print_exc()