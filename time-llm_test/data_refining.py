import json

def convert_to_hyperclova_instruction_format(input_file, output_file):
    """
    기존 messages 형식을 HyperCLOVA X Instruction 형식(C_ID, T_ID 포함)으로 변환
    """
    converted_data = []
    c_id = 0  # 대화 시나리오 ID
    
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                if line.strip():
                    try:
                        data = json.loads(line)
                        
                        if 'messages' in data:
                            user_content = ""
                            assistant_content = ""
                            
                            for msg in data['messages']:
                                if msg.get('role') == 'user':
                                    user_content = msg.get('content', '')
                                elif msg.get('role') == 'assistant':
                                    assistant_content = msg.get('content', '')
                            
                            if user_content and assistant_content:
                                # HyperCLOVA X Instruction 형식으로 변환
                                converted_item = {
                                    "C_ID": c_id,
                                    "T_ID": 0,  # 단일 턴이므로 항상 0
                                    "Text": user_content,
                                    "Completion": assistant_content
                                }
                                converted_data.append(converted_item)
                                c_id += 1  # 다음 대화 시나리오로 증가
                        
                    except json.JSONDecodeError as e:
                        print(f"라인 {line_num} 파싱 오류: {e}")
                        continue
        
        # 새 파일로 저장 (JSONL 형식)
        with open(output_file, 'w', encoding='utf-8') as f:
            for item in converted_data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        
        print(f"✅ HyperCLOVA X Instruction 형식 변환 완료:")
        print(f"  입력: {input_file}")
        print(f"  출력: {output_file}")
        print(f"  변환된 데이터: {len(converted_data)}개")
        print(f"  C_ID 범위: 0 ~ {c_id-1}")
        
        # 최소 데이터 요구량 확인
        if len(converted_data) < 400:
            print(f"⚠️ 데이터 부족: {len(converted_data)}개 (최소 400개 필요)")
        else:
            print(f"✅ 데이터 요구량 충족: {len(converted_data)}개")
        
        return output_file, len(converted_data)
        
    except Exception as e:
        print(f"❌ 변환 실패: {e}")
        return None, 0

def create_csv_format(jsonl_file, csv_file):
    """
    JSONL을 CSV 형식으로도 생성 (C_ID, T_ID, Text, Completion 순서)
    """
    import pandas as pd
    
    data = []
    
    with open(jsonl_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                data.append({
                    'C_ID': item['C_ID'],
                    'T_ID': item['T_ID'],
                    'Text': item['Text'],
                    'Completion': item['Completion']
                })
    
    df = pd.DataFrame(data)
    # 컬럼 순서 확인: C_ID, T_ID, Text, Completion
    df = df[['C_ID', 'T_ID', 'Text', 'Completion']]
    df.to_csv(csv_file, index=False, encoding='utf-8-sig')
    
    print(f"✅ CSV 파일 생성: {csv_file}")
    print(f"첫 번째 행: C_ID, T_ID, Text, Completion")
    return csv_file

def validate_instruction_dataset(file_path):
    """
    Instruction 데이터셋 유효성 검사
    """
    print(f"\n=== {file_path} 유효성 검사 ===")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        valid_count = 0
        error_count = 0
        
        for line_num, line in enumerate(f, 1):
            if line.strip():
                try:
                    item = json.loads(line)
                    
                    # 필수 필드 확인
                    required_fields = ['C_ID', 'T_ID', 'Text', 'Completion']
                    if all(field in item for field in required_fields):
                        # C_ID, T_ID가 숫자인지 확인
                        if isinstance(item['C_ID'], int) and isinstance(item['T_ID'], int):
                            # Text, Completion이 비어있지 않은지 확인
                            if item['Text'].strip() and item['Completion'].strip():
                                valid_count += 1
                            else:
                                print(f"라인 {line_num}: Text 또는 Completion이 비어있음")
                                error_count += 1
                        else:
                            print(f"라인 {line_num}: C_ID 또는 T_ID가 숫자가 아님")
                            error_count += 1
                    else:
                        print(f"라인 {line_num}: 필수 필드 누락")
                        error_count += 1
                        
                except json.JSONDecodeError as e:
                    print(f"라인 {line_num}: JSON 파싱 오류 - {e}")
                    error_count += 1
    
    print(f"✅ 유효한 데이터: {valid_count}개")
    print(f"❌ 오류 데이터: {error_count}개")
    print(f"총 데이터: {valid_count + error_count}개")
    
    return valid_count > 0

# 실행
if __name__ == '__main__':
    print("=== HyperCLOVA X Instruction 데이터셋 변환 ===\n")
    
    # 1단계: 기존 파일을 Instruction 형식으로 변환
    input_file = 'fixed_tuning_data_20250628_122248.jsonl'
    output_jsonl = 'hyperclova_instruction_dataset.jsonl'
    output_csv = 'hyperclova_instruction_dataset.csv'
    
    # JSONL 형식으로 변환
    new_file, count = convert_to_hyperclova_instruction_format(input_file, output_jsonl)
    
    if new_file and count > 0:
        # CSV 형식으로도 생성
        csv_file = create_csv_format(output_jsonl, output_csv)
        
        # 유효성 검사
        validate_instruction_dataset(output_jsonl)
        
        print(f"\n=== 완료 ===")
        print(f"JSONL 파일: {output_jsonl}")
        print(f"CSV 파일: {output_csv}")
        print(f"Object Storage에 업로드할 파일: {output_jsonl} 또는 {output_csv}")
        
        # 샘플 데이터 출력
        print(f"\n=== 샘플 데이터 ===")
        with open(output_jsonl, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i < 2:  # 처음 2개만 출력
                    sample = json.loads(line)
                    print(f"샘플 {i+1}: {sample}")
                else:
                    break
    else:
        print("❌ 변환 실패")
