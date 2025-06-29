# scripts/run_hyperclova_tuning.py
import os
import sys
import json
import logging
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# 프로젝트 루트를 Python path에 추가
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.hyperclova_api.tuning_client import HyperClovaXTuningClient
from src.hyperclova_api.prediction_client import HyperClovaXPredictionClient

def setup_directories():
    """디렉토리 생성"""
    directories = [
        "data/hyperclova_tuning",
        "results/hyperclova_results",
        "logs"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)

def setup_logging():
    """로깅 설정"""
    log_filename = f"logs/hyperclova_tuning_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)

def main():
    """메인 HyperClova X 파인튜닝 실행"""
    print("="*80)
    print("🚀 순수 HyperClova X API 기반 Time-LLM 파인튜닝 시스템 실행")
    print("="*80)
    
    # 환경 설정
    setup_directories()
    logger = setup_logging()
    
    # API 키 확인
    api_key = os.getenv("CLOVASTUDIO_API_KEY")
    if not api_key:
        logger.error("CLOVASTUDIO_API_KEY가 설정되지 않았습니다.")
        print("\n❌ 오류: API 키가 설정되지 않았습니다.")
        print("📝 .env 파일에 다음 내용을 추가하세요:")
        print("   CLOVASTUDIO_API_KEY=nv-************")
        return
    
    try:
        # 1. 튜닝 클라이언트 초기화
        logger.info("1단계: HyperClova X 튜닝 클라이언트 초기화")
        tuning_client = HyperClovaXTuningClient()
        
        # 2. 튜닝 데이터 준비
        logger.info("2단계: 튜닝 데이터 준비")
        json_file_path = "data/time_llm_format/hyperclova_x_tuning_data.json"
        
        if not os.path.exists(json_file_path):
            logger.error(f"튜닝 데이터 파일을 찾을 수 없습니다: {json_file_path}")
            return
        
        jsonl_file_path = tuning_client.prepare_tuning_data(json_file_path)
        
        # 3. 파일 업로드
        logger.info("3단계: 튜닝 파일 업로드")
        file_id = tuning_client.upload_training_file(jsonl_file_path)
        
        # 4. 튜닝 작업 생성
        logger.info("4단계: 튜닝 작업 생성")
        task_id = tuning_client.create_tuning_job(file_id)
        
        # 5. 튜닝 완료 대기
        logger.info("5단계: 튜닝 완료 대기")
        print("\n⏳ 튜닝이 진행 중입니다...")
        model_id = tuning_client.wait_for_completion(task_id)
        
        # 6. 튜닝된 모델 테스트
        logger.info("6단계: 튜닝된 모델 테스트")
        predictor = HyperClovaXPredictionClient(model_id=model_id)
        
        # 간단한 테스트
        sample_prices = [25400, 25600, 25300, 25800, 25900] * 6  # 30일치
        result = predictor.predict_etf_prices("KODEX200", sample_prices)
        
        # 7. 결과 저장
        results_path = f"results/hyperclova_results/tuning_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        final_results = {
            'tuning_info': {
                'task_id': task_id,
                'model_id': model_id,
                'file_id': file_id,
                'jsonl_file_path': jsonl_file_path,
                'completion_time': datetime.now().isoformat()
            },
            'test_prediction': result,
            'api_type': 'Pure HyperClova X API'
        }
        
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(final_results, f, ensure_ascii=False, indent=2)
        
        # 8. 결과 요약
        print("\n" + "="*80)
        print("📊 HyperClova X 파인튜닝 결과 요약")
        print("="*80)
        print(f"✅ 튜닝 작업 ID: {task_id}")
        print(f"✅ 튜닝된 모델 ID: {model_id}")
        print(f"✅ 테스트 예측 완료")
        print(f"✅ 결과 저장: {results_path}")
        print(f"✅ 순수 HyperClova X API 사용")
        print("="*80)
        
    except Exception as e:
        logger.error(f"파인튜닝 과정 중 오류 발생: {str(e)}")
        print(f"\n❌ 오류 발생: {str(e)}")
        raise

if __name__ == "__main__":
    main()
