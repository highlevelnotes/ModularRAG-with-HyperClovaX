# scripts/run_langchain_hyperclova_tuning.py
import os
import sys
import json
import logging
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

# 환경 변수 로드
load_dotenv()

# 프로젝트 루트를 Python path에 추가
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.langchain_integration.time_llm_data_processor import TimeLLMDataProcessor
from src.langchain_integration.hyperclova_tuning_client import LangChainHyperClovaTuningClient
from src.langchain_integration.time_llm_predictor import LangChainTimeLLMPredictor

def setup_directories():
    """디렉토리 생성"""
    directories = [
        "data/langchain_tuning",
        "results/langchain_results",
        "logs"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)

def setup_logging():
    """로깅 설정"""
    log_filename = f"logs/langchain_tuning_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
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
    """메인 LangChain HyperClova X 파인튜닝 실행"""
    print("="*80)
    print("🚀 LangChain 기반 HyperClova X Time-LLM 파인튜닝 시스템 실행")
    print("="*80)
    
    # 환경 설정
    setup_directories()
    logger = setup_logging()
    
    # 환경 변수 확인
    api_key = os.getenv("NCP_CLOVASTUDIO_API_KEY")
    apigw_key = os.getenv("NCP_APIGW_API_KEY")
    
    if not api_key or not apigw_key:
        logger.error("API 키가 설정되지 않았습니다. .env 파일을 확인하세요.")
        print("\n❌ 오류: API 키가 설정되지 않았습니다.")
        print("📝 .env 파일에 다음 내용을 추가하세요:")
        print("   NCP_CLOVASTUDIO_API_KEY=your_clova_studio_api_key")
        print("   NCP_APIGW_API_KEY=your_apigw_api_key")
        return
    
    try:
        # 1. 데이터 처리기 초기화
        logger.info("1단계: LangChain 데이터 처리기 초기화")
        data_processor = TimeLLMDataProcessor()
        
        # 2. 튜닝 데이터 준비
        logger.info("2단계: LangChain 튜닝 데이터 준비")
        json_file_path = "data/time_llm_format/hyperclova_x_tuning_data.json"
        
        if not os.path.exists(json_file_path):
            logger.error(f"튜닝 데이터 파일을 찾을 수 없습니다: {json_file_path}")
            print(f"\n❌ 오류: 튜닝 데이터 파일을 찾을 수 없습니다.")
            print(f"📁 다음 경로에 파일이 있는지 확인하세요: {json_file_path}")
            print("💡 먼저 데이터 수집 스크립트를 실행하세요: python scripts/collect_data.py")
            return
        
        langchain_data = data_processor.prepare_langchain_tuning_data(json_file_path)
        
        # 3. JSONL 파일 저장
        logger.info("3단계: LangChain 튜닝 파일 저장")
        jsonl_file_path = f"data/langchain_tuning/time_llm_tuning_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
        data_processor.save_langchain_tuning_file(langchain_data, jsonl_file_path)
        
        # 4. 튜닝 클라이언트 초기화
        logger.info("4단계: LangChain 튜닝 클라이언트 초기화")
        tuning_client = LangChainHyperClovaTuningClient()
        
        # 5. 파일 업로드
        logger.info("5단계: 튜닝 파일 업로드")
        file_id = tuning_client.upload_training_file(jsonl_file_path)
        
        # 6. 튜닝 작업 생성
        logger.info("6단계: 튜닝 작업 생성")
        task_id = tuning_client.create_tuning_job(file_id)
        
        # 7. 튜닝 완료 대기
        logger.info("7단계: 튜닝 완료 대기")
        print("\n⏳ 튜닝이 진행 중입니다. 완료까지 시간이 걸릴 수 있습니다...")
        print("   (Ctrl+C를 눌러 대기를 중단할 수 있습니다)")
        
        model_id = tuning_client.wait_for_completion(task_id)
        
        if not model_id:
            logger.error("튜닝이 완료되지 않았습니다.")
            return
        
        # 8. 튜닝된 모델 테스트
        logger.info("8단계: 튜닝된 모델 테스트")
        predictor = LangChainTimeLLMPredictor(model_id=model_id)
        
        # 배치 테스트 실행
        test_results = predictor.batch_predict(json_file_path, num_samples=5)
        
        # 9. 결과 저장
        results_path = f"results/langchain_results/tuning_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        final_results = {
            'tuning_info': {
                'task_id': task_id,
                'model_id': model_id,
                'file_id': file_id,
                'jsonl_file_path': jsonl_file_path,
                'completion_time': datetime.now().isoformat()
            },
            'test_results': test_results,
            'langchain_config': {
                'framework': 'LangChain',
                'model_type': 'ChatClovaX',
                'api_keys_used': ['NCP_CLOVASTUDIO_API_KEY', 'NCP_APIGW_API_KEY']
            }
        }
        
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(final_results, f, ensure_ascii=False, indent=2)
        
        # 10. 결과 요약
        print("\n" + "="*80)
        print("📊 LangChain HyperClova X 파인튜닝 결과 요약")
        print("="*80)
        print(f"✅ 튜닝 작업 ID: {task_id}")
        print(f"✅ 튜닝된 모델 ID: {model_id}")
        print(f"✅ 테스트 샘플 수: {test_results['total_samples']}")
        print(f"✅ 평균 MSE: {test_results['average_mse']:.6f}")
        print(f"✅ 결과 저장: {results_path}")
        print(f"✅ LangChain 프레임워크 사용 (dotenv 환경 변수 관리)")
        
        print(f"\n🎯 다음 단계:")
        print(f"   • 모델 ID '{model_id}'를 사용하여 LangChain 기반 ETF 예측 서비스 구축")
        print(f"   • 퇴직연금 포트폴리오 최적화 AI 에이전트에 통합")
        print("="*80)
        
    except Exception as e:
        logger.error(f"LangChain 파인튜닝 과정 중 오류 발생: {str(e)}")
        print(f"\n❌ 오류 발생: {str(e)}")
        print("📋 로그 파일을 확인하여 자세한 오류 내용을 확인하세요.")
        raise

if __name__ == "__main__":
    main()
