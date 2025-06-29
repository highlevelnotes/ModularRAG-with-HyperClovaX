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

from src.tuning.hyperclova_tuning_client import HyperClovaXTuningClient
from src.tuning.model_tester import TimeLLMTester

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
    print("🚀 HyperClova X Time-LLM 파인튜닝 시스템 실행")
    print("="*80)
    
    # 환경 설정
    logger = setup_logging()
    
    # API 키 설정 (환경변수에서 가져오기)
    api_key = os.getenv('CLOVA_STUDIO_API_KEY')
    api_key_primary = os.getenv('CLOVA_STUDIO_API_KEY_PRIMARY')
    
    if not api_key or not api_key_primary:
        logger.error("API 키가 설정되지 않았습니다. 환경변수를 확인하세요.")
        return
    
    try:
        # 1. 튜닝 클라이언트 초기화
        logger.info("1단계: HyperClova X 튜닝 클라이언트 초기화")
        tuning_client = HyperClovaXTuningClient(
            api_key=api_key,
            api_key_primary=api_key_primary
        )
        
        # 2. 튜닝 데이터 준비
        logger.info("2단계: 튜닝 데이터 준비")
        json_file_path = "data/time_llm_format/hyperclova_x_tuning_data.json"
        
        if not os.path.exists(json_file_path):
            logger.error(f"튜닝 데이터 파일을 찾을 수 없습니다: {json_file_path}")
            return
        
        tuning_data = tuning_client.prepare_tuning_data(json_file_path)
        
        # 3. 파일 업로드
        logger.info("3단계: 튜닝 파일 업로드")
        file_id = tuning_client.upload_training_file(tuning_data)
        
        # 4. 튜닝 작업 생성
        logger.info("4단계: 튜닝 작업 생성")
        task_id = tuning_client.create_tuning_job(file_id)
        
        # 5. 튜닝 완료 대기
        logger.info("5단계: 튜닝 완료 대기")
        model_id = tuning_client.wait_for_completion(task_id)
        
        # 6. 모델 테스트
        logger.info("6단계: 튜닝된 모델 테스트")
        tester = TimeLLMTester(
            api_key=api_key,
            api_key_primary=api_key_primary,
            model_id=model_id
        )
        
        # 배치 테스트 실행
        test_results = tester.batch_test(json_file_path, num_samples=5)
        
        # 7. 결과 저장
        results_path = f"results/tuning_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        os.makedirs(os.path.dirname(results_path), exist_ok=True)
        
        final_results = {
            'tuning_info': {
                'task_id': task_id,
                'model_id': model_id,
                'file_id': file_id,
                'completion_time': datetime.now().isoformat()
            },
            'test_results': test_results
        }
        
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(final_results, f, ensure_ascii=False, indent=2)
        
        # 8. 결과 요약
        print("\n" + "="*80)
        print("📊 HyperClova X 파인튜닝 결과 요약")
        print("="*80)
        print(f"✅ 튜닝 작업 ID: {task_id}")
        print(f"✅ 튜닝된 모델 ID: {model_id}")
        print(f"✅ 테스트 샘플 수: {test_results['total_samples']}")
        print(f"✅ 평균 MSE: {test_results['average_mse']:.6f}")
        print(f"✅ 결과 저장: {results_path}")
        
        print(f"\n🎯 다음 단계:")
        print(f"   • 모델 ID '{model_id}'를 사용하여 실제 ETF 예측 서비스 구축")
        print(f"   • 퇴직연금 포트폴리오 최적화 AI 에이전트에 통합")
        print("="*80)
        
    except Exception as e:
        logger.error(f"파인튜닝 과정 중 오류 발생: {str(e)}")
        raise

if __name__ == "__main__":
    main()
