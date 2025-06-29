# examples/simple_prediction_example.py
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# 환경 변수 로드
load_dotenv()

# 프로젝트 루트를 Python path에 추가
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.langchain_integration.time_llm_predictor import LangChainTimeLLMPredictor

def simple_prediction_example():
    """간단한 예측 예제"""
    
    # 튜닝된 모델 ID (실제 튜닝 완료 후 교체)
    model_id = "your_tuned_model_id_here"  # 실제 모델 ID로 교체
    
    try:
        # 예측기 초기화
        predictor = LangChainTimeLLMPredictor(model_id=model_id)
        
        # 샘플 ETF 가격 데이터 (30일치)
        sample_prices = [
            25400, 25600, 25300, 25800, 25900, 26100, 26200, 26000, 26300, 26400,
            26600, 26500, 26700, 26800, 26900, 27000, 27100, 26950, 27200, 27300,
            27400, 27500, 27600, 27700, 27800, 27900, 28000, 28100, 28200, 28300
        ]
        
        # 예측 실행
        result = predictor.predict_etf_prices("KODEX200", sample_prices)
        
        print("="*60)
        print("🔮 LangChain Time-LLM ETF 가격 예측 결과")
        print("="*60)
        print(f"ETF 이름: {result['etf_name']}")
        print(f"입력 가격 (마지막 5일): {result['input_sequence'][-5:]}")
        print(f"예측 가격 (향후 10일): {[f'{p:.0f}' for p in result['predicted_prices']]}")
        print(f"예측 시간: {result['prediction_timestamp']}")
        print("="*60)
        
    except Exception as e:
        print(f"❌ 예측 실행 중 오류: {str(e)}")
        print("💡 .env 파일의 API 키와 모델 ID를 확인하세요.")

if __name__ == "__main__":
    simple_prediction_example()
