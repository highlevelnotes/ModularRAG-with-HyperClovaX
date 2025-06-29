# scripts/collect_data.py
import os
import sys
import json
import pandas as pd
import logging
from datetime import datetime
from pathlib import Path

# 프로젝트 루트를 Python path에 추가
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from config.config import config
from src.collector.etf_collector import KoreanETFCollector
from src.collector.data_processor import ETFPriceDataCollector
from src.collector.time_llm_converter import TimeLLMConverter

def setup_directories():
    """데이터 저장 디렉토리 생성"""
    directories = [
        config.data_dir,
        config.raw_data_dir,
        config.processed_data_dir,
        config.time_llm_data_dir,
        "logs"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)

def setup_logging():
    """로깅 설정"""
    log_filename = f"logs/data_collection_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
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
    """메인 데이터 수집 실행 함수"""
    print("="*80)
    print("🚀 Time-LLM용 국내 ETF 데이터 수집 시스템 실행")
    print("="*80)
    
    # 환경 설정
    setup_directories()
    logger = setup_logging()
    
    try:
        # 1. ETF 목록 수집 및 필터링
        logger.info("1단계: ETF 목록 수집 및 필터링")
        etf_collector = KoreanETFCollector(config)
        etf_list_df = etf_collector.get_comprehensive_etf_list()
        
        if etf_list_df.empty:
            logger.error("ETF 목록 수집 실패")
            return
        
        # ETF 목록 저장
        etf_list_path = f"{config.raw_data_dir}/pension_eligible_etfs.csv"
        etf_list_df.to_csv(etf_list_path, index=False, encoding='utf-8-sig')
        logger.info(f"퇴직연금 투자 가능 ETF 목록 저장: {etf_list_path}")
        
        # 테스트용 ETF 수 제한
        if len(etf_list_df) > config.max_etfs:
            etf_list_df = etf_list_df.head(config.max_etfs)
            logger.info(f"테스트용으로 {config.max_etfs}개 ETF로 제한")
        
        etf_codes = etf_list_df['code'].tolist()
        
        # 2. 가격 데이터 수집
        logger.info("2단계: ETF 가격 데이터 수집")
        price_collector = ETFPriceDataCollector(config)
        price_data = price_collector.collect_all_etf_prices(etf_codes)
        
        if not price_data:
            logger.error("가격 데이터 수집 실패")
            return
        
        # 가격 데이터 저장 (개별 파일)
        for etf_code, df in price_data.items():
            price_path = f"{config.raw_data_dir}/{etf_code}_price_data.csv"
            df.to_csv(price_path, index=False, encoding='utf-8-sig')
        
        logger.info(f"가격 데이터 저장 완료: {len(price_data)}개 ETF")
        
        # 3. Time-LLM 형식 변환
        logger.info("3단계: Time-LLM 형식 변환")
        time_llm_converter = TimeLLMConverter(config)
        time_llm_data = time_llm_converter.convert_to_time_llm_format(price_data)
        
        if not time_llm_data:
            logger.error("Time-LLM 형식 변환 실패")
            return
        
        # Time-LLM 데이터 저장
        time_llm_path = f"{config.time_llm_data_dir}/time_llm_training_data.json"
        with open(time_llm_path, 'w', encoding='utf-8') as f:
            json.dump(time_llm_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Time-LLM 학습 데이터 저장: {time_llm_path}")
        
        # 4. HyperClova X 파인튜닝 형식 생성
        logger.info("4단계: HyperClova X 파인튜닝 형식 생성")
        hyperclova_data = time_llm_converter.create_hyperclova_x_format(time_llm_data)
        
        hyperclova_path = f"{config.time_llm_data_dir}/hyperclova_x_tuning_data.json"
        with open(hyperclova_path, 'w', encoding='utf-8') as f:
            json.dump(hyperclova_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"HyperClova X 튜닝 데이터 저장: {hyperclova_path}")
        
        # 5. 수집 결과 요약
        print("\n" + "="*80)
        print("📊 데이터 수집 결과 요약")
        print("="*80)
        print(f"✅ 퇴직연금 투자 가능 ETF: {len(etf_list_df)}개")
        print(f"✅ 가격 데이터 수집 성공: {len(price_data)}개")
        print(f"✅ Time-LLM 학습 샘플: {len(time_llm_data)}개")
        print(f"✅ HyperClova X 튜닝 샘플: {len(hyperclova_data)}개")
        
        # 저장된 파일 목록
        print(f"\n📁 저장된 파일:")
        print(f"   • ETF 목록: {etf_list_path}")
        print(f"   • Time-LLM 데이터: {time_llm_path}")
        print(f"   • HyperClova X 데이터: {hyperclova_path}")
        print(f"   • 개별 가격 데이터: {config.raw_data_dir}/*_price_data.csv")
        
        print(f"\n🎯 다음 단계: HyperClova X에서 {hyperclova_path} 파일을 업로드하여 파인튜닝을 진행하세요.")
        print("="*80)
        
    except Exception as e:
        logger.error(f"데이터 수집 중 오류 발생: {str(e)}")
        raise

if __name__ == "__main__":
    main()
