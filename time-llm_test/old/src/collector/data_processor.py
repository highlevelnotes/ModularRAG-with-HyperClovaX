# src/collector/data_processor.py
import pandas as pd
import numpy as np
import FinanceDataReader as fdr
import yfinance as yf
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
import time
from tqdm import tqdm
import logging

class ETFPriceDataCollector:
    """ETF 가격 데이터 수집기 - 다중 시간대 지원"""
    
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # 10년치 데이터 수집을 위한 기간 설정
        self.start_date_10y = (datetime.now() - timedelta(days=3650)).strftime('%Y-%m-%d')  # 10년
        self.end_date = datetime.now().strftime('%Y-%m-%d')
        
        # 시간대별 리샘플링 설정
        self.resample_rules = {
            'daily': None,      # 원본 일별 데이터
            '5day': '5D',       # 5일별
            'weekly': 'W',      # 주별 (일요일 기준)
            'monthly': 'M'      # 월별 (월말 기준)
        }
    
    def get_etf_price_data_multi_timeframe(self, etf_code: str) -> Dict[str, pd.DataFrame]:
        """개별 ETF의 다중 시간대 가격 데이터 수집"""
        try:
            self.logger.info(f"{etf_code} 다중 시간대 데이터 수집 중... ({self.start_date_10y} ~ {self.end_date})")
            
            # 일별 데이터 수집 (기본)
            daily_df = self._get_single_etf_data(etf_code, self.start_date_10y, self.end_date)
            
            if daily_df.empty:
                self.logger.warning(f"{etf_code}: 데이터 수집 실패")
                return {}
            
            # 다중 시간대 데이터 생성
            multi_timeframe_data = {}
            
            # 1. 일별 데이터 (원본)
            multi_timeframe_data['daily'] = daily_df.copy()
            
            # 2. 5일별, 주별, 월별 데이터 생성
            for timeframe, rule in self.resample_rules.items():
                if rule is None:  # daily는 이미 처리됨
                    continue
                    
                try:
                    resampled_df = self._resample_data(daily_df, rule, timeframe)
                    if not resampled_df.empty:
                        multi_timeframe_data[timeframe] = resampled_df
                        self.logger.info(f"{etf_code} {timeframe}: {len(resampled_df)}개 데이터 포인트")
                    else:
                        self.logger.warning(f"{etf_code} {timeframe}: 리샘플링 실패")
                        
                except Exception as e:
                    self.logger.error(f"{etf_code} {timeframe} 리샘플링 오류: {str(e)}")
            
            # 데이터 품질 검증
            total_points = sum(len(df) for df in multi_timeframe_data.values())
            self.logger.info(f"{etf_code}: 총 {total_points}개 데이터 포인트 수집 완료")
            
            return multi_timeframe_data
            
        except Exception as e:
            self.logger.error(f"{etf_code} 다중 시간대 데이터 수집 오류: {str(e)}")
            return {}
    
    def _get_single_etf_data(self, etf_code: str, start_date: str, end_date: str) -> pd.DataFrame:
        """개별 ETF 일별 데이터 수집"""
        try:
            # FinanceDataReader 먼저 시도
            try:
                df = fdr.DataReader(etf_code, start_date, end_date)
                if not df.empty:
                    df.reset_index(inplace=True)
                    df['code'] = etf_code
                    return self._standardize_price_data(df)
            except Exception as e:
                self.logger.warning(f"FinanceDataReader 실패 ({etf_code}): {str(e)}")
            
            # yfinance 시도 (한국 ETF는 .KS 접미사)
            try:
                ticker = f"{etf_code}.KS"
                df = yf.download(ticker, start=start_date, end=end_date, progress=False)
                if not df.empty:
                    df.reset_index(inplace=True)
                    df['code'] = etf_code
                    return self._standardize_price_data(df)
            except Exception as e:
                self.logger.warning(f"yfinance 실패 ({etf_code}): {str(e)}")
            
            return pd.DataFrame()
            
        except Exception as e:
            self.logger.error(f"{etf_code} 데이터 수집 오류: {str(e)}")
            return pd.DataFrame()
    
    def _resample_data(self, daily_df: pd.DataFrame, rule: str, timeframe: str) -> pd.DataFrame:
        """일별 데이터를 다른 시간대로 리샘플링"""
        try:
            # 날짜를 인덱스로 설정
            df_copy = daily_df.copy()
            df_copy.set_index('date', inplace=True)
            
            # OHLCV 데이터 리샘플링 규칙
            agg_rules = {
                'open_price': 'first',      # 시가: 첫 번째 값
                'high_price': 'max',        # 고가: 최대값
                'low_price': 'min',         # 저가: 최소값
                'close_price': 'last',      # 종가: 마지막 값
                'volume': 'sum',            # 거래량: 합계
                'code': 'first'             # 코드: 첫 번째 값
            }
            
            # adj_close_price가 있는 경우 추가
            if 'adj_close_price' in df_copy.columns:
                agg_rules['adj_close_price'] = 'last'
            
            # 리샘플링 실행
            resampled = df_copy.resample(rule).agg(agg_rules)
            
            # 결측값 제거
            resampled = resampled.dropna()
            
            # 인덱스를 다시 컬럼으로 변환
            resampled.reset_index(inplace=True)
            
            # 시간대 정보 추가
            resampled['timeframe'] = timeframe
            
            return resampled
            
        except Exception as e:
            self.logger.error(f"리샘플링 오류 ({timeframe}): {str(e)}")
            return pd.DataFrame()
    
    def _standardize_price_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """가격 데이터 표준화"""
        # 컬럼명 통일
        column_mapping = {
            'Date': 'date',
            'Open': 'open_price',
            'High': 'high_price',
            'Low': 'low_price',
            'Close': 'close_price',
            'Adj Close': 'adj_close_price',
            'Volume': 'volume'
        }
        
        df = df.rename(columns=column_mapping)
        
        # 필수 컬럼 확인
        required_columns = ['date', 'close_price']
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"필수 컬럼 누락: {col}")
        
        # 날짜 형식 통일
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
        
        # 결측값 처리
        df = df.dropna(subset=['close_price'])
        
        # 정렬
        df = df.sort_values('date').reset_index(drop=True)
        
        return df
    
    def collect_all_etf_prices_multi_timeframe(self, etf_list: List[str]) -> Dict[str, Dict[str, pd.DataFrame]]:
        """모든 ETF의 다중 시간대 가격 데이터 수집"""
        self.logger.info(f"{len(etf_list)}개 ETF 다중 시간대 데이터 수집 시작...")
        
        all_price_data = {}
        failed_etfs = []
        
        for etf_code in tqdm(etf_list, desc="ETF 다중 시간대 데이터 수집"):
            try:
                multi_timeframe_data = self.get_etf_price_data_multi_timeframe(etf_code)
                
                if multi_timeframe_data:
                    # 최소 데이터 포인트 검증 (일별 데이터 기준)
                    daily_data = multi_timeframe_data.get('daily', pd.DataFrame())
                    if len(daily_data) >= 100:  # 최소 100일 데이터 (약 4개월)
                        all_price_data[etf_code] = multi_timeframe_data
                        
                        # 수집 결과 로깅
                        summary = []
                        for timeframe, df in multi_timeframe_data.items():
                            summary.append(f"{timeframe}: {len(df)}개")
                        self.logger.info(f"{etf_code} 수집 완료 - {', '.join(summary)}")
                    else:
                        failed_etfs.append(etf_code)
                        self.logger.warning(f"{etf_code}: 데이터 부족 (일별 {len(daily_data)}개)")
                else:
                    failed_etfs.append(etf_code)
                    self.logger.warning(f"{etf_code}: 데이터 수집 실패")
                
                # API 제한을 위한 대기
                time.sleep(0.1)
                
            except Exception as e:
                failed_etfs.append(etf_code)
                self.logger.error(f"{etf_code} 수집 실패: {str(e)}")
        
        # 수집 결과 요약
        successful_count = len(all_price_data)
        failed_count = len(failed_etfs)
        
        self.logger.info(f"다중 시간대 데이터 수집 완료: 성공 {successful_count}개, 실패 {failed_count}개")
        
        if failed_etfs:
            self.logger.warning(f"실패한 ETF: {failed_etfs}")
        
        # 전체 데이터 통계
        total_datapoints = 0
        for etf_code, timeframe_data in all_price_data.items():
            for timeframe, df in timeframe_data.items():
                total_datapoints += len(df)
        
        self.logger.info(f"전체 수집된 데이터 포인트: {total_datapoints:,}개")
        
        return all_price_data
    
    def save_multi_timeframe_data(self, all_price_data: Dict[str, Dict[str, pd.DataFrame]], base_path: str):
        """다중 시간대 데이터를 파일로 저장"""
        try:
            import os
            
            # 시간대별 디렉토리 생성
            for timeframe in self.resample_rules.keys():
                timeframe_dir = f"{base_path}/{timeframe}"
                os.makedirs(timeframe_dir, exist_ok=True)
            
            # ETF별, 시간대별 데이터 저장
            for etf_code, timeframe_data in all_price_data.items():
                for timeframe, df in timeframe_data.items():
                    file_path = f"{base_path}/{timeframe}/{etf_code}_{timeframe}_data.csv"
                    df.to_csv(file_path, index=False, encoding='utf-8-sig')
            
            self.logger.info(f"다중 시간대 데이터 저장 완료: {base_path}")
            
        except Exception as e:
            self.logger.error(f"다중 시간대 데이터 저장 오류: {str(e)}")
    
    def get_data_summary(self, all_price_data: Dict[str, Dict[str, pd.DataFrame]]) -> Dict:
        """수집된 데이터 요약 정보 생성"""
        summary = {
            'total_etfs': len(all_price_data),
            'timeframes': {},
            'date_ranges': {},
            'total_datapoints': 0
        }
        
        for timeframe in self.resample_rules.keys():
            timeframe_count = 0
            min_date = None
            max_date = None
            
            for etf_code, timeframe_data in all_price_data.items():
                if timeframe in timeframe_data:
                    df = timeframe_data[timeframe]
                    timeframe_count += len(df)
                    
                    if not df.empty:
                        etf_min_date = df['date'].min()
                        etf_max_date = df['date'].max()
                        
                        if min_date is None or etf_min_date < min_date:
                            min_date = etf_min_date
                        if max_date is None or etf_max_date > max_date:
                            max_date = etf_max_date
            
            summary['timeframes'][timeframe] = timeframe_count
            summary['date_ranges'][timeframe] = {
                'start': str(min_date) if min_date else None,
                'end': str(max_date) if max_date else None
            }
            summary['total_datapoints'] += timeframe_count
        
        return summary

    # 기존 메서드들과의 호환성을 위한 래퍼 메서드
    def collect_all_etf_prices(self, etf_list: List[str]) -> Dict[str, pd.DataFrame]:
        """기존 인터페이스 호환성을 위한 일별 데이터만 반환하는 메서드"""
        multi_timeframe_data = self.collect_all_etf_prices_multi_timeframe(etf_list)
        
        # 일별 데이터만 추출하여 반환
        daily_data = {}
        for etf_code, timeframe_data in multi_timeframe_data.items():
            if 'daily' in timeframe_data:
                daily_data[etf_code] = timeframe_data['daily']
        
        return daily_data
