import pandas as pd
import numpy as np
from itertools import combinations

from catboost import CatBoostRegressor
from datetime import timedelta

from typing import Union
from typing import List

from ml_utils import *

class Predictor:
    def __init__(self, full_model_path: Union[str, List[str]]):
        self.model  = CatBoostRegressor()
        self.model.load_model(full_model_path, format='cbm')

    @staticmethod
    def model_name() -> Union[str, List[str]]:
        return '20240110-203409_model10.cbm'

    def predict(self, features: pd.DataFrame) -> pd.Series:
        """
        Method is called once every time new submission received
            Params:
                Your features returned from `calc_features` method

            Returns: pd.Series[float]
                Array of predicted returns (price_{n + 1} / price_{n} - 1).
                One value must be generated for every bbo dataframe timestamp 
                so that len(Series) == len(bbos)
        """

        predict = pd.Series(self.model.predict(features))

        return predict

    
    def calc_features(self,
                      bbos: pd.DataFrame,
                      lobs: pd.DataFrame,
                      trades: pd.DataFrame
                      ) -> pd.DataFrame:
        """
            Params
            ----------
            bbos: pd.DataFrame
                Index - timestamp - int
                columns - ["bid_price", "bid_amount", "ask_price", "ask_amount", "midprice"]
                dtypes -  float

            lobs: pd.DataFrame
                Index - timestamp - int
                columns - ["asks[0].price", "asks[0].amount", "bids[0].price", "bids[0].amount", ..., "asks[10].price", "asks[10].amount", "bids[10].price", "bids[10].amount"]
                dtypes -  float
                level price or vol can be empty

            trades: pd.DataFrame
                Index - timestamp - int
                columns - ["side", "price", "size"]
                dtypes -  [int   , float  , float ]
                side: 1 for bid and 0 for ask

            Returns:
                Pandas DataFrame of all your features of format `bbos timestamps` x `your features`
                so that len(DataFrame) == len(bbos)
        """

        vimba_up_3_level_df = vimba_up_level(lobs, up_level=3)
        vimba_up_5_level_df = vimba_up_level(lobs, up_level=5)
        vimba_up_10_level_df = vimba_up_level(lobs, up_level=9)

        vimba_at_levels_df = vimba_level(lobs, levels=[0, 1, 2, 3])

        bd_diff_np = book_depth(lobs, size=15, side='ask').values - book_depth(lobs, size=15, side='bid').values
        bd_diff_df = pd.DataFrame({'book_depth_diff_15_btc': bd_diff_np}, index=lobs.index)
        
        mid_price = midprice(lobs)
        vol = volume(lobs)
        liqd_imb = liquidity_imbalance(lobs)
        price_imb = price_imbalance(lobs)
        size_imb = size_imbalance(lobs)
        spread = price_spread(lobs)
        b_spread = bid_spread(lobs)
        a_spread = ask_spread(lobs)
        spread_int = spread_intensity(spread)
        spread_dep = spread_depth_ratio(lobs)
        
        wap_at_0 = wap(lobs, 0)
        wap_at_1 = wap(lobs, 1)
        wap_at_2 = wap(lobs, 2)
        wap_at_3 = wap(lobs, 3)
        #wap_at_4 = wap(lobs, 4)
        wap_at_5 = wap(lobs, 5)
        wap_bal_10 = wap_balance(wap_at_0, wap_at_1, [0, 1])
        wap_bal_20 = wap_balance(wap_at_0, wap_at_2, [0, 2])
        wap_bal_30 = wap_balance(wap_at_0, wap_at_3, [0, 3])
        wap_bal_50 = wap_balance(wap_at_0, wap_at_5, [0, 5])
        wap_bal_21 = wap_balance(wap_at_1, wap_at_2, [1, 2])
        
        market_urg = market_urgency(spread, liqd_imb)
        relative_sp = relative_spread(spread, wap_at_0)
        
        lobs['mid_price'] = mid_price['mid_price']
        mid_price_move = midprice_movement(lobs)
        
        lobs['wap_at_0'] = wap_at_0['wap_at_0']
        lobs['wap_at_1'] = wap_at_1['wap_at_1']
        lobs['wap_at_2'] = wap_at_2['wap_at_2']
        lobs['wap_at_3'] = wap_at_3['wap_at_3']
        
        prices = ["asks[0].price", "bids[0].price", "asks[1].price", "bids[1].price", 
                  "wap_at_0", "wap_at_1", "wap_at_2", "wap_at_3"]
        sizes = ["bids[0].amount", "asks[0].amount", "bids[1].amount", "asks[1].amount"]
        
        triplet_imb_price = calculate_triplet_imbalance_numba(prices, lobs)
        triplet_imb_size = calculate_triplet_imbalance_numba(sizes, lobs)
        
        cols = ["asks[0].price", "bids[0].price", "bids[0].amount", "asks[0].amount"]
        diff_features = differences(lobs, cols, [1, 2, 3, 5, 10])
        
        features = pd.concat([
            vimba_up_3_level_df,
            vimba_up_5_level_df,
            vimba_up_10_level_df,
            vimba_at_levels_df,
            bd_diff_df,
            mid_price,
            vol,
            liqd_imb,
            price_imb,
            size_imb,
            spread,
            b_spread,
            a_spread,
            spread_int,
            spread_dep,
            wap_at_0,
            wap_at_1,
            wap_at_2,
            wap_at_3,
            #wap_at_4,
            wap_at_5,
            wap_bal_10,
            wap_bal_20,
            wap_bal_30,
            wap_bal_50,
            wap_bal_21,
            market_urg,
            relative_sp,
            mid_price_move,
            triplet_imb_price,
            triplet_imb_size,
            diff_features
        ], axis=1).asof(bbos.index)
        return features
