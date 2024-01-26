import pandas as pd
import numpy as np
from numba import njit, prange
from itertools import combinations


from typing import List

def book_depth(lob, size, side, lvls_cnt: int = 10, fillna = True):
    side = side.lower()
    assert side in ('ask', 'bid'), f"unknown side: {side}"

    vol = 0
    fst_mask = pd.Series(np.zeros(lob.shape[0]))
    fst_mask.columns = [f'{side}s[0].amount']
    fst_mask.index = lob.index

    res = pd.Series(np.zeros(lob.shape[0]))
    res.index = lob.index

    for i in range(lvls_cnt):
        vol += (vol < size) * lob[f'{side}s[{i}].amount']
        fst_mask += (vol >= size)
        res[fst_mask == 1] = lob[f'{side}s[{i}].price']
        
    if fillna:
        res[res == 0] = lob[f'{side}s[{lvls_cnt-1}].price']
        
    return res


def vimba_up_level(book_df, up_level):
    vimba_up_level = pd.DataFrame()
    ask = sum([book_df[f'asks[{lvl}].amount'] for lvl in range(up_level)])
    vimba_up_level[f'vimba_up_{up_level}'] = ask / (sum([book_df[f'bids[{lvl}].amount'] for lvl in range(up_level)]) + ask)
    return vimba_up_level


def vimba_level(df, levels):
    vimba_at_level = pd.DataFrame()
    for lvl in levels:
        vimba_at_level[f'vimba_at_{lvl}'] = (df[f'asks[{lvl}].amount']) / (df[f'asks[{lvl}].amount'] + df[f'bids[{lvl}].amount'])
    return vimba_at_level


def zscore_calc(df, features, lookbacks: List[int]):
    df_zscore = pd.DataFrame()
    for feature in features:
        for lb in lookbacks:
            df_zscore[f'zscore_{feature}_{lb}'] = (df[feature] - df[feature].rolling(min_periods=lb // 2,
                                                                                     window=lb).mean()) / (
                                                          df[feature].rolling(min_periods=lb // 2,
                                                                              window=lb).std() + 1)
    return df_zscore

def vwap_all_by_size(df, up_size):
    vwap_side = pd.DataFrame()
    sum_vol = defaultdict(int)
    vwap = 0

    for side in ('ask', 'bid'):
        for lvl in range(num_of_levels):
            cur_vol_mult = np.maximum(np.minimum(up_size - sum_vol[side], df[f'{side}s[{lvl}].amount']), 0)
            sum_vol[side] += cur_vol_mult
            vwap += cur_vol_mult * df[f'{side}s[{lvl}].price']

    vwap_side[f'vwap_all_qty_{str(up_size).replace(".", "p")}'] = vwap / (sum_vol['ask'] + sum_vol['bid'])

    return vwap_side

def midprice(df):
    mid = pd.DataFrame()
    mid['mid_price'] = (df['bids[0].price'] + df['asks[0].price']) / 2
    return mid

def midprice_movement(df):
    mid_move = pd.DataFrame()
    mid_move['mid_price_movement'] = (df['mid_price'].diff(periods=5).apply(
            lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
        )
    return mid_move

def differences(df, cols, windows):
    feat = pd.DataFrame()
    for col in cols:
        for window in windows:
            feat[f"{col}_diff_{window}"] = df[col].diff(window)
    return feat

def volume(df):
    vol = pd.DataFrame()
    vol['volume'] = df['asks[0].amount'] + df['bids[0].amount']
    return vol

def liquidity_imbalance(df):
    liq_imb = pd.DataFrame()
    liq_imb['liquidity_imbalance'] = ((df['bids[0].amount'] - df['asks[0].amount']) / 
                                      (df['asks[0].amount'] + df['bids[0].amount']))
    return liq_imb

def price_imbalance(df):
    price_imb = pd.DataFrame()
    price_imb['price_imbalance'] = ((df['bids[0].price'] - df['asks[0].price']) / 
                                      (df['asks[0].price'] + df['bids[0].price']))
    return price_imb

def size_imbalance(df):
    size_imb = pd.DataFrame()
    size_imb['size_imbalance'] = df['bids[0].amount'] / df['asks[0].amount'] 
    return size_imb

def price_spread(df):
    spread = pd.DataFrame()
    spread['price_spread'] = df['asks[0].price'] - df['bids[0].price']
    return spread

def bid_spread(df):
    b_spread = pd.DataFrame()
    b_spread['bid_spread'] = df['bids[0].price'] - df['bids[1].price']
    return b_spread 

def ask_spread(df):
    a_spread = pd.DataFrame()
    a_spread['ask_spread'] = df['asks[0].price'] - df['asks[1].price']
    return a_spread 

def spread_intensity(spread):
    spread_intens = pd.DataFrame()
    spread_intens['spread_intensity'] = spread.diff()
    return spread_intens

def spread_depth_ratio(df):
    spread_dep = pd.DataFrame()
    spread_dep['spread_depth_ratio'] = ((df['asks[0].price'] - df['bids[0].price']) / 
                                      (df['asks[0].amount'] + df['bids[0].amount']))
    return spread_dep

def wap(df, level):
    wap_ = pd.DataFrame()
    wap_[f'wap_at_{level}'] = ((df[f'bids[{level}].price'] * df[f'asks[{level}].amount'] + 
                        df[f'asks[{level}].price'] * df[f'bids[{level}].amount']) / 
                            (df[f'asks[{level}].amount'] + df[f'bids[{level}].amount']))
    return wap_

def wap_balance(wap_1, wap_2, levels):
    wap_bal = pd.DataFrame()
    wap_bal[f'wap_balance_{levels[0]}{levels[1]}'] = np.abs(wap_1[f'wap_at_{levels[0]}'] - wap_2[f'wap_at_{levels[1]}'])
    return wap_bal

def market_urgency(spread, liq_imb):
    market_urg = pd.DataFrame()
    market_urg['market_urgency'] = (spread['price_spread'] * liq_imb['liquidity_imbalance'])
    return market_urg

def relative_spread(spread, wap_):
    relative_sp = pd.DataFrame()
    relative_sp['relative_spread'] = spread['price_spread'] / wap_['wap_at_0']
    return relative_sp

@njit(parallel=True)
def compute_triplet_imbalance(df_values, comb_indices):
    num_rows = df_values.shape[0]
    num_combinations = len(comb_indices)
    imbalance_features = np.empty((num_rows, num_combinations))

    for i in prange(num_combinations):
        a, b, c = comb_indices[i]
        for j in range(num_rows):
            max_val = max(df_values[j, a], df_values[j, b], df_values[j, c])
            min_val = min(df_values[j, a], df_values[j, b], df_values[j, c])
            mid_val = df_values[j, a] + df_values[j, b] + df_values[j, c] - min_val - max_val
            if mid_val == min_val:  # Prevent division by zero
                imbalance_features[j, i] = np.nan
            else:
                imbalance_features[j, i] = (max_val - mid_val) / (mid_val - min_val)

    return imbalance_features

def calculate_triplet_imbalance_numba(price, df):
    # Convert DataFrame to numpy array for Numba compatibility
    df_values = df[price].values
    comb_indices = [(price.index(a), price.index(b), price.index(c)) for a, b, c in combinations(price, 3)]

    # Calculate the triplet imbalance
    features_array = compute_triplet_imbalance(df_values, comb_indices)

    # Create a DataFrame from the results
    columns = [f"{a}_{b}_{c}_imb2" for a, b, c in combinations(price, 3)]
    features = pd.DataFrame(features_array, columns=columns, index=df.index)

    return features
