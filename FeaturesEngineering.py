import pandas as pd
import numpy as np


def max_size(levels):
    return max((q for p, q in levels), default=np.nan)

def streak_counter(arr):
    count = 0
    out = []
    for x in arr:
        if x:
            count += 1
        else:
            count = 0
        out.append(count)
    return out

def size_levels(levels, n):
    return sum(q for (p, q) in levels[:n])
def price_weighted_side(levels, mid, side, n_levels=None, max_dist=None, eps=1e-6):
    """
    Compute price-weighted depth score for one side of the book.

    levels : list of [price, size]
    mid    : midprice (float)
    side   : 'bid' or 'ask'
    n_levels : optionally use only the first N levels
    max_dist : optionally ignore levels farther than this (in price units)
    eps    : small number to avoid division by zero
    """
    if not len(levels) or np.isnan(mid):
        return 0.0

    if n_levels is not None:
        levels = levels[:n_levels]

    score = 0.0

    for p, q in levels:
        p = float(p)
        q = float(q)

        if side == "bid":
            dist = mid - p
        else:  # 'ask'
            dist = p - mid

        # skip if price is on the wrong side or exactly mid
        if dist <= 0:
            continue

        if max_dist is not None and dist > max_dist:
            continue

        score += q / max(dist, eps)

    return score


def price_weighted_imbalance_row(bids, asks, n_levels=5, max_dist=None, eps=1e-8):
    """
    Compute price-weighted imbalance for a single book snapshot.

    Returns a float in [-1, 1].
    """
    if not len(bids) or not len(asks):
        return np.nan

    best_bid = float(bids[0][0])
    best_ask = float(asks[0][0])

    if best_ask <= best_bid:
        # crossed or invalid book
        return np.nan

    mid = 0.5 * (best_bid + best_ask)

    bid_score = price_weighted_side(bids, mid, side="bid",
                                    n_levels=n_levels, max_dist=max_dist, eps=eps)
    ask_score = price_weighted_side(asks, mid, side="ask",
                                    n_levels=n_levels, max_dist=max_dist, eps=eps)

    denom = bid_score + ask_score
    if denom <= 0:
        return 0.0

    return (bid_score - ask_score) / denom

def compute_momentum_features(df):
    """
    Compute multi-horizon momentum and volatility features.
    Uses nearest timestamp match instead of shift(freq=...).
    Assumes df.index is a DatetimeIndex.
    """

    df = df.sort_index()

    horizons = [10, 30, 60, 300]  # seconds

    for sec in horizons:
        # target timestamp = current index - sec
        target_times = df.index - pd.Timedelta(f"{sec}s")
        df[f"_target_{sec}"] = target_times
        tolerance = str(int(sec//2))
        # Perform nearest timestamp lookup (backward or nearest)
        merged = pd.merge_asof(
            df.sort_values(f"_target_{sec}"),
            df[["vwap"]].dropna().rename(columns={"vwap": f"vwap_lag_{sec}"}),
            left_on=f"_target_{sec}",
            right_index=True,
            direction="backward",           # or "backward" if you prefer
            tolerance=pd.Timedelta(tolerance+"s"),  # adjust based on your data spacing
        ).sort_index()

        # Compute return
        lag = merged[f"vwap_lag_{sec}"]
        curr = merged["vwap"]
        df[f"return_{sec}s"] = (curr - lag) / lag

        # Clean temporary columns
        df = df.drop(columns=[f"_target_{sec}", f"vwap_lag_{sec}"], errors="ignore")

    # Weighted momentum (recent more important)
    df["weighted_momentum"] = (
        0.5 * df["return_10s"] +
        0.3 * df["return_30s"] +
        0.2 * df["return_60s"]
    )

    # Rolling 5-minute price position
    roll_min = df["vwap"].rolling("5min").min()
    roll_max = df["vwap"].rolling("5min").max()

    df["price_position_5m"] = (df["vwap"] - roll_min) / (roll_max - roll_min)
    df["price_position_5m"].replace([np.inf, -np.inf], np.nan, inplace=True)

    # Short-term volatility (use 10-second returns)
    df["volatility_30s"] = df["return_10s"].rolling("30s").std()
    df["volatility_1m"]  = df["return_10s"].rolling("1min").std()
    df["volatility_5m"]  = df["return_10s"].rolling("5min").std()

    return df

def compute_trade_flow_features(df):
    """
    Build momentum and flow features from buyer_ratio and traded_notional.
    Assumes DatetimeIndex for time-based rolling windows.
    """

    # base signals
    df["buyer_ratio"] = df["buyer_strength"].clip(0, 1)  # just in case
    df["buyer_imbalance"] = df["buyer_ratio"] * 2 - 1      # [0,1] -> [-1,1]

    for window in ["10s", "30s", "60s", "5min"]:
        # total notional in window
        vol = df["notional_traded"].rolling(window).sum()
        df[f"volume_{window}"] = vol

        # simple (unweighted) rolling mean of buyer_ratio
        df[f"buyer_ratio_{window}"] = df["buyer_ratio"].rolling(window).mean()

        # volume-weighted buyer ratio over window
        buy_vol = (df["buyer_ratio"] * df["notional_traded"]).rolling(window).sum()
        df[f"vw_buyer_ratio_{window}"] = buy_vol / vol.replace(0, np.nan)

    # flow acceleration type features
    df["flow_acceleration_30"] = df["buyer_ratio_30s"] - df["buyer_ratio_60s"]
    df["flow_acceleration_10"]   = df["buyer_ratio_10s"] - df["buyer_ratio_60s"]

    return df

if __name__ == "__main__":
    coin='doge'
    path='data/' + coin +'/'
    print('reading in parquet file')
    df=pd.read_parquet(path+coin+'_data_all.parquet')
    split_date = pd.Timestamp("2025-11-07")
    #print("splitting data into training and testing")
    #df = df_all[df_all.index < split_date]
    #df_test = df_all[df_all.index >= split_date]
    #del df_all
    #df.set_index('timestamp_ms', inplace=True)
    print('creating order book features')
    df["bid_price_1"] = df["bids"].apply(lambda x: x[0][0] if len(x) else np.nan)
    df["bid_size_1"] = df["bids"].apply(lambda x: x[0][1] if len(x) else np.nan)
    df["ask_price_1"] = df["asks"].apply(lambda x: x[0][0] if len(x) else np.nan)
    df["ask_size_1"] = df["asks"].apply(lambda x: x[0][1] if len(x) else np.nan)

    df["bid_max_size"] = df["bids"].apply(max_size)
    df["ask_max_size"] = df["asks"].apply(max_size)

    window = '5min'

    print('computing z scores')

    df["bid_size_mean"] = df["bid_max_size"].rolling(window).mean()
    df["bid_size_std"] = df["bid_max_size"].rolling(window).std()

    df["ask_size_mean"] = df["ask_max_size"].rolling(window).mean()
    df["ask_size_std"] = df["ask_max_size"].rolling(window).std()

    df["bid_z"] = (df["bid_max_size"] - df["bid_size_mean"]) / df["bid_size_std"]
    df["ask_z"] = (df["ask_max_size"] - df["ask_size_mean"]) / df["ask_size_std"]

    df["bid_large_z"] = df["bid_z"].where(df["bid_z"] > 2, 0)
    df["ask_large_z"] = df["ask_z"].where(df["ask_z"] > 2, 0)

    df["bid_is_large"] = (df["bid_z"] > 2).astype(int)
    df["ask_is_large"] = (df["ask_z"] > 2).astype(int)
    df["bid_large_streak"] = streak_counter(df["bid_is_large"].values)
    df["ask_large_streak"] = streak_counter(df["ask_is_large"].values)
    print('computing price imbalance')
    df["price_imbalance"] = df.apply(
        lambda row: price_weighted_imbalance_row(
            row["bids"], row["asks"],
            n_levels=None,     # or None for all levels
            max_dist=None,  # or something like max_dist=50 for within $50 of mid
            eps=1e-6
        ),axis=1)
    df[['price_imbalance',"bid_large_streak", "ask_large_streak","bid_large_z","ask_large_z"]].describe()
    #Create features from buy/sell indicator
    df=compute_trade_flow_features(df)
    #Create momentum features
    df=compute_momentum_features(df)
    df[['weighted_momentum', 'volatility_1m', 'volatility_5m']].describe()
    print("writing to parquet")
    df.to_parquet(path+coin+'_with_features.parquet', compression="snappy")
