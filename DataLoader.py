import pandas as pd
import numpy as np
import ijson
import time
from statsmodels.tsa.stattools import adfuller

def stream_filtered_books(json_path, chunk_size=50_000, min_size=0.01):
    """
    Stream a large Binance-style depth file and yield chunks as DataFrames.

    Fields returned:
      - h_it               : header internal time
      - event_time         : pandas Timestamp, UTC, floored to ms
      - asks               : filtered list of [price, size]
      - bids               : filtered list of [price, size]

    Filtering rules:
      1) Top-of-book = first level with quantity > 0 to 8 decimal places.
      2) Always keep TOB regardless of quantity.
      3) For subsequent levels, keep only levels with size >= min_size.
    """

    def ms_to_timestamp(ms):
        """Convert Binance milliseconds → pandas Timestamp floored to ms."""
        # Drop any floating-point .0 safely
        if ms is None:
            return None
        ms_int = int(ms)
        return pd.to_datetime(ms_int, unit="ms")

    def process_side(raw_levels):
        """Process asks or bids according to TOB + threshold rules."""
        if not raw_levels:
            return []

        # Convert to floats early
        parsed = [[float(p), float(q)] for p, q in raw_levels]

        # --- Find TOP OF BOOK ---
        tob_index = None
        for i, (p, q) in enumerate(parsed):
            # round(q,8) > 0 → true non-zero quote
            if round(q, 8) > 0:
                tob_index = i
                break

        if tob_index is None:
            return []  # all zero quantities

        filtered = []

        # 1) Insert TOB always
        filtered.append(parsed[tob_index])

        # 2) Add later levels only if size >= min_size
        for j in range(tob_index + 1, len(parsed)):
            p, q = parsed[j]
            if q >= min_size:
                filtered.append([p, q])

        return filtered

    # --- STREAM PARSE ---
    with open(json_path, "rb") as f:
        batch = []

        for obj in ijson.items(f, "item"):

            h = obj.get("h", {})
            c = obj.get("c", {})
            data = c.get("data", {})

            h_it = h.get("it")
            event_raw = data.get("E")

            # convert to Timestamp
            event_time = ms_to_timestamp(event_raw)

            raw_asks = data.get("a", []) or []
            raw_bids = data.get("b", []) or []

            asks = process_side(raw_asks)
            bids = process_side(raw_bids)

            batch.append(
                {
                    "h_it": h_it,
                    "event_time": event_time,
                    "asks": asks,
                    "bids": bids,
                }
            )

            if len(batch) >= chunk_size:
                yield pd.DataFrame(batch)
                batch = []

        # flush last chunk
        if batch:
            yield pd.DataFrame(batch)

def get_closest_row(df, ts):
    idx = df.index.searchsorted(ts)

    # candidates: the one before and after
    candidates = []
    if idx > 0:
        candidates.append(df.index[idx - 1])
    if idx < len(df.index):
        candidates.append(df.index[min(idx, len(df.index)-1)])

    # pick the one with smallest absolute diff
    closest = min(candidates, key=lambda d: abs(d - ts))
    return df.loc[closest]


def compute_forward_metrics(df, target_notional=10000, price_col='vwap', notional_col='notional_traded'):
    """
    Compute time until target notional is traded and VWAP over that period

    Args:
        df: DataFrame with DatetimeIndex and columns 'price' and 'traded_notional'
        target_notional: Target cumulative notional to reach
        price_col: Name of price column
        notional_col: Name of notional column
    """
    n = len(df)
    time_to_target = np.full(n, np.nan)
    forward_vwap = np.full(n, np.nan)

    prices = df[price_col].to_numpy()
    notionals = df[notional_col].to_numpy()
    timestamps = df.index.to_numpy()

    # Initialize window
    left = 0
    right = 0
    cumsum_notional = 0
    cumsum_price_notional = 0

    for left in range(n):
        # If we moved left pointer, subtract the previous left value
        if left > 0:
            cumsum_notional -= notionals[left - 1]
            cumsum_price_notional -= prices[left - 1] * notionals[left - 1]

        # Expand right pointer until we hit target
        while right < n and cumsum_notional < target_notional:
            cumsum_notional += notionals[right]
            cumsum_price_notional += prices[right] * notionals[right]
            right += 1

        # If we hit the target, record metrics
        if cumsum_notional >= target_notional:
            time_diff = (timestamps[right - 1] - timestamps[left]) / pd.Timedelta(seconds=1)
            time_to_target[left] = time_diff
            forward_vwap[left] = cumsum_price_notional / cumsum_notional

    df['time_to_target'] = time_to_target
    df['forward_vwap'] = forward_vwap

    return df


def analyze_time_to_notional(df_orig, notionals=[100000, 250000, 500000,750000, 1000000, 2000000]):
    results = {}

    for notional in notionals:
        # Calculate time to reach this notional
        df = compute_forward_metrics(df_orig, notional)  # Your existing function
        time_to_reach = df['time_to_target']
        results[notional] = {
            'mean_time': time_to_reach.mean(),
            'median_time': time_to_reach.median(),
            'p25_time': time_to_reach.quantile(0.25),
            'p75_time': time_to_reach.quantile(0.75),
            'p95_time': time_to_reach.quantile(0.95),
            'pct_reached_in_60s': (time_to_reach <= 60).mean() * 100,
            'pct_reached_in_180s': (time_to_reach <= 180).mean() * 100,
            'pct_reached_in_5m': (time_to_reach <= 300).mean() * 100,
        }

    results_df = pd.DataFrame(results).T
    print(results_df)
    return results_df

if __name__ == "__main__":
    #df_bbo = pd.read_parquet('data_small/btc/bbo_btc.parquet')
    #records = ijson.items(open("data/btc/book_update_btc.json"), "item")
    #obdf = pd.DataFrame(records)
    coin='doge'
    path='data/' +coin+'/'
    print("reading in trades data")
    df_trades = pd.read_parquet(path+'trades_'+coin+'.parquet')
    #print("aggregating trades to millisecond granularity")
    df_trades['direction']=2*df_trades["content_b"]-1
    df_trades['notional_traded'] = df_trades['content_p'] * df_trades['content_q']
    df_trades['timestamp'] = pd.to_datetime(df_trades['content_t'], unit='us')
    # Round timestamp to millisecond
    df_trades['timestamp_ms'] = df_trades['timestamp'].dt.floor('1ms')
    print("aggregating trades data")
    # Aggregate
    df_agg = df_trades.groupby('timestamp_ms').agg({
        'notional_traded': 'sum',
        'content_q': 'sum',
        'content_b':'mean'
    }).reset_index()

    df_agg["buyer_strength"] = df_agg["content_b"]
    # Calculate VWAP
    df_agg['vwap'] = df_agg['notional_traded'] / df_agg['content_q']
    del df_trades
    print("reading in order book information")
    ob_path = path+"book_update_"+coin+".json"

    all_chunks = []
    for chunk in stream_filtered_books(ob_path):
        all_chunks.append(chunk)

    df = pd.concat(all_chunks, ignore_index=True)
    df_agg = df_agg.sort_values('timestamp_ms')
    df = df.sort_values('event_time')
    df = df.dropna(subset=['event_time'])
    print("merging trades with orderbook information")
    df_merged = pd.merge_asof(
        df_agg,
        df[['event_time', 'bids', 'asks']],  # select only columns you need
        left_on='timestamp_ms',
        right_on='event_time',
        direction='backward'  # gets the most recent orderbook that's <= trade time
    )
    print("deleting unneeded dfs")
    del df
    del df_agg
    del all_chunks
    df_merged.set_index('timestamp_ms', inplace=True)
    df_merged.sort_index(inplace=True)
    print("computing forward vwaps")
    #df_final = compute_forward_metrics(df_merged, target_notional=5000000, price_col='vwap')
    #df_final['vwap_60s'] = df_final['vwap'].rolling('60s').apply(
        #lambda x: (x * df_final.loc[x.index, 'notional_traded']).sum() /
                  #df_final.loc[x.index, 'notional_traded'].sum()
    #)
    #del df_merged
    print("performing time analysis on notional volumes")
    analysis_df=analyze_time_to_notional(df_merged)
    analysis_df.to_csv(coin+'_time_analysis.csv')
    df_final = compute_forward_metrics(df_merged, target_notional=250000, price_col='vwap')
    df_final.dropna(subset=['vwap', 'bids', 'asks'], inplace=True)
    df_final.to_parquet("data/"+coin+"/"+coin+"_data_all.parquet", compression="snappy")
    df_sample= df_final[:100000]
    print("ADF of forward VWAP prices:")
    print(adfuller(df_sample['forward_vwap'],maxlag=50, autolag=None, regression='c' ))
    df_sample['future_returns']= (df_sample['forward_vwap']-df_sample['vwap'])/df_sample['vwap']
    print("ADF of forward VWAP returns:")
    print(adfuller(df_sample['future_returns'],maxlag=50, autolag=None, regression='c'))