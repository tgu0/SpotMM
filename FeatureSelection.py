import pandas as pd
import numpy as np
import pdfkit
import datetime as dt
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
import quantstats as qs
from math import sqrt
qs.extend_pandas()

def enforce_single_position(
    df,
    signal_col="signal",
    ttt_col="time_to_target",
    vwap_col="vwap",
    bid_col="bid_price_1",
    ask_col="ask_price_1",
    future_vwap_col="forward_vwap",
    out_signal_col="signal_single_pos",
    ret_col="strategy_ret_gross",
    use_mid=False
):
    """
    Enforce that only one position can be open at a time AND compute per-trade returns.

    Logic:
      - At time t, if signal != 0 and we are not already in a trade:
          * If signal < 0 (sell): entry = bid_price_1
              return = (bid_price_1 - future_vwap) / vwap
          * If signal > 0 (buy): entry = ask_price_1
              return = (future_vwap - ask_price_1) / vwap
          * This return is recorded ONLY on the entry row in `ret_col`.
          * If use_mid is turned on the algorithm assumes fills at the
          mid price instead of the bid or ask price
      - After opening a trade at time t with time_to_target = T (seconds),
        no new trade can be opened until t + T.
      - All non-entry rows get 0.0 in `ret_col`.

    Assumes:
      - df has a DatetimeIndex
      - `ttt_col` is time-to-target in seconds (numeric)
      - vwap, bid, ask, and future_vwap columns exist and are numeric
    """

    df = df.sort_index().copy()

    sig = df[signal_col].to_numpy()
    ttt = df[ttt_col].to_numpy()
    times = df.index.to_numpy()

    vwap = df[vwap_col].to_numpy()
    bid  = df[bid_col].to_numpy()
    ask  = df[ask_col].to_numpy()
    fvw  = df[future_vwap_col].to_numpy()

    out_signal = np.zeros_like(sig, dtype=float)
    strat_ret  = np.zeros_like(sig, dtype=float)

    next_free_time = None  # timestamp when we're allowed to open next trade

    for i in range(len(sig)):
        ts = times[i]

        # If we're still in a trade window, suppress any new trades
        if next_free_time is not None and ts < next_free_time:
            continue

        s = sig[i]

        # Only consider opening a trade if we have a non-zero signal
        if s == 0 or np.isnan(vwap[i]) or np.isnan(fvw[i]) or vwap[i] == 0 or np.isnan(ttt[i]):
            continue

        # Compute return based on direction and execution prices
        if s < 0:  # SELL / short
            if np.isnan(bid[i]):
                continue
            entry_price = bid[i]
            if use_mid:
                entry_price=vwap[i]
            trade_ret = (entry_price - fvw[i]) / vwap[i]
        elif s > 0:  # BUY / long
            if np.isnan(ask[i]):
                continue
            entry_price = ask[i]
            if use_mid:
                entry_price=vwap[i]
            trade_ret = (fvw[i] - entry_price) / vwap[i]
        else:
            continue  # shouldn't happen, but just in case

        # Record trade
        out_signal[i] = s
        strat_ret[i] = trade_ret

        # Set next time we are allowed to open another trade
        hold_seconds = int(ttt[i])
        next_free_time = ts + np.timedelta64(hold_seconds, "s")

    df[out_signal_col] = out_signal
    df[ret_col] = strat_ret

    return df


if __name__ == "__main__":
    coin='btc'
    path='data/' + coin+'/'
    print('reading in parquet file')
    df_all=pd.read_parquet(path+coin+'_with_features.parquet')
    split_date = pd.Timestamp("2025-11-07")
    print("splitting data into training and testing")
    df_all['future_returns']= (df_all['forward_vwap']-df_all['vwap'])/df_all['vwap']
    df = df_all[df_all.index < split_date]
    df_test = df_all[df_all.index >= split_date]
    ntrain = len(df)
    ntest = len(df_test)
    df.dropna(inplace=True)
    df_test.dropna(inplace=True)
    print("dropped " +str(ntrain-len(df)) + " rows for training set")
    print("dropped " + str(ntest - len(df_test)) + " rows for testing set")
    del df_all
    #df.dropna(inplace=True)
    features = ['notional_traded', 'content_q', 'content_b', 'buyer_strength', 'vwap', 'bid_price_1', 'bid_size_1',
                'ask_price_1', 'ask_size_1', 'bid_max_size', 'ask_max_size', 'bid_large_z', 'ask_large_z',
                'bid_large_streak', 'ask_large_streak', 'price_imbalance', 'buyer_ratio', 'buyer_imbalance',
                'volume_10s', 'vw_buyer_ratio_10s', 'volume_30s', 'vw_buyer_ratio_30s', 'volume_60s',
                'vw_buyer_ratio_60s', 'volume_5min', 'buyer_ratio_5min', 'vw_buyer_ratio_5min',
                'flow_acceleration_30', 'flow_acceleration_10', 'return_10s', 'return_30s', 'return_60s',
                'return_300s', 'weighted_momentum', 'price_position_5m', 'volatility_30s', 'volatility_1m', 'volatility_5m']
    X_train = df[features]
    Y_train = df['future_returns']
    model = make_pipeline(
        StandardScaler(),
        Ridge(alpha=0.1)
    )

    model.fit(X_train, Y_train)
    X_test = df_test[features]
    y_test = df_test["future_returns"]
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    print("Train R²:", r2_score(Y_train, y_pred_train))
    print("Test  R²:", r2_score(y_test, y_pred_test))

    print("Train RMSE:", np.sqrt(mean_squared_error(Y_train, y_pred_train)))
    print("Test  RMSE:", np.sqrt(mean_squared_error(y_test, y_pred_test)))

    scaler = model.named_steps["standardscaler"]
    ridge = model.named_steps["ridge"]

    coefs = ridge.coef_/ scaler.scale_
    feature_importance = pd.Series(coefs, index=features).sort_values(key=abs, ascending=False)
    print(feature_importance)
    features=list(feature_importance.index[:15])
    pred_sign = np.sign(y_pred_test)
    true_sign = np.sign(y_test)

    print("Directional accuracy:", (pred_sign == true_sign).mean())

    #Test a simple back of the envelope trading strategy
    sell_thresh = pd.Series(y_pred_train).quantile(0.05)
    buy_thresh = pd.Series(y_pred_train).quantile(0.95)
    df_test['y_pred']=y_pred_test
    df_test["signal"] = np.where(
        df_test["y_pred"] > buy_thresh, 1,
        np.where(df_test["y_pred"] < sell_thresh, -1, 0)
    )
    df_test=enforce_single_position(df_test, use_mid=True)
    #df_test["strategy_ret_gross"] = df_test["signal_single_pos"] * df_test["future_returns"]
    #generate_report(df_test["strategy_ret_gross"], 'btc-mm')
    roi_series = df_test.loc[df_test["strategy_ret_gross"]!=0, "strategy_ret_gross"].copy()

    # remove NaNs
    roi_series = roi_series.dropna()

    # ensure it's datetime-indexed
    roi_series.index = pd.to_datetime(roi_series.index)

    # now give returns to quantstats
    qs.reports.html(roi_series, output=coin+"_marketmaking.html")
    '''
    model = LinearRegression()
    model.fit(X_train[["vwap"]], Y_train)
    y_pred_train = model.predict(X_train[["vwap"]])
    y_pred_test = model.predict(X_test[["vwap"]])

    print("Train R²:", r2_score(Y_train, y_pred_train))
    print("Test  R²:", r2_score(y_test, y_pred_test))
    '''