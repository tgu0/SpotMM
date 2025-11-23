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

def enforce_single_position(df, signal_col="signal", ttt_col="time_to_target",
                            out_col="signal_single_pos"):
    """
    Enforce that only one position can be open at a time.
    If a trade is opened at time t with time_to_target = T (seconds),
    no new trade can be opened until t + T.

    df must have:
      - DatetimeIndex
      - signal_col: raw trading signal (+1, -1, 0)
      - ttt_col: time-to-target in seconds (numeric or float)
    """
    df = df.sort_index()

    sig = df[signal_col].to_numpy()
    ttt = df[ttt_col].to_numpy()
    times = df.index.to_numpy()

    out = np.zeros_like(sig)
    next_free_time = None  # timestamp when we are allowed to open next trade

    for i in range(len(sig)):
        ts = times[i]

        # If we're still in a trade window, we must stay flat
        if next_free_time is not None and ts < next_free_time:
            # suppress any new signal
            continue

        # We're allowed to take a new trade
        if sig[i] != 0:
            out[i] = sig[i]

            # If time_to_target is NaN, treat as no enforced hold
            if not np.isnan(ttt[i]):
                # assume time_to_target is in seconds
                hold_delta = np.timedelta64(int(ttt[i]), "s")
                next_free_time = ts + hold_delta
            else:
                next_free_time = None

        # If sig[i] == 0 and we're past next_free_time, we just remain flat

    df[out_col] = out
    return df


if __name__ == "__main__":
    path='data/btc/'
    print('reading in parquet file')
    df_all=pd.read_parquet(path+'btc_with_features.parquet')
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
    features2=['price_imbalance', 'notional_traded', 'content_q', 'volume_60s', 'volume_30s', 'vw_buyer_ratio_10s',
               'vwap', 'return_300s', 'vw_buyer_ratio_60s', 'volatility_30s', 'volatility_5m', 'volume_5min',
               'vw_buyer_ratio_30s', 'content_b', 'buyer_strength']
    X_train = df[features2]
    Y_train = df['future_returns']
    model = make_pipeline(
        StandardScaler(),
        Ridge(alpha=0.1)
    )

    model.fit(X_train, Y_train)
    X_test = df_test[features2]
    y_test = df_test["future_returns"]
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    print("Train R²:", r2_score(Y_train, y_pred_train))
    print("Test  R²:", r2_score(y_test, y_pred_test))

    print("Train RMSE:", np.sqrt(mean_squared_error(Y_train, y_pred_train)))
    print("Test  RMSE:", np.sqrt(mean_squared_error(y_test, y_pred_test)))

    scaler = model.named_steps["standardscaler"]
    ridge = model.named_steps["ridge"]

    coefs = ridge.coef_
    feature_importance = pd.Series(coefs, index=features).sort_values(key=abs, ascending=False)
    print(feature_importance)

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
    df_test=enforce_single_position(df_test)
    df_test["strategy_ret_gross"] = df_test["signal_single_pos"] * df_test["future_returns"]
    #generate_report(df_test["strategy_ret_gross"], 'btc-mm')
    roi_series = df_test.loc[df_test["strategy_ret_gross"]!=0, "strategy_ret_gross"].copy()

    # remove NaNs
    roi_series = roi_series.dropna()

    # ensure it's datetime-indexed
    roi_series.index = pd.to_datetime(roi_series.index)

    # now give returns to quantstats
    qs.reports.html(roi_series, output="btcmarketmaking.html")
    '''
    model = LinearRegression()
    model.fit(X_train[["vwap"]], Y_train)
    y_pred_train = model.predict(X_train[["vwap"]])
    y_pred_test = model.predict(X_test[["vwap"]])

    print("Train R²:", r2_score(Y_train, y_pred_train))
    print("Test  R²:", r2_score(y_test, y_pred_test))
    '''