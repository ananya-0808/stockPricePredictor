import numpy as np                      
import pandas as pd                        
import matplotlib.pyplot as plt           
import yfinance as yf                     
from sklearn.preprocessing import StandardScaler  
from tensorflow import keras               

TICKER = "AAPL"      
START = "2015-01-01"   
WINDOW = 60            
N_SPLITS = 4         
TEST_SIZE = 100       
MIN_TRAIN = 400      
EPOCHS = 25           

df = yf.download(TICKER, start=START, auto_adjust=True, progress=False)
 

if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)
 
df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
print(f"Downloaded {len(df)} trading days for {TICKER}")
df.tail()  

close = df["Close"]
features = pd.DataFrame(index=df.index)
 
features["ret_1"] = np.log(close).diff()           
features["ret_5"] = np.log(close).diff(5)         
 
features["hl_range"] = (df["High"] - df["Low"]) / close     
features["oc_change"] = (close - df["Open"]) / df["Open"]   
 
features["vol_chg"] = np.log1p(df["Volume"]).diff()
 
features["sma10_ratio"] = close / close.rolling(10).mean() - 1
features["sma30_ratio"] = close / close.rolling(30).mean() - 1

features["volatility_10"] = features["ret_1"].rolling(10).std()
 
delta = close.diff()
avg_gain = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
avg_loss = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
features["rsi_14"] = 100 - 100 / (1 + avg_gain / avg_loss.replace(0, np.nan))
 
ema12 = close.ewm(span=12, adjust=False).mean()
ema26 = close.ewm(span=26, adjust=False).mean()
macd = ema12 - ema26
features["macd"] = macd / close                                  
features["macd_signal"] = macd.ewm(span=9, adjust=False).mean() / close

sma20 = close.rolling(20).mean()
std20 = close.rolling(20).std()
features["bb_width"] = (4 * std20) / sma20                      
features["bb_pos"] = (close - (sma20 - 2 * std20)) / (4 * std20)  
 
print(f"{features.shape[1]} features built:", list(features.columns))

target = np.log(close).diff().shift(-1)

ma5 = close.rolling(5).mean()
 
# The first ~30 rows have missing values (a 30-day average needs 30 days of history).
# Keep only rows where every feature exists.
valid = features.notna().all(axis=1)
features, target, close, ma5 = features[valid], target[valid], close[valid], ma5[valid]

F = features.values.astype("float32")
end_rows = np.arange(WINDOW - 1, len(F))  
 
X_all = np.stack([F[i - WINDOW + 1 : i + 1] for i in end_rows]) 
y_all = target.values[end_rows]
dates_all = features.index[end_rows]
close_all = close.values[end_rows]
ma5_all = ma5.values[end_rows]
 
X_latest = X_all[-1:]
has_answer = ~np.isnan(y_all)
X, y = X_all[has_answer], y_all[has_answer]
dates, close_arr, ma5_arr = dates_all[has_answer], close_all[has_answer], ma5_all[has_answer]
 
print("X shape:", X.shape, "-> (samples, days in window, features)")
print("y shape:", y.shape, "-> one next-day return per sample")
 
 # %% STEP 7: THE MODEL + TRAINING FUNCTION
def build_model(window, n_features):
    """A 2-layer LSTM. An LSTM is a neural network with 'memory' for sequences."""
    model = keras.Sequential([
        keras.layers.Input(shape=(window, n_features)),
        keras.layers.LSTM(64, return_sequences=True),  # return_sequences=True passes the whole
                                                       # sequence on so a second LSTM can stack on top
        keras.layers.Dropout(0.2),                     # randomly ignores 20% of units while training;
                                                       # this fights overfitting (memorizing the past)
        keras.layers.LSTM(32),                         # second LSTM returns just its final summary
        keras.layers.Dropout(0.2),
        keras.layers.Dense(16, activation="relu"),     
        keras.layers.Dense(1),                         
    ])
    # Adam = the optimizer that adjusts the weights. Huber loss = like mean squared error,
    # but less thrown off by rare giant market moves.
    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="huber")
    return model
 
def scale(X_part, scaler):
    """Standardize features (mean 0, spread 1) so no feature dominates just because its numbers
    are bigger (RSI is ~50 while returns are ~0.01). Clip extremes to +/-5 for stability."""
    n, w, f = X_part.shape
    return np.clip(scaler.transform(X_part.reshape(-1, f)).reshape(n, w, f), -5, 5)
 
 
def fit_model(X_train, y_train):
    """Train a fresh model. Returns (model, scaler, y_std) so we can predict later."""
    keras.utils.set_random_seed(42)   # makes results repeatable run to run
 
    # IMPORTANT: the scaler learns its mean/spread from TRAINING data only. If it peeked at the
    # test period, information from the "future" would leak into training and inflate results.
    scaler = StandardScaler().fit(X_train.reshape(-1, X_train.shape[2]))
    y_std = float(np.std(y_train))            # returns are tiny (~0.01), so we rescale them too
    Xs, ys = scale(X_train, scaler), y_train / y_std

   # Hold out the LAST 10% of training data (in time order!) to monitor progress.
    n_val = max(int(len(Xs) * 0.1), 1)
    model = build_model(X_train.shape[1], X_train.shape[2])
    model.fit(
        Xs[:-n_val], ys[:-n_val],
        validation_data=(Xs[-n_val:], ys[-n_val:]),
        epochs=EPOCHS, batch_size=32, verbose=0,
        # Early stopping: quit when validation error stops improving for 5 epochs,
        # and roll back to the best version. Another guard against overfitting.
        callbacks=[keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)],
    )
    return model, scaler, y_std
 
 
def predict_returns(model, scaler, y_std, X_part):
    """Predict next-day log returns (undoing the rescaling of y)."""
    return model.predict(scale(X_part, scaler), verbose=0).ravel() * y_std

def predict_returns(model, scaler, y_std, X_part):
    """Predict next-day log returns (undoing the rescaling of y)."""
    return model.predict(scale(X_part, scaler), verbose=0).ravel() * y_std
 
 
# %% STEP 8: WALK-FORWARD VALIDATION (the honest test)
# The tutorial approach trains once and tests on the last 5%. That's one lucky-or-unlucky sample.
# Walk-forward repeats it:  train on the past -> predict the next block -> slide forward.
#
#   Fold 1: [ train ................ ][test]
#   Fold 2: [ train ....................][test]
#   Fold 3: [ train ........................][test]
#
# The model never sees the future during training, just like in real life.
n = len(y)
assert n >= MIN_TRAIN + N_SPLITS * TEST_SIZE, (
    f"Only {n} samples. Use an earlier START, or lower N_SPLITS / TEST_SIZE."
)
 
predicted = np.full(n, np.nan)   # will hold predictions for test blocks only
for k in range(N_SPLITS):
    test_end = n - (N_SPLITS - 1 - k) * TEST_SIZE
    test_start = test_end - TEST_SIZE
    print(f"Fold {k + 1}/{N_SPLITS}: training on {test_start} samples, testing on {TEST_SIZE}...")
    model, scaler, y_std = fit_model(X[:test_start], y[:test_start])
    predicted[test_start:test_end] = predict_returns(model, scaler, y_std, X[test_start:test_end])

# Keep only the rows that were actually predicted (the test blocks).
m = ~np.isnan(predicted)
pred_ret, true_ret = predicted[m], y[m]
t_dates, t_close, t_ma5 = dates[m], close_arr[m], ma5_arr[m]
print("Done. Out-of-sample predictions:", len(pred_ret))
 
 
# %% STEP 9: COMPARE AGAINST BASELINES
# A model is only impressive if it beats simple rules. We convert returns back to prices:
# price_tomorrow = price_today * e^(return).
true_price = t_close * np.exp(true_ret)
true_dir = np.sign(true_price - t_close)    # +1 went up, -1 went down

candidates = {
    "LSTM": t_close * np.exp(pred_ret),
    "Naive (tomorrow = today)": t_close,    # baseline 1: predict no change
    "5-day moving average": t_ma5,          # baseline 2: predict the recent average
}
 
rows = []
for name, pred_price in candidates.items():
    err = pred_price - true_price
    pred_dir = np.sign(pred_price - t_close)
    rows.append({
        "Model": name,
        "RMSE": np.sqrt(np.mean(err ** 2)),               # typical error, punishes big misses
        "MAE": np.mean(np.abs(err)),                      # typical error in dollars
        "MAPE %": np.mean(np.abs(err / true_price)) * 100,  # typical error as a percent
        # Directional accuracy: how often it got up-vs-down right (50% = coin flip).
        "Directional acc %": np.mean(pred_dir == true_dir) * 100 if np.any(pred_dir != 0) else np.nan,
    })
# Baseline 3: always guess "up". Stocks drift upward, so this often scores above 50%!
rows.append({"Model": "Always predict up", "Directional acc %": np.mean(true_dir > 0) * 100})
 
results = pd.DataFrame(rows).set_index("Model").round(3)
results   # displays as a table

# %% STEP 10: BACKTEST A SIMPLE STRATEGY
# Rule: if the model predicts a positive return, hold the stock tomorrow; otherwise sit in cash.
# We charge a small fee every time the position changes (0.05%), since real trading isn't free.
COST = 0.0005
position = (pred_ret > 0).astype(float)                      # 1 = holding, 0 = cash
trades = np.abs(np.diff(position, prepend=0.0))              # 1 whenever we switch
strategy_ret = position * true_ret - trades * COST           # daily result of the strategy
 
 
def summarize(r):
    equity = np.exp(np.cumsum(r))                            # growth of $1
    drawdown = equity / np.maximum.accumulate(equity) - 1    # drop from the previous peak
    return {
        "Total return %": round((equity[-1] - 1) * 100, 2),
        # Sharpe ratio = return per unit of risk, annualized (252 trading days per year).
        "Sharpe": round(r.mean() / r.std() * np.sqrt(252), 2) if r.std() > 0 else np.nan,
        "Worst drop %": round(drawdown.min() * 100, 2),      # max drawdown
    }