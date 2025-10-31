# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

# ========== Config ==========
DATES = ["2025-09-01", "2025-09-02", "2025-09-03"]  # Dates to generate (local Adelaide calendar)
INPUT_STEPS = 336               # Lookback window (hours) ~ 14 days
HORIZON_HOURS = 24              # Predict next 24 hours
OUTPUT_DIR = Path.cwd() / "outputs"

# ========== A. Time parsing: treat Excel times as local wall-clock ==========
LOCAL_TZ = "Australia/Adelaide"

def to_adelaide_time(df: pd.DataFrame) -> pd.DataFrame:
    """
    The Excel 'time' column is already local wall-clock time for Adelaide.
    - If the parsed timestamps are naive (no tz), keep them as is (no tz conversion).
    - If the column happens to be timezone-aware, convert to Adelaide and drop tz info.
    """
    out = df.copy()
    t = pd.to_datetime(out["time"], errors="coerce")
    if getattr(t.dt, "tz", None) is not None:
        t = t.dt.tz_convert(LOCAL_TZ).dt.tz_localize(None)
    out["time"] = t
    return out

# ========== B. Feature engineering (with seasonal features) ==========
def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["time"] = pd.to_datetime(df["time"])
    df = df.sort_values("time").reset_index(drop=True)

    # Hour-of-day sine/cosine (daily cycle)
    df["hour"] = df["time"].dt.hour
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

    # Seasonal sine/cosine (annual cycle)
    doy = df["time"].dt.dayofyear.astype(float)
    df["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)

    # Lag & smoothing (keep original column names)
    df["irr_prev_day"] = df["shortwave_radiation (W/m2)"].shift(24).bfill()
    df["irr_ma3"] = (
        df["shortwave_radiation (W/m2)"].rolling(window=7, min_periods=1).mean()
    )
    return df

FEATURE_COLS = [
    "temperature_2m",
    "relative_humidity_2m (%)",
    "precipitation (mm)",
    "wind_speed_10m (km/h)",
    "cloud_cover (%)",
    "hour_sin", "hour_cos",
    "doy_sin", "doy_cos",        # seasonal features
    "irr_prev_day", "irr_ma3",
]
TARGET_COL = "shortwave_radiation (W/m2)"

# ========== C. Sliding windows & inverse transform ==========
def create_sequences(df, scaler, input_steps, output_steps, feature_cols, target_col):
    data = scaler.transform(df[feature_cols + [target_col]])
    X, y = [], []
    for i in range(len(data) - input_steps - output_steps + 1):
        X.append(data[i:i + input_steps, :-1])                           # features
        y.append(data[i + input_steps:i + input_steps + output_steps, -1])  # target
    return np.array(X), np.array(y)

def inverse_only_target(pred_scaled: np.ndarray, scaler: MinMaxScaler) -> np.ndarray:
    """Inverse-transform only the target column back to W/m²."""
    res = []
    for arr in pred_scaled:
        arr = arr.reshape(-1, 1)
        pad = np.zeros((arr.shape[0], scaler.n_features_in_ - 1))
        inv = scaler.inverse_transform(np.concatenate([pad, arr], axis=1))[:, -1]
        res.append(inv)
    return np.array(res)

# ========== D. Daylight mask (monthly approximation) ==========
MONTH_DAYLIGHT = {
    1:(6.1,20.4),  2:(6.5,20.0),  3:(7.0,19.2),
    4:(7.3,18.2),  5:(7.0,17.2),  6:(6.6,17.0),
    7:(6.9,17.2),  8:(6.6,17.8),  9:(6.1,18.3),
    10:(5.6,19.0), 11:(5.8,19.8), 12:(6.0,20.3),
}
def daylight_mask(hours_index: pd.DatetimeIndex) -> np.ndarray:
    """Return a boolean mask for daylight hours (True=daylight) using a coarse monthly table."""
    m = int(hours_index[0].month)
    rise, set_ = MONTH_DAYLIGHT.get(m, (6.0, 18.5))
    hour_float = hours_index.hour + hours_index.minute/60.0
    return (hour_float >= np.floor(rise)) & (hour_float <= np.ceil(set_))

# ========== 1) Read data ==========
train_df = pd.read_excel("天气和辐照数据2024.xlsx")
pred_df  = pd.read_excel("天气和辐照数据2025.xlsx")

# Parse as local wall-clock time
train_df = to_adelaide_time(train_df)
pred_df  = to_adelaide_time(pred_df)

# Features
train_df = feature_engineering(train_df)
pred_df  = feature_engineering(pred_df)

# ========== 2) Scaling & training ==========
scaler = MinMaxScaler()
scaler.fit(train_df[FEATURE_COLS + [TARGET_COL]])

X_train, y_train = create_sequences(train_df, scaler, INPUT_STEPS, HORIZON_HOURS, FEATURE_COLS, TARGET_COL)
print(f"Train samples: {X_train.shape[0]}")

model = Sequential([
    LSTM(128, return_sequences=True, input_shape=(INPUT_STEPS, len(FEATURE_COLS))),
    Dropout(0.3),
    LSTM(64),
    Dropout(0.2),
    Dense(HORIZON_HOURS),
])
model.compile(optimizer="adam", loss="mse")
early_stop = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2, callbacks=[early_stop])

# ========== 3) Batch predict multiple days + save CSV + plot comparison ==========
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def run_one_day(date_str: str):
    target_date = pd.Timestamp(date_str)
    t0 = pd.Timestamp(f"{target_date.date()} 00:00:00")

    # Context = tail of training + all of prediction dataframe
    context_cols = ["time"] + FEATURE_COLS + [TARGET_COL]
    context_df = pd.concat(
        [train_df[context_cols].tail(INPUT_STEPS), pred_df[context_cols]],
        ignore_index=True
    )

    # First index >= t0 → right boundary of the input window (exclusive)
    idx_after = context_df.index[context_df["time"] >= t0]
    if len(idx_after) == 0:
        print(f"[SKIP] {date_str}: out of data range.")
        return
    end_idx = idx_after[0]
    if end_idx - INPUT_STEPS < 0:
        print(f"[SKIP] {date_str}: not enough history ({INPUT_STEPS}h).")
        return

    # Predict
    data_all = scaler.transform(context_df[FEATURE_COLS + [TARGET_COL]])
    x_infer = data_all[end_idx - INPUT_STEPS:end_idx, :-1][np.newaxis, ...]
    y_pred_scaled = model.predict(x_infer, verbose=0)
    y_pred = inverse_only_target(y_pred_scaled, scaler)[0]  # [24,]

    # Constraints: non-negative + daylight mask
    hours_local = pd.date_range(start=t0, periods=24, freq="h")
    y_pred = np.clip(y_pred, 0.0, None)
    y_pred = np.where(daylight_mask(hours_local), y_pred, 0.0)

    # Save CSV
    csv_path = OUTPUT_DIR / f"lstm_forecast_{target_date.date()}.csv"
    pd.DataFrame({
        "time_local_adelaide": hours_local,
        "predicted_irradiance_Wm2": y_pred
    }).to_csv(csv_path, index=False, encoding="utf-8-sig")

    # True values (hourly, aligned)
    true_all = pd.concat(
        [train_df[["time", TARGET_COL]], pred_df[["time", TARGET_COL]]],
        ignore_index=True
    ).dropna(subset=["time"]).sort_values("time")

    true_hourly = (
        true_all
        .loc[(true_all["time"] >= t0) & (true_all["time"] < t0 + pd.Timedelta(days=1))]
        .set_index("time")[TARGET_COL]
        .resample("h").mean()
        .reindex(hours_local)
        .interpolate(method="time", limit=3, limit_direction="both")
        .fillna(0.0)
        .clip(lower=0.0)
    )
    y_true = true_hourly.to_numpy(dtype=float)

    # Metrics
    mse = float(np.mean((y_pred - y_true) ** 2))
    mae = float(np.mean(np.abs(y_pred - y_true)))

    # Plot & save
    plt.figure(figsize=(10, 6))
    plt.plot(range(24), y_true, label="True")
    plt.plot(range(24), y_pred, label="Predicted")
    plt.xlim(0, 23); plt.xticks(range(24))
    plt.xlabel("Hour"); plt.ylabel("Irradiance (W/m²)")
    plt.title(f"LSTM 24-hour Irradiance Forecast for {target_date.date()}\n"
              f"MSE={mse:.2f}   MAE={mae:.2f}")
    plt.legend(); plt.tight_layout()
    fig_path = OUTPUT_DIR / f"Fig_LSTM_{target_date.date()}.png"
    plt.savefig(fig_path, dpi=300)
    plt.close()

    print(f"[OK] {date_str}: CSV→{csv_path.name}  FIG→{fig_path.name}  MSE={mse:.2f}  MAE={mae:.2f}")

# Run for all dates
for d in DATES:
    run_one_day(d)
