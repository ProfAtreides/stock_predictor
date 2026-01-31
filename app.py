import argparse
import datetime as dt
from dataclasses import dataclass

import numpy as np
import pandas as pd
import yfinance as yf

import urllib.parse

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras


@dataclass(frozen=True)
class Symbols:
    # Yahoo (fallback)
    kghm_yf: str = "KGH.WA"
    copper_yf: str = "HG=F"
    silver_yf: str = "SI=F"
    wig_yf: str = "^WIG"

    kghm_stooq: str = "kgh"
    wig_stooq: str = "wig"

    copper_stooq: str = "copper"
    silver_stooq: str = "silver"


def _extract_close(df: pd.DataFrame, symbol_hint: str | None = None) -> pd.Series:
    if df is None or df.empty:
        raise RuntimeError("Pusty DataFrame")

    if isinstance(df, pd.Series):
        return df.copy()

    if "Close" in df.columns:
        s = df["Close"]
        if isinstance(s, pd.DataFrame):
            s = s.iloc[:, 0]
        return s.copy()

    if isinstance(df.columns, pd.MultiIndex):
        level0 = df.columns.get_level_values(0)  # type: ignore[attr-defined]
        if symbol_hint and ("Close", symbol_hint) in df.columns:
            s = df[("Close", symbol_hint)]
            if isinstance(s, pd.DataFrame):
                s = s.iloc[:, 0]
            return s.copy()
        if "Close" in level0:
            close_block = df.xs("Close", axis=1, level=0)
            return close_block.iloc[:, 0].copy()

    return df.iloc[:, 0].copy()


def download_close_stooq(symbol: str, start: str, end: str) -> pd.Series:
    d1 = pd.to_datetime(start).strftime("%Y%m%d")
    d2 = pd.to_datetime(end).strftime("%Y%m%d")

    params = {"s": symbol, "i": "d", "d1": d1, "d2": d2}
    url = "https://stooq.com/q/d/l/?" + urllib.parse.urlencode(params)

    df = pd.read_csv(url)
    if df is None or df.empty:
        raise RuntimeError(f"Brak danych Stooq dla symbolu: {symbol}")

    # kolumny zwykle: Date,Open,High,Low,Close,Volume
    if "Date" not in df.columns or "Close" not in df.columns:
        raise RuntimeError(f"Nieoczekiwany format CSV Stooq dla: {symbol}")

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"]).sort_values("Date")
    s = df.set_index("Date")["Close"].astype(float).copy()
    s.name = symbol
    return s


def download_close_yfinance(symbol: str, start: str, end: str) -> pd.Series:
    df = yf.download(symbol, start=start, end=end, progress=False, auto_adjust=False, group_by="column")
    if df is None or df.empty:
        raise RuntimeError(f"Brak danych Yahoo dla symbolu: {symbol}")
    s = _extract_close(df, symbol_hint=symbol)
    s.name = symbol
    return s


def download_close_any(stooq_symbol: str, yf_symbol: str, start: str, end: str) -> pd.Series:
    try:
        s = download_close_stooq(stooq_symbol, start, end)
        if s.dropna().empty:
            raise RuntimeError("Stooq zwrócił same NaN")
        return s
    except Exception:
        return download_close_yfinance(yf_symbol, start, end)


def build_dataset(start: str, end: str, symbols: Symbols) -> pd.DataFrame:
    kghm = download_close_any(symbols.kghm_stooq, symbols.kghm_yf, start, end)
    kghm.name = "kghm"

    wig = download_close_any(symbols.wig_stooq, symbols.wig_yf, start, end)
    wig.name = "wig"

    copper = download_close_any(symbols.copper_stooq, symbols.copper_yf, start, end)
    copper.name = "copper"

    silver = download_close_any(symbols.silver_stooq, symbols.silver_yf, start, end)
    silver.name = "silver"

    df = pd.concat([kghm, copper, silver, wig], axis=1)
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()

    for c in ["copper", "silver", "wig"]:
        df[c] = df[c].ffill().bfill()

    df = df.dropna(subset=["kghm"]).copy()
    return df

def add_supervised_target(df: pd.DataFrame, horizon_days: int = 1) -> pd.DataFrame:
    out = df.copy()

    out["y"] = np.log(out["kghm"].shift(-horizon_days) / out["kghm"])  # t -> t+1

    out["kghm_t"] = out["kghm"]

    out = out.dropna(subset=["y", "kghm_t"]).copy()
    return out


def time_split(df: pd.DataFrame, train_ratio: float):
    n = len(df)
    n_train = int(np.floor(n * train_ratio))
    train = df.iloc[:n_train].copy()
    test = df.iloc[n_train:].copy()
    return train, test


def metrics(y_true, y_pred) -> dict:
    mse = mean_squared_error(y_true, y_pred)
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mse)),
        "R2": float(r2_score(y_true, y_pred)),
    }



def _to_log_price(x: np.ndarray | pd.Series) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64)
    return np.log(arr)


def _from_log_price(x: np.ndarray | pd.Series) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64)
    return np.exp(arr)


def _clip(arr: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return np.clip(arr, lo, hi)

def returns_to_price(kghm_t: np.ndarray, y_logret: np.ndarray) -> np.ndarray:
    k0 = np.asarray(kghm_t, dtype=np.float64)
    r = np.asarray(y_logret, dtype=np.float64)
    return k0 * np.exp(r)


def sklearn_tabular_models(train: pd.DataFrame, test: pd.DataFrame, random_state: int = 42):
    feature_cols = ["copper", "silver", "wig"]

    X_train = train[feature_cols]
    y_train = train["y"].astype(float)
    X_test = test[feature_cols]
    y_test = test["y"].astype(float)

    pre = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", RobustScaler(with_centering=True, with_scaling=True)),
                    ]
                ),
                feature_cols,
            )
        ],
        remainder="drop",
    )

    lin = Pipeline(steps=[("pre", pre), ("model", LinearRegression())])

    mlp = Pipeline(
        steps=[
            ("pre", pre),
            (
                "model",
                MLPRegressor(
                    hidden_layer_sizes=(64, 32),
                    activation="relu",
                    solver="adam",
                    alpha=1e-3,
                    learning_rate_init=5e-4,
                    max_iter=2000,
                    early_stopping=True,
                    n_iter_no_change=25,
                    random_state=random_state,
                ),
            ),
        ]
    )

    lin.fit(X_train, y_train)
    mlp.fit(X_train, y_train)

    pred_lin_r = lin.predict(X_test)
    pred_mlp_r = mlp.predict(X_test)

    y_true_price = returns_to_price(test["kghm_t"].to_numpy(), y_test.to_numpy())
    pred_lin_price = returns_to_price(test["kghm_t"].to_numpy(), pred_lin_r)
    pred_mlp_price = returns_to_price(test["kghm_t"].to_numpy(), pred_mlp_r)

    return {
        "linear_regression": {
            "model": lin,
            "metrics": metrics(y_true_price, pred_lin_price),
            "y_true": y_true_price,
            "y_pred": pred_lin_price,
            "dates": test.index.to_numpy(),
        },
        "mlp_regressor": {
            "model": mlp,
            "metrics": metrics(y_true_price, pred_mlp_price),
            "y_true": y_true_price,
            "y_pred": pred_mlp_price,
            "dates": test.index.to_numpy(),
        },
    }


def make_windows(df: pd.DataFrame, lookback: int, horizon_days: int = 1):
    X = df[["copper", "silver", "wig"]].to_numpy(dtype=np.float32)
    y = df["y"].to_numpy(dtype=np.float32)  # log-return celu
    kghm_t = df["kghm_t"].to_numpy(dtype=np.float32)

    Xs = []
    ys = []
    kts = []
    dates = []
    for i in range(lookback - 1, len(df) - horizon_days):
        Xs.append(X[i - lookback + 1 : i + 1])
        ys.append(y[i])
        kts.append(kghm_t[i])
        dates.append(df.index[i])

    Xs = np.stack(Xs, axis=0)
    ys = np.array(ys, dtype=np.float32)
    kts = np.array(kts, dtype=np.float32)
    dates = np.array(dates)
    return Xs, ys, kts, dates


def build_cnn_model(lookback: int, n_features: int):
    inputs = keras.Input(shape=(lookback, n_features))
    x = keras.layers.Conv1D(
        filters=32,
        kernel_size=3,
        padding="causal",
        activation="relu",
        kernel_regularizer=keras.regularizers.l2(1e-4),
    )(inputs)
    x = keras.layers.Conv1D(
        filters=32,
        kernel_size=3,
        padding="causal",
        activation="relu",
        kernel_regularizer=keras.regularizers.l2(1e-4),
    )(x)
    x = keras.layers.GlobalAveragePooling1D()(x)
    x = keras.layers.Dense(64, activation="relu", kernel_regularizer=keras.regularizers.l2(1e-4))(x)
    x = keras.layers.Dropout(0.2)(x)
    outputs = keras.layers.Dense(1)(x)

    model = keras.Model(inputs, outputs)
    model.compile(optimizer=keras.optimizers.Adam(5e-4), loss="mse")
    return model


def cnn_model(train: pd.DataFrame, test: pd.DataFrame, lookback: int = 20, random_state: int = 42):
    tf.random.set_seed(random_state)
    np.random.seed(random_state)

    scaler = RobustScaler(with_centering=True, with_scaling=True)
    train_X_raw = train[["copper", "silver", "wig"]].to_numpy(dtype=np.float32)
    test_X_raw = test[["copper", "silver", "wig"]].to_numpy(dtype=np.float32)

    scaler.fit(train_X_raw)

    train_scaled = train.copy()
    test_scaled = test.copy()
    train_scaled[["copper", "silver", "wig"]] = scaler.transform(train_X_raw)
    test_scaled[["copper", "silver", "wig"]] = scaler.transform(test_X_raw)

    X_train, y_train_r, _, _ = make_windows(train_scaled, lookback=lookback, horizon_days=1)

    bridge = pd.concat([train_scaled.iloc[-(lookback - 1):], test_scaled], axis=0)
    X_test, y_test_r, kt_test, dates = make_windows(bridge, lookback=lookback, horizon_days=1)

    model = build_cnn_model(lookback=lookback, n_features=3)

    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=7, min_lr=1e-5),
    ]

    model.fit(
        X_train,
        y_train_r,
        validation_split=0.2,
        epochs=300,
        batch_size=32,
        verbose=0,
        callbacks=callbacks,
    )

    pred_r = model.predict(X_test, verbose=0).reshape(-1)

    y_true = returns_to_price(kt_test, y_test_r)
    y_pred = returns_to_price(kt_test, pred_r)

    return {
        "cnn": {
            "model": model,
            "scaler": scaler,
            "metrics": metrics(y_true, y_pred),
            "y_true": y_true,
            "y_pred": y_pred,
            "dates": dates,
        }
    }


def plot_predictions(results: dict, title: str, out_path: str | None = None, train_df: pd.DataFrame | None = None):
    plt.figure(figsize=(12, 5))

    common_dates = None
    for r in results.values():
        d = pd.to_datetime(r["dates"])
        s = pd.Index(d)
        common_dates = s if common_dates is None else common_dates.intersection(s)

    if common_dates is None or len(common_dates) == 0:
        raise RuntimeError("Brak wspólnych dat do porównania predykcji")

    common_dates = common_dates.sort_values()

    if train_df is not None and len(train_df) > 0:
        train_dates = pd.to_datetime(train_df.index)
        # w train_df target to log-zwrot; do ceny używamy kghm_t oraz y
        train_price = returns_to_price(train_df["kghm_t"].to_numpy(), train_df["y"].to_numpy())
        plt.plot(train_dates, train_price, label="Rzeczywiste (train)", linewidth=2, alpha=0.9)
        train_end_date = train_dates.max()
    else:
        train_end_date = None

    first = next(iter(results.values()))
    first_df = pd.DataFrame(
        {"y": np.asarray(first["y_true"], dtype=float)},
        index=pd.to_datetime(first["dates"]),
    ).reindex(common_dates)

    plt.plot(common_dates, first_df["y"].to_numpy(), label="Rzeczywiste (test)", linewidth=2)

    for name, r in results.items():
        pred_df = pd.DataFrame(
            {"y": np.asarray(r["y_pred"], dtype=float)},
            index=pd.to_datetime(r["dates"]),
        ).reindex(common_dates)
        plt.plot(common_dates, pred_df["y"].to_numpy(), label=f"{name} (test)")

    if train_end_date is not None:
        plt.axvline(x=train_end_date, color="black", linestyle="--", linewidth=1.5, label="Koniec treningu")

    plt.title(title)
    plt.xlabel("Data")
    plt.ylabel("Cena KGHM - predykcja t+1")
    plt.legend()
    plt.tight_layout()

    if out_path:
        plt.savefig(out_path, dpi=150)
    else:
        plt.show()


def run_experiment(period_label: str, df: pd.DataFrame, train_ratio: float, lookback: int = 20):
    df_sup = add_supervised_target(df, horizon_days=1)
    train, test = time_split(df_sup, train_ratio=train_ratio)

    sk = sklearn_tabular_models(train, test)
    cn = cnn_model(train, test, lookback=lookback)

    combined = {**{k: v for k, v in sk.items()}, **{k: v for k, v in cn.items()}}

    print(f"\n=== Okres: {period_label} | Podział train/test: {int(train_ratio*100)}/{int((1-train_ratio)*100)} ===")
    print(f"Zakres dat: {df.index.min().date()} -> {df.index.max().date()} | próbek: {len(df_sup)}")
    for name, r in combined.items():
        m = r["metrics"]
        print(f"{name:>18} | MAE={m['MAE']:.4f} RMSE={m['RMSE']:.4f} R2={m['R2']:.4f}")

    return combined, train


def _prepare_plot_series(results: dict, train_df: pd.DataFrame | None):
    common_dates = None
    for r in results.values():
        d = pd.to_datetime(r["dates"])
        s = pd.Index(d)
        common_dates = s if common_dates is None else common_dates.intersection(s)
    if common_dates is None or len(common_dates) == 0:
        raise RuntimeError("Brak wspólnych dat do porównania predykcji")
    common_dates = common_dates.sort_values()

    first = next(iter(results.values()))
    y_true_test = (
        pd.DataFrame({"y": np.asarray(first["y_true"], dtype=float)}, index=pd.to_datetime(first["dates"]))
        .reindex(common_dates)["y"]
    )

    preds_test = {}
    for name, r in results.items():
        preds_test[name] = (
            pd.DataFrame({"y": np.asarray(r["y_pred"], dtype=float)}, index=pd.to_datetime(r["dates"]))
            .reindex(common_dates)["y"]
        )

    train_dates = None
    train_price = None
    train_end_date = None
    if train_df is not None and len(train_df) > 0:
        train_dates = pd.to_datetime(train_df.index)
        train_price = pd.Series(
            returns_to_price(train_df["kghm_t"].to_numpy(), train_df["y"].to_numpy()),
            index=train_dates,
            name="train_price",
        )
        train_end_date = train_dates.max()

    return train_price, train_end_date, y_true_test, preds_test


def _filter_last_year(y_true_test: pd.Series, preds_test: dict[str, pd.Series]):
    if y_true_test.empty:
        return y_true_test, preds_test
    cutoff = y_true_test.index.max() - pd.Timedelta(days=365)
    y_true_f = y_true_test[y_true_test.index >= cutoff]
    preds_f = {k: v[v.index >= cutoff] for k, v in preds_test.items()}
    return y_true_f, preds_f


def _filter_last_year_from_end(y_true_test: pd.Series, preds_test: dict[str, pd.Series], end_date) -> tuple[pd.Series, dict[str, pd.Series]]:
    if y_true_test.empty:
        return y_true_test, preds_test
    end_ts = pd.to_datetime(end_date)
    cutoff = end_ts - pd.Timedelta(days=365)
    y_true_f = y_true_test[(y_true_test.index >= cutoff) & (y_true_test.index <= end_ts)]
    preds_f = {k: v[(v.index >= cutoff) & (v.index <= end_ts)] for k, v in preds_test.items()}
    return y_true_f, preds_f


def _plot_on_axis(
    ax,
    title: str,
    train_price: pd.Series | None,
    train_end_date,
    y_true_test: pd.Series,
    preds_test: dict[str, pd.Series],
    show_train: bool = True,
    actual_price: pd.Series | None = None,
    show_actual_left: bool = False,
):
    ax_pred = ax

    ''''''
    if show_actual_left and actual_price is not None and not actual_price.empty:
        ax_left = ax_pred
        ax_right = ax_pred.twinx()

        ax_pred = ax_right
        ax_pred.set_ylabel("Cena KGHM (t+1) – predykcja")

    if show_train and train_price is not None:
        ax_pred.plot(train_price.index, train_price.values, label="Rzeczywiste (train)", linewidth=2, alpha=0.9)

    ax_pred.plot(y_true_test.index, y_true_test.values, label="Rzeczywiste (test)", linewidth=2)

    for name, s in preds_test.items():
        ax_pred.plot(s.index, s.values, label=f"{name} (test)")

    if show_train and train_end_date is not None:
        ax_pred.axvline(
            x=pd.to_datetime(train_end_date),
            color="black",
            linestyle="--",
            linewidth=1.5,
            label="Koniec treningu",
        )

    ax.set_title(title)
    ax.set_xlabel("Data")
    ax.grid(True, alpha=0.2)

    handles, labels = [], []
    for a in ({ax} | {ax_pred} | ({ax_pred.axes} if hasattr(ax_pred, "axes") else set())):
        try:
            h, l = a.get_legend_handles_labels()
            handles += h
            labels += l
        except Exception:
            pass
    if handles:
        ax.legend(handles, labels, fontsize=8)


def render_and_save_dashboard(
    results_by_split: dict[float, dict],
    trains_by_split: dict[float, pd.DataFrame],
    dataset_end_date,
    raw_price: pd.Series,
):
    fig, axes = plt.subplots(2, 2, figsize=(16, 9), sharey=False)

    panels = [
        (0.2, "pełny", "5y_20_80_full.png"),
        (0.2, "ostatni_rok", "5y_20_80_last_year.png"),
        (0.4, "pełny", "5y_40_60_full.png"),
        (0.4, "ostatni_rok", "5y_40_60_last_year.png"),
    ]

    for ax, (split, mode, out_path) in zip(axes.ravel(), panels, strict=False):
        train_price, train_end, y_true_test, preds_test = _prepare_plot_series(
            results_by_split[split],
            trains_by_split[split],
        )

        show_train = True
        show_actual_left = False
        actual_series = None

        if mode == "ostatni_rok":
            y_true_test, preds_test = _filter_last_year_from_end(y_true_test, preds_test, dataset_end_date)
            title = f"5 lat | split {int(split*100)}/{int((1-split)*100)} | ostatni rok"
            show_train = False
        else:
            title = f"5 lat | split {int(split*100)}/{int((1-split)*100)} | 5 lat"
            show_actual_left = True
            actual_series = raw_price

        _plot_on_axis(
            ax,
            title,
            train_price,
            train_end,
            y_true_test,
            preds_test,
            show_train=show_train,
            actual_price=actual_series,
            show_actual_left=show_actual_left,
        )

        tmp_fig = plt.figure(figsize=(12, 5))
        tmp_ax = tmp_fig.add_subplot(1, 1, 1)
        _plot_on_axis(
            tmp_ax,
            title,
            train_price,
            train_end,
            y_true_test,
            preds_test,
            show_train=show_train,
            actual_price=actual_series,
            show_actual_left=show_actual_left,
        )
        tmp_fig.tight_layout()
        tmp_fig.savefig(out_path, dpi=150)
        plt.close(tmp_fig)

    fig.suptitle("KGHM – 5 lat – porównanie modeli")
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    fig.savefig("dashboard_5y_splits.png", dpi=150)
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Predykcja cen KGHM na bazie: miedź, srebro, WIG")
    parser.add_argument("--end", type=str, default=dt.date.today().isoformat(), help="Data końcowa (YYYY-MM-DD)")
    args = parser.parse_args()

    symbols = Symbols()

    end = pd.to_datetime(args.end).date()
    start_5y = (end - dt.timedelta(days=365 * 5)).isoformat()
    end_s = (end + dt.timedelta(days=1)).isoformat()

    df_5y = build_dataset(start_5y, end_s, symbols)

    splits = [0.2, 0.4]

    results_by_split = {}
    trains_by_split = {}

    for split in splits:
        results_by_split[split], trains_by_split[split] = run_experiment("5 lat", df_5y, train_ratio=split)

    render_and_save_dashboard(
        results_by_split,
        trains_by_split,
        dataset_end_date=df_5y.index.max(),
        raw_price=df_5y["kghm"].copy(),
    )


if __name__ == "__main__":
    main()
