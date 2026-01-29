"""Aplikacja CLI do pobierania danych giełdowych NVDA i porównania modeli regresyjnych.

Uwaga: To jest przykład edukacyjny, nie porada inwestycyjna.
Źródło danych: Yahoo Finance (pakiet yfinance).

Wersja:
- pobieranie danych od wskazanej daty (domyślnie: 2016-01-01)
- porównanie modeli: regresja liniowa, SVR, RandomForestRegressor, MLPRegressor
- porównanie dla splitów: 80/20 oraz 20/80 (train/test)
- wykresy predykcji na train/test
"""

from __future__ import annotations

from datetime import datetime
import time
from pathlib import Path
import re

import numpy as np
import pandas as pd

import yfinance as yf
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from typing import Protocol, Any


class Regressor(Protocol):
    def fit(self, X, y) -> Any: ...

    def predict(self, X) -> Any: ...


# Domyślne ustawienia (bez parametryzowania CLI)
TICKER = "NVDA"
START_DATE = "2021-01-01"
END_DATE = None  # np. "2025-12-31" albo None (do dziś)
TEST_SIZES = (0.2, 0.8)  # 80/20 oraz 20/80
RANDOM_STATE = 42
PLOT = True
PLOTS_DIR = Path("plots")  # gdy brak GUI (Tk), zapisujemy wykresy tutaj


def _normalize_yfinance_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Normalizuje wynik z yfinance do postaci z kolumnami: Date, Open, High, Low, Close, Volume.

    yfinance potrafi zwracać:
    - kolumny płaskie (Open/High/...) oraz indeks DatetimeIndex
    - MultiIndex kolumn (np. ('Close','NVDA')) przy pewnych wersjach/ustawieniach
    """
    if df is None or df.empty:
        return df

    # Jeśli kolumny są MultiIndex (np. OHLCV x ticker) wybierz pierwszy ticker.
    if isinstance(df.columns, pd.MultiIndex):
        # Zwykle poziom 1 to ticker.
        tickers = list(dict.fromkeys([c[1] for c in df.columns if len(c) > 1]))
        if tickers:
            df = df.xs(tickers[0], axis=1, level=1, drop_level=True)
        else:
            # awaryjnie: spłaszcz
            df.columns = [str(c) for c in df.columns]

    # Reset index do kolumny z datą
    df = df.reset_index()

    # Ustal nazwę kolumny daty
    date_col = None
    for cand in ("Date", "Datetime", "index"):
        if cand in df.columns:
            date_col = cand
            break
    if date_col is None:
        # czasem reset_index tworzy kolumnę o nazwie None
        if df.columns[0] not in {"Open", "High", "Low", "Close", "Adj Close", "Volume"}:
            date_col = df.columns[0]

    if date_col and date_col != "Date":
        df = df.rename(columns={date_col: "Date"})

    # Przytnij nazwy kolumn (czasem pojawiają się spacje)
    df.columns = [str(c).strip() for c in df.columns]

    # Jeśli nadal mamy kolumny w stylu 'Close NVDA' itp. (rzadkie), spróbuj mapować.
    if "Close" not in df.columns:
        for base in ("Open", "High", "Low", "Close", "Volume"):
            if base in df.columns:
                continue
            matches = [
                c
                for c in df.columns
                if isinstance(c, str) and (c.startswith(base + " ") or c.endswith(" " + base))
            ]
            if len(matches) == 1:
                df = df.rename(columns={matches[0]: base})

    return df


def download_history(ticker: str, start: str, end: str | None) -> pd.DataFrame:
    """Pobiera historię notowań (dzienną) z zakresu dat.

    yfinance bywa niestabilne (rate limits, zmiany po stronie Yahoo, błędy timezone).
    Stosujemy kilka prób: download() oraz Ticker().history().
    Jeżeli Yahoo jest chwilowo niedostępne, próbujemy fallback do Stooq (CSV).
    """

    def _try_download(sym: str) -> pd.DataFrame:
        df0 = yf.download(
            sym,
            start=start,
            end=end,
            interval="1d",
            auto_adjust=False,
            progress=False,
            group_by="column",
            threads=False,
        )
        return _normalize_yfinance_dataframe(df0)

    def _try_history(sym: str) -> pd.DataFrame:
        tk = yf.Ticker(sym)
        df0 = tk.history(
            start=start,
            end=end,
            interval="1d",
            auto_adjust=False,
            actions=False,
        )
        return _normalize_yfinance_dataframe(df0)

    def _try_stooq_csv(sym: str) -> pd.DataFrame:
        """Fallback: Stooq daily CSV.

        Dla akcji z USA: np. NVDA -> nvda.us
        Źródło: https://stooq.com/q/d/l/?s=nvda.us&i=d
        """
        sym = sym.strip().lower()
        if not sym.endswith(".us"):
            sym = f"{sym}.us"

        url = f"https://stooq.com/q/d/l/?s={sym}&i=d"
        df0 = pd.read_csv(url)
        if df0 is None or df0.empty:
            return df0

        # Stooq: Date,Open,High,Low,Close,Volume (czasem brak Volume)
        df0.columns = [str(c).strip().title() for c in df0.columns]
        rename_map = {"Date": "Date", "Open": "Open", "High": "High", "Low": "Low", "Close": "Close"}
        if "Vol" in df0.columns and "Volume" not in df0.columns:
            df0 = df0.rename(columns={"Vol": "Volume"})
        if "Volume" not in df0.columns:
            df0["Volume"] = np.nan

        # Ujednolicenie typów
        df0 = df0.rename(columns=rename_map)
        df0["Date"] = pd.to_datetime(df0["Date"], errors="coerce")
        for c in ("Open", "High", "Low", "Close", "Volume"):
            if c in df0.columns:
                df0[c] = pd.to_numeric(df0[c], errors="coerce")

        # Filtr zakresu dat
        start_dt = pd.to_datetime(start)
        end_dt = pd.to_datetime(end) if end else None
        df0 = df0[df0["Date"].notna()]
        df0 = df0[df0["Date"] >= start_dt]
        if end_dt is not None:
            df0 = df0[df0["Date"] <= end_dt]

        return df0.reset_index(drop=True)

    # Warianty symbolu (czasem użytkownicy podają sufiksy typu .US itp.)
    candidates = [ticker]
    if isinstance(ticker, str) and ticker.upper() == "NVDA":
        candidates.append("NVDA")
    if isinstance(ticker, str) and "." in ticker:
        candidates.append(ticker.split(".", 1)[0])

    last_err: Exception | None = None
    for sym in candidates:
        for fetch in (_try_download, _try_history, _try_stooq_csv):
            # Krótkie retry na przypadki typu: "Expecting value..." / "No timezone found"
            for attempt in range(1, 4):
                try:
                    df = fetch(sym)
                    if df is not None and not df.empty:
                        required = {"Date", "Open", "High", "Low", "Close", "Volume"}
                        missing = required - set(df.columns)
                        if not missing:
                            out = df.sort_values("Date").reset_index(drop=True)
                            # usuń wiersze z brakami tylko w wymaganych kolumnach
                            out = out.dropna(subset=["Date", "Open", "High", "Low", "Close"])
                            return out.reset_index(drop=True)
                    last_err = RuntimeError("Pobrano pusty dataframe lub brakuje wymaganych kolumn.")
                except Exception as e:
                    last_err = e
                # backoff
                time.sleep(0.6 * attempt)

    # Jeśli tu jesteśmy, nic się nie udało
    raise RuntimeError(
        f"Nie udało się pobrać danych dla {ticker} od {start}. "
        f"Ostatni błąd: {type(last_err).__name__}: {last_err}"
    )


def make_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Tworzy cechy z danych OHLCV oraz target (Close na kolejny dzień).

    Zwraca:
      X, y, meta
    gdzie meta zawiera Date oraz Close (dla wizualizacji).
    """
    work = df.copy()

    work["Return_1d"] = work["Close"].pct_change(1)
    work["Return_5d"] = work["Close"].pct_change(5)
    work["MA_5"] = work["Close"].rolling(5).mean()
    work["MA_10"] = work["Close"].rolling(10).mean()
    work["MA_20"] = work["Close"].rolling(20).mean()
    work["Vol_5"] = work["Volume"].rolling(5).mean()
    work["Vol_20"] = work["Volume"].rolling(20).mean()

    y = work["Close"].shift(-1)

    feature_cols = [
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
        "Return_1d",
        "Return_5d",
        "MA_5",
        "MA_10",
        "MA_20",
        "Vol_5",
        "Vol_20",
    ]

    X = work[feature_cols]
    meta = work[["Date", "Close"]].copy()

    mask = X.notna().all(axis=1) & y.notna()
    X = X.loc[mask].reset_index(drop=True)
    y = y.loc[mask].reset_index(drop=True)
    meta = meta.loc[mask].reset_index(drop=True)

    return X, y, meta


def build_models(random_state: int) -> dict[str, Regressor]:
    """Zwraca słownik nazw -> estymator (pipeline)."""
    return {
        "LinearRegression": Pipeline(
            steps=[("scaler", StandardScaler()), ("model", LinearRegression())]
        ),
        "SVR (RBF)": Pipeline(
            steps=[("scaler", StandardScaler()), ("model", SVR(C=10.0, gamma="scale"))]
        ),
        "RandomForest": RandomForestRegressor(
            n_estimators=800,
            random_state=random_state,
            n_jobs=-1,
            min_samples_leaf=2,
        ),
        "MLPRegressor": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "model",
                    MLPRegressor(
                        hidden_layer_sizes=(128, 64),
                        activation="relu",
                        solver="adam",
                        random_state=random_state,
                        max_iter=800,
                        early_stopping=True,
                        validation_fraction=0.15,
                        n_iter_no_change=25,
                    ),
                ),
            ]
        ),
    }


def evaluate_model(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = float(r2_score(y_true, y_pred))
    return {"MAE": mae, "RMSE": rmse, "R2": r2}


def chronological_split(X: pd.DataFrame, y: pd.Series, meta: pd.DataFrame, test_size: float):
    """Podział train/test bez mieszania (szereg czasowy)."""
    X_train, X_test, y_train, y_test, meta_train, meta_test = train_test_split(
        X, y, meta, test_size=test_size, shuffle=False
    )
    return X_train, X_test, y_train, y_test, meta_train, meta_test


def plot_predictions(
    ticker: str,
    split_label: str,
    model_name: str,
    meta_train: pd.DataFrame,
    y_train: pd.Series,
    pred_train: np.ndarray,
    meta_test: pd.DataFrame,
    y_test: pd.Series,
    pred_test: np.ndarray,
):
    # Matplotlib często domyślnie używa backendu Tk (TkAgg). Jeśli Tcl/Tk jest
    # źle zainstalowane (częste na Windows), tworzenie okna kończy się TclError.
    # Żeby aplikacja NIE crashowała, przełączamy się na backend "Agg"
    # i zapisujemy wykresy do plików.
    try:
        import matplotlib as mpl
        mpl.use("Agg")  # bez GUI, działa nawet gdy Tcl/Tk nie działa
        import matplotlib.pyplot as plt
    except Exception:
        return

    try:
        fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharey=True)
        fig.suptitle(f"{ticker} — {model_name} — split {split_label}")

        axes[0].plot(meta_train["Date"], y_train.values, label="True", linewidth=1)
        axes[0].plot(meta_train["Date"], pred_train, label="Pred", linewidth=1)
        axes[0].set_title("Train")
        axes[0].legend()

        axes[1].plot(meta_test["Date"], y_test.values, label="True", linewidth=1)
        axes[1].plot(meta_test["Date"], pred_test, label="Pred", linewidth=1)
        axes[1].set_title("Test")
        axes[1].legend()

        plt.tight_layout()

        PLOTS_DIR.mkdir(parents=True, exist_ok=True)
        safe_model = re.sub(r"[^A-Za-z0-9._-]+", "_", model_name).strip("_")
        safe_split = re.sub(r"[^A-Za-z0-9._-]+", "_", split_label).strip("_")
        out_path = PLOTS_DIR / f"{ticker}_{safe_model}_{safe_split}.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
    finally:
        # ważne przy wielu wykresach w pętli
        try:
            plt.close("all")
        except Exception:
            pass


def main() -> int:
    history = download_history(TICKER, START_DATE, END_DATE)
    X, y, meta = make_features(history)

    if len(X) < 200:
        raise RuntimeError("Za mało danych po przygotowaniu cech. Sprawdź zakres dat.")

    models = build_models(RANDOM_STATE)

    rows: list[dict[str, object]] = []

    for test_size in TEST_SIZES:
        split_label = f"{int((1 - test_size) * 100)}/{int(test_size * 100)}"
        X_train, X_test, y_train, y_test, meta_train, meta_test = chronological_split(
            X, y, meta, test_size
        )

        for model_name, model in models.items():
            model.fit(X_train, y_train)
            pred_train = np.asarray(model.predict(X_train), dtype=float)
            pred_test = np.asarray(model.predict(X_test), dtype=float)

            m_train = evaluate_model(y_train.values, pred_train)
            m_test = evaluate_model(y_test.values, pred_test)

            rows.append(
                {
                    "Split": split_label,
                    "Model": model_name,
                    "Train_MAE": m_train["MAE"],
                    "Train_RMSE": m_train["RMSE"],
                    "Train_R2": m_train["R2"],
                    "Test_MAE": m_test["MAE"],
                    "Test_RMSE": m_test["RMSE"],
                    "Test_R2": m_test["R2"],
                    "N_train": len(X_train),
                    "N_test": len(X_test),
                }
            )

            if PLOT:
                plot_predictions(
                    TICKER,
                    split_label,
                    model_name,
                    meta_train,
                    y_train,
                    pred_train,
                    meta_test,
                    y_test,
                    pred_test,
                )

    results = pd.DataFrame(rows)
    results = results.sort_values(["Split", "Test_MAE"]).reset_index(drop=True)

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 140)

    print(f"Ticker: {TICKER}")
    print(f"Zakres danych: {history['Date'].min().date()} -> {history['Date'].max().date()}")
    print(f"Trening/Test: {', '.join([f'{int((1-ts)*100)}/{int(ts*100)}' for ts in TEST_SIZES])}")
    print(f"Wygenerowano: {datetime.now().isoformat(timespec='seconds')}")
    print("\nPorównanie modeli (im mniejsze MAE/RMSE tym lepiej, im większe R2 tym lepiej):\n")
    print(results)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
