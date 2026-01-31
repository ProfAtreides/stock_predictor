# KGHM Predictor (miedź + srebro + WIG)

Projekt służy do **predykcji ceny akcji KGHM na następny dzień (t+1)** na podstawie danych rynkowych:
- ceny miedzi,
- ceny srebra,
- indeksu WIG.

Aplikacja porównuje 3 podejścia ML:
1. **Regresja liniowa (LinearRegression)**
2. **Sieć MLP (MLPRegressor, scikit-learn)**
3. **Sieć konwolucyjna 1D (CNN, Keras/TensorFlow)**

Dodatkowo wykonywane są dwa podziały danych w czasie (time split), aby porównać zachowanie modeli:
- **20/80** (train/test)
- **40/60** (train/test)

Wyniki są:
- wypisywane w konsoli (MAE, RMSE, R²),
- wizualizowane w oknie Matplotlib (GUI) jako **4 wykresy** (2 splity × 2 zakresy),
- zapisywane do plików PNG.

---

## Skąd są dane?

### Źródła
- **KGHM oraz WIG**: Stooq (CSV API)
- **Miedź i srebro**: domyślnie próba ze Stooq, a jeśli brak danych – fallback do Yahoo Finance (`yfinance`).

W kodzie znajduje się logika „spróbuj Stooq, jak się nie uda to Yahoo”, żeby projekt działał bez ręcznej zmiany tickerów.

### Symbole
Zdefiniowane w klasie `Symbols`:
- `kghm_stooq = "kgh"` (KGHM)
- `wig_stooq = "wig"` (WIG)
- `copper_yf = "HG=F"` (miedź – futures)
- `silver_yf = "SI=F"` (srebro – futures)

---

## Jaki jest target (co dokładnie przewidujemy)?

Zamiast uczyć modele bezpośrednio na **poziomie ceny**, projekt uczy się na **logarytmicznym zwrocie** KGHM:

\[
 y_t = \ln\left(\frac{P_{t+1}}{P_t}\right)
\]

To podejście ma dwa cele:
- **stabilizuje uczenie** (zwłaszcza dla sieci neuronowych),
- zmniejsza ryzyko „agresywnych”/odjechanych predykcji.

Dopiero do metryk i wykresów zwrot jest przeliczany z powrotem na cenę:

\[
 \hat{P}_{t+1} = P_t \cdot e^{\hat{y}_t}
\]

W danych po przygotowaniu pojawiają się kolumny:
- `y` – log-zwrot KGHM (target)
- `kghm_t` – cena KGHM w chwili t (do rekonstrukcji \(P_{t+1}\))

---

## Podział danych (train/test)

Podział jest **czasowy** (bez mieszania):
- pierwsze X% próbek → **train**
- reszta → **test**

W projekcie porównywane są:
- split **20/80**
- split **40/60**

To pozwala ocenić:
- jak model radzi sobie przy mniejszej liczbie danych treningowych,
- jak zmienia się jakość predykcji w różnych horyzontach testowych.

---

## Modele i strojenie

### 1) Regresja liniowa (baseline)
Implementacja: `sklearn.linear_model.LinearRegression`.

Cechy wejściowe (X):
- `copper`, `silver`, `wig`

Target (y):
- log-zwrot KGHM.

Rola w projekcie:
- to **punkt odniesienia** (baseline). Jeśli modele neuronowe nie przebijają regresji liniowej, zwykle znaczy to, że:
  - problem jest trudny,
  - cechy są niewystarczające,
  - sieci wymagają innej architektury/cech.

### 2) MLPRegressor (sieć gęsta)
Implementacja: `sklearn.neural_network.MLPRegressor` w pipeline.

Najważniejsze elementy strojenia:
- `hidden_layer_sizes=(64, 32)` – 2 warstwy ukryte
- `activation="relu"`
- `solver="adam"`
- `learning_rate_init=5e-4` – obniżony LR dla stabilności
- `alpha=1e-3` – regularyzacja L2 (zmniejszenie przeuczenia)
- `max_iter=2000`
- `early_stopping=True`, `n_iter_no_change=25` – wczesne zatrzymanie

Skalowanie cech:
- `RobustScaler` (odporny na outliery), poprzedzony imputacją medianą.

Dlaczego takie ustawienia?
- MLP jest wrażliwy na skalowanie i outliery.
- Regularyzacja + mniejszy LR + early stopping ograniczają „odjazdy” predykcji.

### 3) CNN 1D (Conv1D)
Implementacja: TensorFlow/Keras.

Wejście to sekwencje (okna czasowe) o długości `lookback=20` dni:
- kształt: `(N, 20, 3)`
- cechy w czasie: `[copper, silver, wig]`

Architektura (w skrócie):
- `Conv1D(32, kernel_size=3, padding="causal", activation="relu", l2=1e-4)`
- `Conv1D(32, kernel_size=3, padding="causal", activation="relu", l2=1e-4)`
- `GlobalAveragePooling1D()`
- `Dense(64, relu, l2=1e-4)`
- `Dropout(0.2)`
- `Dense(1)` (predykcja log-zwrotu)

Trening:
- optimizer: `Adam(5e-4)`
- loss: `mse`
- `EarlyStopping(patience=15, restore_best_weights=True)`
- `ReduceLROnPlateau(factor=0.5, patience=7, min_lr=1e-5)`
- `epochs=300`, `batch_size=32`

Skalowanie cech:
- `RobustScaler` fitowany tylko na train.

Dodatkowo:
- do predykcji na teście dokładany jest „bridge”: końcówka train (`lookback-1` wierszy), aby pierwsze okna testowe miały kontekst.

---

## Metryki

Modele porównywane są na cenie (po rekonstrukcji z log-zwrotu):
- **MAE** – średni błąd bezwzględny
- **RMSE** – pierwiastek z MSE
- **R²** – współczynnik determinacji

---

## Wykresy / GUI

Po uruchomieniu powstaje okno Matplotlib z 4 panelami (2×2):

- split 20/80:
  - pełny zakres
  - ostatni rok (bez linii treningu)
- split 40/60:
  - pełny zakres
  - ostatni rok (bez linii treningu)

Pliki wyjściowe (PNG):
- `pred_5y_20_80_full.png`
- `pred_5y_20_80_last_year.png`
- `pred_5y_40_60_full.png`
- `pred_5y_40_60_last_year.png`
- `dashboard_5y_splits.png` (cały układ 2×2 w jednym pliku)

---

## Uruchomienie

1) Instalacja zależności:

```bash
pip install -r requirements.txt
```

2) Start aplikacji:

```bash
python app.py --end 2026-01-31
```

Parametr `--end` ustawia datę końcową pobieranych danych.

---

## Struktura plików

- `app.py` – cała logika: pobieranie danych, przygotowanie datasetu, trening modeli, metryki, wizualizacja i zapis wykresów.
- `requirements.txt` – zależności.
- `pred_*.png`, `dashboard_*.png` – wyniki uruchomienia.

---

## Uwagi / ograniczenia

- Projekt nie gwarantuje „sensownych” wyników inwestycyjnych – to demonstrator ML.
- Użycie samych 3 cech (miedź/srebro/WIG) może być niewystarczające.
- Typowe usprawnienia, jeśli chcesz poprawić jakość:
  - dodać cechy opóźnione (lag features),
  - dodać techniczne wskaźniki (SMA/EMA/RSI) dla KGHM i/lub WIG,
  - dodać kurs USDPLN (bo surowce są w USD),
  - przewidywać rozkład (quantile regression) zamiast jednej wartości.

