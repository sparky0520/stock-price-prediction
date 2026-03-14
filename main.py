import asyncio
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone

import ccxt.pro as ccxtpro
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier


MODEL_PATH = "models/catboost_classifier.cbm"
SYMBOL = "BTC/USDT"
TIMEFRAME = "1m"
FEE_RATE = 0.0004
TRADE_NOTIONAL_USDT = 1_000.0

# Feature order mirrors the processed crypto training dataset.
FEATURE_COLUMNS = [
    "Open",
    "High",
    "Low",
    "Close",
    "Volume",
    "SMA_9",
    "SMA_21",
    "SMA_50",
    "EMA_9",
    "RSI_14",
    "MACD",
    "MACD_Signal",
    "BB_Upper",
    "BB_Lower",
    "ATR_14",
    "Hour",
    "Minute",
    "DayOfWeek",
]


def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    close = df["Close"]
    high = df["High"]
    low = df["Low"]

    df["SMA_9"] = close.rolling(window=9).mean()
    df["SMA_21"] = close.rolling(window=21).mean()
    df["SMA_50"] = close.rolling(window=50).mean()
    df["EMA_9"] = close.ewm(span=9, adjust=False).mean()

    delta = close.diff(1)
    gain = delta.where(delta > 0, 0.0).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(window=14).mean()
    rs = gain / loss.replace(0, np.nan)
    df["RSI_14"] = 100 - (100 / (1 + rs))

    exp1 = close.ewm(span=12, adjust=False).mean()
    exp2 = close.ewm(span=26, adjust=False).mean()
    df["MACD"] = exp1 - exp2
    df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()

    rolling_mean = close.rolling(window=20).mean()
    rolling_std = close.rolling(window=20).std()
    df["BB_Upper"] = rolling_mean + (2 * rolling_std)
    df["BB_Lower"] = rolling_mean - (2 * rolling_std)

    high_low = high - low
    high_close = (high - close.shift()).abs()
    low_close = (low - close.shift()).abs()
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df["ATR_14"] = true_range.rolling(window=14).mean()

    dt_index = df.index
    df["Hour"] = dt_index.hour
    df["Minute"] = dt_index.minute
    df["DayOfWeek"] = dt_index.dayofweek
    return df


@dataclass
class Position:
    side: str
    qty: float
    entry_price: float


class PaperTrader:
    def __init__(self, starting_cash: float = 10_000.0):
        self.cash = starting_cash
        self.position: Position | None = None
        self.realized_pnl = 0.0

    def _close_position(self, price: float) -> float:
        if self.position is None:
            return 0.0

        pos = self.position
        notional = pos.qty * price
        fee = notional * FEE_RATE

        if pos.side == "long":
            pnl = (price - pos.entry_price) * pos.qty - fee
            self.cash += notional - fee
        else:
            pnl = (pos.entry_price - price) * pos.qty - fee
            self.cash -= notional + fee

        self.realized_pnl += pnl
        self.position = None
        return pnl

    def _open_position(self, side: str, price: float) -> None:
        qty = TRADE_NOTIONAL_USDT / price
        notional = qty * price
        fee = notional * FEE_RATE

        if side == "long":
            self.cash -= notional + fee
        else:
            self.cash += notional - fee

        self.position = Position(side=side, qty=qty, entry_price=price)

    def on_signal(self, signal: str, price: float) -> str:
        action = "HOLD"

        if signal == "Buy":
            if self.position and self.position.side == "short":
                pnl = self._close_position(price)
                action = f"CLOSE_SHORT pnl={pnl:,.2f}"
            if self.position is None:
                self._open_position("long", price)
                action = (action + " | " if action != "HOLD" else "") + "OPEN_LONG"

        elif signal == "Sell":
            if self.position and self.position.side == "long":
                pnl = self._close_position(price)
                action = f"CLOSE_LONG pnl={pnl:,.2f}"
            if self.position is None:
                self._open_position("short", price)
                action = (action + " | " if action != "HOLD" else "") + "OPEN_SHORT"

        return action

    def equity(self, mark_price: float) -> float:
        if self.position is None:
            return self.cash

        pos = self.position
        unrealized = (mark_price - pos.entry_price) * pos.qty
        if pos.side == "short":
            unrealized = -unrealized
        return self.cash + unrealized


def parse_ticker_time(ticker: dict) -> datetime:
    if ticker.get("timestamp"):
        return datetime.fromtimestamp(ticker["timestamp"] / 1000, tz=timezone.utc)
    if ticker.get("datetime"):
        return datetime.fromisoformat(ticker["datetime"].replace("Z", "+00:00"))
    return datetime.now(timezone.utc)


def build_feature_row(candles: deque[dict]) -> np.ndarray | None:
    df = pd.DataFrame(list(candles)).set_index("Datetime")
    df = calculate_indicators(df)
    latest = df.iloc[-1]

    if latest[FEATURE_COLUMNS].isna().any():
        return None

    return latest[FEATURE_COLUMNS].to_numpy(dtype=float)


async def preload_recent_candles(
    exchange: ccxtpro.Exchange, symbol: str, timeframe: str, target: int = 50
) -> list[dict]:
    # Request one extra bar and drop the newest one to avoid using an in-progress minute.
    ohlcv = await exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=target + 1)
    if not ohlcv:
        return []

    closed_bars = ohlcv[:-1] if len(ohlcv) > target else ohlcv
    recent = closed_bars[-target:]

    return [
        {
            "Datetime": datetime.fromtimestamp(ts / 1000, tz=timezone.utc),
            "Open": float(open_price),
            "High": float(high_price),
            "Low": float(low_price),
            "Close": float(close_price),
            "Volume": float(volume),
        }
        for ts, open_price, high_price, low_price, close_price, volume in recent
    ]


async def stream_and_simulate() -> None:
    exchange = ccxtpro.binance()
    model = CatBoostClassifier()
    model.load_model(MODEL_PATH)
    trader = PaperTrader()

    candles: deque[dict] = deque(maxlen=200)
    current_candle: dict | None = None
    last_base_volume: float | None = None

    historical_candles = await preload_recent_candles(exchange, SYMBOL, TIMEFRAME, target=50)
    candles.extend(historical_candles)

    print(f"Loaded model from {MODEL_PATH}")
    print(f"Preloaded {len(historical_candles)} recent {TIMEFRAME} candles")
    print(f"Streaming {SYMBOL} ticker via websocket...")

    try:
        while True:
            ticker = await exchange.watch_ticker(SYMBOL)
            price = float(ticker["last"])
            tick_time = parse_ticker_time(ticker)
            minute_ts = tick_time.replace(second=0, microsecond=0)
            base_volume_total = ticker.get("baseVolume")

            if current_candle is None or current_candle["Datetime"] != minute_ts:
                if current_candle is not None:
                    candles.append(current_candle)

                current_candle = {
                    "Datetime": minute_ts,
                    "Open": price,
                    "High": price,
                    "Low": price,
                    "Close": price,
                    "Volume": 0.0,
                }
            else:
                current_candle["High"] = max(current_candle["High"], price)
                current_candle["Low"] = min(current_candle["Low"], price)
                current_candle["Close"] = price

            if base_volume_total is not None:
                base_volume_total = float(base_volume_total)
                if last_base_volume is not None:
                    delta = max(base_volume_total - last_base_volume, 0.0)
                    current_candle["Volume"] += delta
                last_base_volume = base_volume_total

            feature_row = None
            if len(candles) >= 50:
                temp = deque(candles, maxlen=200)
                temp.append(current_candle)
                feature_row = build_feature_row(temp)

            signal = "WARMUP"
            action = "-"
            if feature_row is not None:
                prediction = model.predict(feature_row.reshape(1, -1))
                signal = str(prediction[0][0])
                action = trader.on_signal(signal, price)

            equity = trader.equity(price)
            print(
                f"{tick_time.isoformat()}  ${price:,.2f}  "
                f"signal={signal:<6} action={action:<28} "
                f"equity=${equity:,.2f} realized=${trader.realized_pnl:,.2f}"
            )

    finally:
        await exchange.close()


def main() -> None:
    asyncio.run(stream_and_simulate())


if __name__ == "__main__":
    main()
