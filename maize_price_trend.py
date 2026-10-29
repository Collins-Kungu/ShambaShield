"""Maize price trend for Kenya from LIVE WFP price data (via the Humanitarian Data Exchange).

Edited from: Maize price trend.ipynb  (simulated prices replaced by real WFP data)

Data: "Kenya - Food Prices" (WFP), published on HDX, updated roughly monthly.
      License: CC BY-IGO. Please credit "WFP, via HDX".
How it works
    1. Asks the HDX API for the dataset's current CSV link and downloads it
       (cached in data/wfp_food_prices_ken.csv, refreshed after 24 h; if the
       download fails the last saved copy is used).
    2. Keeps maize commodities, averages prices per month (optionally per market).
    3. Adds a 3-month moving average and a simple up / down / stable summary.

Notes: WFP data is MONTHLY and lags real time by weeks to months; prices are usually
KES per KG. "Per 90 kg bag" below is a conversion of that KG price, not a separate series.

Run (terminal):   python maize_price_trend.py
Run (Streamlit):  streamlit run maize_price_trend.py
Reuse in your app:
    from maize_price_trend import render_price_tab
    with tab_prices:
        render_price_tab(lang)            # lang = "en" or "sw"
Dependencies: requests pandas matplotlib streamlit
"""

from __future__ import annotations

import functools
import re
import time
from datetime import date
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import requests

HDX_PACKAGE_URL = "https://data.humdata.org/api/3/action/package_show"
HDX_DATASET_ID = "wfp-food-prices-for-kenya"
LOCAL_CSV = Path("data/wfp_food_prices_ken.csv")
MAX_AGE_HOURS = 24          # re-download when the local copy is older than this
STALE_AFTER_DAYS = 120      # warn when the newest data point is older than this
ALL_MARKETS = "All markets"

LABELS = {
    "en": {"title": "🌽 Maize price trend", "commodity": "Commodity", "market": "Market",
           "ptype": "Price type", "bag": "Show price per 90 kg bag", "all": ALL_MARKETS,
           "latest": "Latest monthly average", "up": "Rising", "down": "Falling", "flat": "Stable",
           "vs": "Last 3 months vs the 3 months before", "short": "Not enough data for a trend.",
           "nodata": "No data for this selection.", "stale": "The newest data point is {d}. WFP data is published with a delay.",
           "err": "Could not download price data and no saved copy exists. Download the CSV manually from HDX and save it as data/wfp_food_prices_ken.csv.",
           "src": "Source: WFP Kenya food prices via HDX (CC BY-IGO)."},
    "sw": {"title": "🌽 Mwelekeo wa bei ya mahindi", "commodity": "Bidhaa", "market": "Soko",
           "ptype": "Aina ya bei", "bag": "Onyesha bei kwa gunia la kilo 90", "all": "Masoko yote",
           "latest": "Wastani wa mwezi wa hivi karibuni", "up": "Inapanda", "down": "Inashuka", "flat": "Imetulia",
           "vs": "Miezi 3 iliyopita ikilinganishwa na miezi 3 kabla yake", "short": "Data haitoshi kuonyesha mwelekeo.",
           "nodata": "Hakuna data kwa chaguo hili.", "stale": "Data ya hivi karibuni zaidi ni {d}. Data ya WFP huchapishwa kwa kuchelewa.",
           "err": "Imeshindwa kupakua data ya bei na hakuna nakala iliyohifadhiwa. Pakua CSV kutoka HDX na uihifadhi kama data/wfp_food_prices_ken.csv.",
           "src": "Chanzo: bei za chakula Kenya (WFP) kupitia HDX (CC BY-IGO)."},
}


# --------------------------------------------------------------------------- helpers
def _in_streamlit() -> bool:
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        return get_script_run_ctx(suppress_warning=True) is not None
    except Exception:
        return False


def _cache(func):
    if _in_streamlit():
        import streamlit as st
        return st.cache_data(show_spinner=False)(func)
    return functools.lru_cache(maxsize=8)(func)


def _unit_kg(unit) -> float:
    """'KG' -> 1.0, '90 KG' -> 90.0, anything else -> NaN."""
    m = re.fullmatch(r"\s*(\d+(?:\.\d+)?)?\s*kg\s*", str(unit), flags=re.I)
    return float(m.group(1) or 1) if m else float("nan")


# --------------------------------------------------------------------------- download + load
def resolve_csv_url() -> str:
    """Ask the HDX API for the current download link of the main Kenya prices CSV."""
    r = requests.get(HDX_PACKAGE_URL, params={"id": HDX_DATASET_ID}, timeout=20)
    r.raise_for_status()
    resources = r.json()["result"]["resources"]
    csvs = [x for x in resources
            if str(x.get("format", "")).lower() == "csv"
            and "quickcharts" not in str(x.get("name", "")).lower()
            and "_qc" not in str(x.get("url", "")).lower()]
    if not csvs:
        raise RuntimeError("No CSV resource found in the HDX dataset.")
    csvs.sort(key=lambda x: "wfp_food_prices" not in str(x.get("url", "")).lower())
    return csvs[0]["url"]


def download_prices(force: bool = False) -> Path:
    """Return the path of a local copy of the WFP CSV, refreshing it when it is old."""
    fresh = LOCAL_CSV.exists() and (time.time() - LOCAL_CSV.stat().st_mtime) < MAX_AGE_HOURS * 3600
    if fresh and not force:
        return LOCAL_CSV
    try:
        r = requests.get(resolve_csv_url(), timeout=60)
        r.raise_for_status()
        if not r.content.lstrip(b"\xef\xbb\xbf")[:4].lower() == b"date":
            raise ValueError("Downloaded file does not look like the WFP CSV.")
        LOCAL_CSV.parent.mkdir(parents=True, exist_ok=True)
        tmp = LOCAL_CSV.with_suffix(".tmp")
        tmp.write_bytes(r.content)
        tmp.replace(LOCAL_CSV)
    except (requests.RequestException, RuntimeError, KeyError, ValueError):
        if LOCAL_CSV.exists():          # an older saved copy beats no data
            return LOCAL_CSV
        raise
    return LOCAL_CSV


@_cache
def _read_prices(path: str, mtime: float) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str, encoding="utf-8-sig")
    df = df[~df["date"].str.startswith("#", na=False)].copy()        # drop the HXL tag row
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df = df.dropna(subset=["date", "price"])
    return df[df["commodity"].str.contains("maize", case=False, na=False)].reset_index(drop=True)


def load_wfp_maize_prices(force_download: bool = False) -> pd.DataFrame:
    """Download (if needed) and load all maize price records for Kenya."""
    path = download_prices(force=force_download)
    return _read_prices(str(path), path.stat().st_mtime)


# --------------------------------------------------------------------------- analysis
def default_commodity(df: pd.DataFrame) -> str:
    names = df["commodity"].value_counts()
    for preferred in ("Maize (white)", "Maize"):
        if preferred in names.index:
            return preferred
    return names.index[0]


def fetch_maize_market_data(df: pd.DataFrame | None = None, commodity: str | None = None,
                            market: str = ALL_MARKETS, pricetype: str = "Retail",
                            per_90kg: bool = False) -> pd.DataFrame:
    """Monthly average price. Columns: Date, Price, Unit, Currency (empty frame if no match)."""
    df = load_wfp_maize_prices() if df is None else df
    d = df[(df["commodity"] == (commodity or default_commodity(df))) & (df["pricetype"] == pricetype)]
    if market != ALL_MARKETS:
        d = d[d["market"] == market]
    if d.empty:
        return pd.DataFrame(columns=["Date", "Price", "Unit", "Currency"])

    price, unit = d["price"], d["unit"].mode().iat[0]
    if per_90kg:
        kg = d["unit"].map(_unit_kg)
        if kg.notna().all():
            price, unit = d["price"] / kg * 90, "90 KG"
    monthly = (d.assign(Price=price).groupby(pd.Grouper(key="date", freq="MS"))["Price"].mean().dropna())
    out = monthly.reset_index().rename(columns={"date": "Date"})
    out["Unit"], out["Currency"] = unit, d["currency"].mode().iat[0]
    return out


def calculate_price_trends(df: pd.DataFrame, window: int = 3) -> pd.DataFrame:
    """Add a rolling-mean column named '<window>_Month_Trend'."""
    out = df.copy()
    out[f"{window}_Month_Trend"] = out["Price"].rolling(window=window, min_periods=1).mean()
    return out


def summarise_trend(df: pd.DataFrame, flat_band_pct: float = 3.0) -> dict | None:
    """Compare the mean of the last 3 months with the 3 before. Needs >= 6 months of data."""
    if len(df) < 6:
        return None
    last3, prev3 = df["Price"].iloc[-3:].mean(), df["Price"].iloc[-6:-3].mean()
    pct = (last3 / prev3 - 1) * 100
    direction = "up" if pct > flat_band_pct else "down" if pct < -flat_band_pct else "flat"
    latest = df.iloc[-1]
    return {"direction": direction, "pct_change": pct, "latest_price": latest["Price"],
            "latest_date": latest["Date"], "days_old": (date.today() - latest["Date"].date()).days}


def generate_trend_plot(df: pd.DataFrame, window: int = 3):
    """Return a Matplotlib Figure (use st.pyplot(fig) in Streamlit)."""
    col = f"{window}_Month_Trend"
    if col not in df.columns:
        df = calculate_price_trends(df, window)
    cur, unit = df["Currency"].iat[0], df["Unit"].iat[0]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df["Date"], df["Price"], label="Monthly average price",
            color="#4caf50", alpha=0.7, marker="o", markersize=3)
    ax.plot(df["Date"], df[col], label=f"{window}-month trend", color="#e91e63", linewidth=2.5)
    ax.set_title("Maize price trend in Kenya (WFP market data)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Month")
    ax.set_ylabel(f"Price ({cur} per {unit})")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(fontsize=9)
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- Streamlit UI
def render_price_tab(lang: str = "en", key_prefix: str = "px") -> None:
    """Draw the maize price UI. Call inside a tab/page of your Streamlit app."""
    import streamlit as st
    t = LABELS[lang]

    try:
        with st.spinner("..."):
            df = load_wfp_maize_prices()
    except Exception:
        st.error(t["err"])
        return

    c1, c2, c3 = st.columns(3)
    names = sorted(df["commodity"].unique())
    commodity = c1.selectbox(t["commodity"], names, index=names.index(default_commodity(df)),
                             key=f"{key_prefix}_c")
    sub = df[df["commodity"] == commodity]
    ptypes = sorted(sub["pricetype"].dropna().unique())
    ptype = c3.selectbox(t["ptype"], ptypes, index=ptypes.index("Retail") if "Retail" in ptypes else 0,
                         key=f"{key_prefix}_t")
    markets = [t["all"]] + sorted(sub["market"].dropna().unique())
    market = c2.selectbox(t["market"], markets, key=f"{key_prefix}_m")
    per_bag = st.checkbox(t["bag"], key=f"{key_prefix}_bag")

    data = fetch_maize_market_data(df, commodity, ALL_MARKETS if market == t["all"] else market,
                                   ptype, per_bag)
    if data.empty:
        st.warning(t["nodata"])
        return

    data = calculate_price_trends(data, 3)
    cur, unit = data["Currency"].iat[0], data["Unit"].iat[0]
    st.line_chart(data.set_index("Date")[["Price", "3_Month_Trend"]])
    st.caption(f"{cur} / {unit}")

    summary = summarise_trend(data)
    if summary is None:
        st.info(t["short"])
    else:
        m1, m2 = st.columns(2)
        m1.metric(f"{t['latest']} ({summary['latest_date']:%b %Y})", f"{summary['latest_price']:,.0f} {cur}/{unit}")
        m2.metric(t[summary["direction"]], f"{summary['pct_change']:+.1f}%", help=t["vs"])
        if summary["days_old"] > STALE_AFTER_DAYS:
            st.warning(t["stale"].format(d=f"{summary['latest_date']:%b %Y}"))
    st.caption(t["src"])


def _streamlit_entry() -> None:
    import streamlit as st
    st.set_page_config(page_title="Maize prices", page_icon="🌽")
    lang = st.sidebar.radio("Language / Lugha", ["en", "sw"],
                            format_func={"en": "English", "sw": "Kiswahili"}.get)
    st.title(LABELS[lang]["title"])
    render_price_tab(lang)


# --------------------------------------------------------------------------- CLI
def main() -> None:
    df = load_wfp_maize_prices()
    data = calculate_price_trends(fetch_maize_market_data(df), window=3)
    print(data.tail())
    print("Trend:", summarise_trend(data))
    generate_trend_plot(data).savefig("maize_price_trend.png", dpi=150)
    print("Saved maize_price_trend.png")


if _in_streamlit():
    _streamlit_entry()
elif __name__ == "__main__":
    main()
