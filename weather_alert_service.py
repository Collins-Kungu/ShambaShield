"""Severe weather alerts for Kenya from LIVE Open-Meteo forecasts (no API key needed).

Edited from: Severe weather alert API.ipynb  (hard-coded alerts replaced by live data)

How it works
    1. Open-Meteo geocoding turns a place name into coordinates (Kenya only).
    2. Open-Meteo forecast gives 7 days of rain, temperature and wind gusts.
    3. Simple threshold rules (see RULES) turn the forecast into alerts, in English or Kiswahili.
    4. Optional: Gemini can answer free-text questions by calling get_severe_alerts() as a tool.

IMPORTANT: alerts are indicative and rule-based. They are NOT official warnings.
Tune RULES with KALRO / Kenya Meteorological Department guidance.

Run (terminal):   python weather_alert_service.py Nakuru Eldoret --lang sw
Run (Streamlit):  streamlit run weather_alert_service.py
Reuse in your app:
    from weather_alert_service import render_weather_tab
    with tab_weather:
        render_weather_tab(lang)          # lang = "en" or "sw"
Dependencies: requests pandas streamlit   (optional: google-genai for the Gemini assistant)
"""

from __future__ import annotations

import argparse
import functools
import os

import pandas as pd
import requests

GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
GEMINI_MODEL = "gemini-2.5-flash-lite"
TIMEOUT_S = 10

# Indicative thresholds - adjust for your region and crop stage.
RULES = {
    "flood_risk_mm": 50,      # daily rain at/above this -> high alert
    "heavy_rain_mm": 25,      # daily rain at/above this -> medium alert
    "strong_gust_kmh": 50,
    "heat_c": 33,             # daily max temperature
    "frost_c": 3,             # daily min temperature (frost risk in highlands)
    "dry_spell_days": 7,      # days with < 1 mm rain
}

MESSAGES = {
    "flood": {"en": "Flood risk: very heavy rain expected on {d} ({v:.0f} mm).",
              "sw": "Hatari ya mafuriko: mvua nyingi sana inatarajiwa {d} ({v:.0f} mm)."},
    "rain":  {"en": "Heavy rain expected on {d} ({v:.0f} mm). Protect harvested grain and check drainage.",
              "sw": "Mvua kubwa inatarajiwa {d} ({v:.0f} mm). Linda nafaka na hakikisha maji yanatiririka."},
    "wind":  {"en": "Strong winds expected on {d} (gusts up to {v:.0f} km/h).",
              "sw": "Upepo mkali unatarajiwa {d} (hadi {v:.0f} km/h)."},
    "heat":  {"en": "Very hot day expected on {d} ({v:.0f} °C).",
              "sw": "Joto kali linatarajiwa {d} ({v:.0f} °C)."},
    "frost": {"en": "Frost risk on {d} (minimum {v:.0f} °C). Protect young plants.",
              "sw": "Hatari ya barafu {d} (joto la chini {v:.0f} °C). Linda mimea michanga."},
    "dry":   {"en": "Dry spell: little or no rain forecast on {v:.0f} of the next 7 days.",
              "sw": "Kipindi cha ukame: mvua kidogo au hakuna katika siku {v:.0f} kati ya 7 zijazo."},
    "ok":    {"en": "No weather alerts for the next 7 days.",
              "sw": "Hakuna tahadhari za hali ya hewa kwa siku 7 zijazo."},
}

LABELS = {
    "en": {"loc": "Location (town or county in Kenya)", "pick": "Select the correct place",
           "noloc": "Location not found in Kenya.", "err": "Weather service unavailable. Please try again later.",
           "rain": "Daily rain (mm)", "temp": "Daily temperature (°C)",
           "note": "Indicative alerts from Open-Meteo forecast data. Always follow official Kenya Met Department warnings.",
           "ask": "Ask the weather assistant (Gemini)", "q": "Your question", "title": "⛈️ Weather alerts"},
    "sw": {"loc": "Mahali (mji au kaunti nchini Kenya)", "pick": "Chagua mahali sahihi",
           "noloc": "Mahali hapajapatikana nchini Kenya.", "err": "Huduma ya hali ya hewa haipatikani. Tafadhali jaribu tena baadaye.",
           "rain": "Mvua kwa siku (mm)", "temp": "Joto kwa siku (°C)",
           "note": "Tahadhari za makadirio kutoka data ya Open-Meteo. Fuata maonyo rasmi ya Idara ya Hali ya Hewa Kenya.",
           "ask": "Uliza msaidizi wa hali ya hewa (Gemini)", "q": "Swali lako", "title": "⛈️ Tahadhari za hali ya hewa"},
}


# --------------------------------------------------------------------------- helpers
def _in_streamlit() -> bool:
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        return get_script_run_ctx(suppress_warning=True) is not None
    except Exception:
        return False


def _cache(func):
    """Cache results for an hour: st.cache_data inside Streamlit, lru_cache elsewhere."""
    if _in_streamlit():
        import streamlit as st
        return st.cache_data(ttl=3600, show_spinner=False)(func)
    return functools.lru_cache(maxsize=64)(func)


# --------------------------------------------------------------------------- data
@_cache
def geocode_location(name: str, count: int = 5) -> list[dict]:
    """Return up to `count` Kenyan places matching `name` (empty list if none)."""
    r = requests.get(GEOCODE_URL, params={"name": name, "count": count, "language": "en",
                                          "countryCode": "KE", "format": "json"}, timeout=TIMEOUT_S)
    r.raise_for_status()
    results = r.json().get("results") or []
    return [p for p in results if p.get("country_code", "KE") == "KE"]


@_cache
def fetch_forecast(latitude: float, longitude: float, days: int = 7) -> pd.DataFrame:
    """Daily forecast: date, rain_mm, tmax, tmin, gust_kmh."""
    r = requests.get(FORECAST_URL, params={
        "latitude": latitude, "longitude": longitude, "timezone": "Africa/Nairobi",
        "forecast_days": days,
        "daily": "precipitation_sum,temperature_2m_max,temperature_2m_min,wind_gusts_10m_max",
    }, timeout=TIMEOUT_S)
    r.raise_for_status()
    df = pd.DataFrame(r.json()["daily"]).rename(columns={
        "time": "date", "precipitation_sum": "rain_mm", "temperature_2m_max": "tmax",
        "temperature_2m_min": "tmin", "wind_gusts_10m_max": "gust_kmh"})
    df["date"] = pd.to_datetime(df["date"])
    return df


def build_alerts(df: pd.DataFrame, lang: str = "en", rules: dict = RULES) -> list[dict]:
    """Turn a forecast into alerts: [{'level': 'high'|'medium'|'info', 'kind': str, 'message': str}]."""
    alerts: list[dict] = []

    def add(level, kind, day, value):
        alerts.append({"level": level, "kind": kind,
                       "message": MESSAGES[kind][lang].format(d=day, v=value)})

    for row in df.itertuples():
        day = row.date.strftime("%a %d %b")
        if row.rain_mm >= rules["flood_risk_mm"]:
            add("high", "flood", day, row.rain_mm)
        elif row.rain_mm >= rules["heavy_rain_mm"]:
            add("medium", "rain", day, row.rain_mm)
        if row.gust_kmh >= rules["strong_gust_kmh"]:
            add("medium", "wind", day, row.gust_kmh)
        if row.tmax >= rules["heat_c"]:
            add("medium", "heat", day, row.tmax)
        if row.tmin <= rules["frost_c"]:
            add("medium", "frost", day, row.tmin)

    dry_days = int((df["rain_mm"].fillna(0) < 1).sum())
    if dry_days >= rules["dry_spell_days"]:
        add("medium", "dry", "", dry_days)

    return alerts or [{"level": "info", "kind": "ok", "message": MESSAGES["ok"][lang].format(d="", v=0)}]


# --------------------------------------------------------------------------- Gemini tool (optional)
def get_severe_alerts(location: str) -> str:
    """Fetch live severe weather alerts for the next 7 days for a place in Kenya (Open-Meteo forecast)."""
    try:
        places = geocode_location(location)
        if not places:
            return f"Location '{location}' was not found in Kenya."
        place = places[0]
        df = fetch_forecast(place["latitude"], place["longitude"])
    except requests.RequestException as exc:
        return f"Weather service unavailable: {exc}"
    return f"{place['name']}: " + " ".join(a["message"] for a in build_alerts(df, "en"))


def _get_google_key() -> str | None:
    key = os.environ.get("GOOGLE_API_KEY")
    if key:
        return key
    try:
        import streamlit as st
        return st.secrets.get("GOOGLE_API_KEY")
    except Exception:
        return None


def run_weather_assistant(user_query: str) -> str:
    """Let Gemini answer a question, calling get_severe_alerts() (live data) when it needs to."""
    api_key = _get_google_key()
    if not api_key:
        return "Error: GOOGLE_API_KEY is not set (environment variable or Streamlit secrets)."
    try:
        from google import genai
    except ImportError:
        return "Error: 'google-genai' is not installed (pip install google-genai)."
    try:
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model=GEMINI_MODEL, contents=user_query, config={"tools": [get_severe_alerts]})
        return response.text
    except Exception as exc:
        return f"An error occurred while calling the Gemini API: {exc}"


# --------------------------------------------------------------------------- Streamlit UI
def render_weather_tab(lang: str = "en", key_prefix: str = "wx") -> None:
    """Draw the weather-alert UI. Call inside a tab/page of your Streamlit app."""
    import streamlit as st
    t = LABELS[lang]

    place_name = st.text_input(t["loc"], "Nakuru", key=f"{key_prefix}_place").strip()
    if not place_name:
        return
    try:
        places = geocode_location(place_name)
    except requests.RequestException:
        st.error(t["err"])
        return
    if not places:
        st.warning(t["noloc"])
        return

    names = [f"{p['name']}, {p.get('admin1', '')}".rstrip(", ") for p in places]
    idx = (st.selectbox(t["pick"], range(len(places)), format_func=lambda i: names[i],
                        key=f"{key_prefix}_pick") if len(places) > 1 else 0)
    try:
        df = fetch_forecast(places[idx]["latitude"], places[idx]["longitude"])
    except requests.RequestException:
        st.error(t["err"])
        return

    show = {"high": st.error, "medium": st.warning, "info": st.success}
    for alert in build_alerts(df, lang):
        show[alert["level"]](alert["message"])

    st.subheader(t["rain"])
    st.bar_chart(df.set_index("date")["rain_mm"])
    st.subheader(t["temp"])
    st.line_chart(df.set_index("date")[["tmax", "tmin"]])
    st.caption(t["note"] + " · Data: Open-Meteo.com")

    if _get_google_key():
        with st.expander(t["ask"]):
            q = st.text_input(t["q"], key=f"{key_prefix}_q")
            if q:
                st.write(run_weather_assistant(q))


def _streamlit_entry() -> None:
    import streamlit as st
    st.set_page_config(page_title="Weather alerts", page_icon="⛈️")
    lang = st.sidebar.radio("Language / Lugha", ["en", "sw"],
                            format_func={"en": "English", "sw": "Kiswahili"}.get)
    st.title(LABELS[lang]["title"])
    render_weather_tab(lang)


# --------------------------------------------------------------------------- CLI
def main() -> None:
    parser = argparse.ArgumentParser(description="Live severe-weather alerts for Kenyan places.")
    parser.add_argument("places", nargs="*", default=["Nairobi", "Nakuru", "Mombasa"])
    parser.add_argument("--lang", choices=["en", "sw"], default="en")
    args = parser.parse_args()

    for name in args.places:
        print(f"\n=== {name} ===")
        try:
            places = geocode_location(name)
            if not places:
                print("Location not found in Kenya.")
                continue
            df = fetch_forecast(places[0]["latitude"], places[0]["longitude"])
        except requests.RequestException as exc:
            print(f"Weather service unavailable: {exc}")
            continue
        for a in build_alerts(df, args.lang):
            print(f"[{a['level'].upper()}] {a['message']}")


if _in_streamlit():
    _streamlit_entry()
elif __name__ == "__main__":
    main()
