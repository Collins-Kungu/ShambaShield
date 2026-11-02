"""Farmer credit scoring for the loan waitlist (DEMONSTRATION logic, not a validated credit model).

Two scoring methods live here, because the notebook and the web app used different formulas:

  "bands"    (default) - the formula your Streamlit app has been showing:
                         soil moisture 30 pts (banded) + NDVI 40 pts (ndvi x 40) + farm size 15 pts (banded)
                         + climate risk 15 pts (Low 15 / Medium 8 / High 2). Approved at 55 points or more.
  "weighted"           - the formula from Farmer credit score.ipynb:
                         50% NDVI + 30% soil moisture + 20% farm size (each normalised), multiplied by a
                         climate-risk factor (Low 1.0 / Medium 0.8 / High 0.5). Approved at 50 or more.

Pick the method with evaluate_farmer(..., method="weighted") or the CREDIT_SCORING_METHOD environment
variable. Both the Streamlit app and the FastAPI backend call evaluate_farmer(), so they always agree.

Run:     python credit_score.py
Import:  from credit_score import evaluate_farmer
Please note: a real lending decision needs validated data, fairness testing and human review.
"""

from __future__ import annotations

import os

RISK_ZONES = ("Low", "Medium", "High")
DEFAULT_METHOD = "bands"

# ------------------------------------------------------------------ method 1: "bands" (web app)
BANDS_THRESHOLD = 55.0
BANDS_MAX = {"Soil moisture": 30, "NDVI": 40, "Farm size": 15, "Climate risk": 15}
_RISK_POINTS = {"Low": 15.0, "Medium": 8.0, "High": 2.0}

# ------------------------------------------------------------------ method 2: "weighted" (notebook)
NDVI_RANGE = (0.40, 0.85)
FARM_SIZE_MAX_HA = 10.0
SOIL_MOISTURE_RANGE = (15.0, 45.0)
WEIGHT_NDVI, WEIGHT_SOIL, WEIGHT_FARM = 0.50, 0.30, 0.20
SCORE_THRESHOLD = 50.0
RISK_MULTIPLIERS = {"Low": 1.0, "Medium": 0.8, "High": 0.5}


def _check(ndvi: float, farm_size_ha: float, soil_moisture_pct: float, zone: str) -> None:
    if zone not in RISK_ZONES:
        raise ValueError(f"climate_risk_zone must be one of {list(RISK_ZONES)}, got {zone!r}")
    if not 0.0 <= ndvi <= 1.0:
        raise ValueError("ndvi must be between 0 and 1")
    if not 0.0 <= soil_moisture_pct <= 100.0:
        raise ValueError("soil_moisture_pct must be between 0 and 100")
    if farm_size_ha <= 0:
        raise ValueError("farm_size_ha must be greater than 0")


def _normalise(value: float, low: float, high: float) -> float:
    """Scale to 0-1 and clamp so out-of-range inputs cannot inflate the score."""
    return min(max((value - low) / (high - low), 0.0), 1.0)


def calculate_credit_score(ndvi_value: float, farm_size_ha: float, soil_moisture_pct: float,
                           climate_risk_zone: str, threshold: float = SCORE_THRESHOLD) -> dict:
    """The notebook's weighted formula. Returns base score, multiplier, final score, decision."""
    if climate_risk_zone not in RISK_MULTIPLIERS:
        raise ValueError(f"climate_risk_zone must be one of {list(RISK_MULTIPLIERS)}, got {climate_risk_zone!r}")
    ndvi_norm = _normalise(ndvi_value, *NDVI_RANGE)
    soil_norm = _normalise(soil_moisture_pct, *SOIL_MOISTURE_RANGE)
    farm_norm = _normalise(farm_size_ha, 0.0, FARM_SIZE_MAX_HA)
    base = (ndvi_norm * WEIGHT_NDVI + soil_norm * WEIGHT_SOIL + farm_norm * WEIGHT_FARM) * 100
    multiplier = RISK_MULTIPLIERS[climate_risk_zone]
    final = round(base * multiplier, 1)
    return {"base_score": round(base, 1), "multiplier": multiplier, "final_score": final,
            "threshold": threshold, "shortlisted": final >= threshold,
            "parts": {"ndvi": ndvi_norm * WEIGHT_NDVI * 100, "soil": soil_norm * WEIGHT_SOIL * 100,
                      "farm": farm_norm * WEIGHT_FARM * 100}}


def _bands(ndvi: float, farm_size_ha: float, soil_moisture_pct: float, zone: str) -> dict:
    m = soil_moisture_pct
    moisture = 30.0 if 25.0 <= m <= 60.0 else 15.0 if (15.0 <= m < 25.0 or 60.0 < m <= 80.0) else 5.0
    ndvi_pts = ndvi * 40.0
    size = 15.0 if farm_size_ha >= 5.0 else 10.0 if farm_size_ha >= 1.5 else 5.0
    risk = _RISK_POINTS[zone]
    total = moisture + ndvi_pts + size + risk
    rows = [
        ("Soil moisture", "30%", f"{soil_moisture_pct}%", moisture, 30),
        ("NDVI", "40%", f"{ndvi}", ndvi_pts, 40),
        ("Farm size", "15%", f"{farm_size_ha} ha", size, 15),
        ("Climate risk", "15%", zone, risk, 15),
    ]
    return {"method": "bands", "score": round(total, 1), "threshold": BANDS_THRESHOLD,
            "approved": total >= BANDS_THRESHOLD,
            "breakdown": [{"parameter": p, "weight": w, "input": i, "points": round(pts, 1), "max_points": mx}
                          for p, w, i, pts, mx in rows]}


def _weighted(ndvi: float, farm_size_ha: float, soil_moisture_pct: float, zone: str) -> dict:
    r = calculate_credit_score(ndvi, farm_size_ha, soil_moisture_pct, zone)
    rows = [
        ("NDVI", "50%", f"{ndvi}", r["parts"]["ndvi"], 50),
        ("Soil moisture", "30%", f"{soil_moisture_pct}%", r["parts"]["soil"], 30),
        ("Farm size", "20%", f"{farm_size_ha} ha", r["parts"]["farm"], 20),
    ]
    breakdown = [{"parameter": p, "weight": w, "input": i, "points": round(pts, 1), "max_points": mx}
                 for p, w, i, pts, mx in rows]
    breakdown.append({"parameter": "Climate risk multiplier", "weight": "x", "input": zone,
                      "points": r["multiplier"], "max_points": 1.0})
    return {"method": "weighted", "score": r["final_score"], "threshold": r["threshold"],
            "approved": r["shortlisted"], "breakdown": breakdown}


def evaluate_farmer(soil_moisture_pct: float, ndvi: float, farm_size_ha: float,
                    climate_risk_zone: str, method: str | None = None) -> dict:
    """Score a farmer. Returns {'method','score','threshold','approved','breakdown':[...]}."""
    method = (method or os.environ.get("CREDIT_SCORING_METHOD") or DEFAULT_METHOD).lower()
    if method not in {"bands", "weighted"}:
        raise ValueError("method must be 'bands' or 'weighted'")
    _check(ndvi, farm_size_ha, soil_moisture_pct, climate_risk_zone)
    fn = _bands if method == "bands" else _weighted
    return fn(ndvi, farm_size_ha, soil_moisture_pct, climate_risk_zone)


def main() -> None:
    for method in ("bands", "weighted"):
        r = evaluate_farmer(soil_moisture_pct=35.0, ndvi=0.65, farm_size_ha=2.5,
                            climate_risk_zone="Medium", method=method)
        print(f"[{method}] score {r['score']} / threshold {r['threshold']} -> "
              f"{'APPROVED' if r['approved'] else 'ON HOLD'}")
        for row in r["breakdown"]:
            print(f"   {row['parameter']:<24} {row['input']:>8}  {row['points']} / {row['max_points']}")


if __name__ == "__main__":
    main()
