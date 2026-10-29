"""Farmer credit scoring (climate-adjusted).

Converted from: Farmer credit score.ipynb
Run:  python credit_score.py
Import: from credit_score import calculate_credit_score
"""

# Expected input ranges used for 0-1 normalisation
NDVI_RANGE = (0.40, 0.85)
FARM_SIZE_MAX_HA = 10.0
SOIL_MOISTURE_RANGE = (15.0, 45.0)

# Weights (sum to 1.0)
WEIGHT_NDVI = 0.50
WEIGHT_SOIL = 0.30
WEIGHT_FARM = 0.20

SCORE_THRESHOLD = 50.0

RISK_MULTIPLIERS = {"Low": 1.0, "Medium": 0.8, "High": 0.5}


def _normalise(value: float, low: float, high: float) -> float:
    """Scale value to 0-1 and clamp so out-of-range inputs cannot inflate the score."""
    return min(max((value - low) / (high - low), 0.0), 1.0)


def calculate_credit_score(
    ndvi_value: float,
    farm_size_ha: float,
    soil_moisture_pct: float,
    climate_risk_zone: str,
    threshold: float = SCORE_THRESHOLD,
) -> dict:
    """Return base score, climate multiplier, final score and waitlist decision."""
    if climate_risk_zone not in RISK_MULTIPLIERS:
        raise ValueError(
            f"climate_risk_zone must be one of {list(RISK_MULTIPLIERS)}, got {climate_risk_zone!r}"
        )

    ndvi_norm = _normalise(ndvi_value, *NDVI_RANGE)
    soil_norm = _normalise(soil_moisture_pct, *SOIL_MOISTURE_RANGE)
    farm_norm = _normalise(farm_size_ha, 0.0, FARM_SIZE_MAX_HA)

    base_score = (
        ndvi_norm * WEIGHT_NDVI + soil_norm * WEIGHT_SOIL + farm_norm * WEIGHT_FARM
    ) * 100
    multiplier = RISK_MULTIPLIERS[climate_risk_zone]
    final_score = round(base_score * multiplier, 1)

    return {
        "base_score": round(base_score, 1),
        "multiplier": multiplier,
        "final_score": final_score,
        "threshold": threshold,
        "shortlisted": final_score >= threshold,
    }


def print_report(farmer_id: str, climate_risk_zone: str, result: dict) -> None:
    line = "=" * 50
    print(line)
    print(f"CREDIT SCORE EVALUATION: {farmer_id}")
    print(line)
    print(f"Base Weighted Score: {result['base_score']} / 100")
    print(f"Climate Risk Penalty Multiplier ({climate_risk_zone}): {result['multiplier']}x")
    print(f"Final Climate-Adjusted Score: {result['final_score']} / 100")
    print(f"Minimum Score Required: {result['threshold']}")
    print("-" * 50)
    if result["shortlisted"]:
        print("STATUS: APPROVED -> ADDED TO LOAN WAITLIST")
    else:
        print("STATUS: REJECTED -> DOES NOT MEET CRITERIA")
    print(line)


def main() -> None:
    farmer = {
        "farmer_id": "FARM_011",
        "ndvi_value": 0.72,
        "farm_size_ha": 5.5,
        "soil_moisture_pct": 30.0,
        "climate_risk_zone": "Medium",
    }
    result = calculate_credit_score(
        farmer["ndvi_value"],
        farmer["farm_size_ha"],
        farmer["soil_moisture_pct"],
        farmer["climate_risk_zone"],
    )
    print_report(farmer["farmer_id"], farmer["climate_risk_zone"], result)


if __name__ == "__main__":
    main()
