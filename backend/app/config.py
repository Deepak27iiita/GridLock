from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_PATH = PROJECT_ROOT / "Dataset" / "jan to may police violation_anonymized791b166.csv"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "data" / "models"

VIOLATION_WEIGHTS = {
    "DOUBLE PARKING": 1.0,
    "PARKING IN A MAIN ROAD": 0.95,
    "PARKING NEAR ROAD CROSSING": 0.90,
    "PARKING NEAR TRAFFIC LIGHT OR ZEBRA CROSS": 0.85,
    "NO PARKING": 0.70,
    "WRONG PARKING": 0.65,
    "PARKING OPPOSITE TO ANOTHER PARKED VEHICLE": 0.75,
    "PARKING NEAR BUSTOP/SCHOOL/HOSPITAL ETC": 0.80,
    "PARKING ON FOOTPATH": 0.40,
    "DEFECTIVE NUMBER PLATE": 0.10,
}

VEHICLE_SIZE_FACTOR = {
    "TANKER": 1.25,
    "VAN": 1.15,
    "MAXI-CAB": 1.10,
    "CAR": 1.0,
    "MOTOR CYCLE": 0.85,
    "PASSENGER AUTO": 0.90,
    "GOODS AUTO": 0.95,
    "SCOOTER": 0.70,
    "BUS": 1.30,
}

PCIS_WEIGHTS = {
    "violation_density": 0.28,
    "severity_score": 0.22,
    "junction_proximity": 0.20,
    "temporal_congestion": 0.18,
    "spillover_risk": 0.12,
}

H3_RESOLUTION = 9
BASE_DELAY_PER_VIOLATION = 0.82
