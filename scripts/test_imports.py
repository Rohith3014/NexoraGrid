import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src import (
    config, utils, data_loader, data_validation, sensor_fusion,
    waste_classifier, forecasting, anomaly_detection, root_cause,
    recommendations, circular_accounting, incentives, database
)

print("All modules imported successfully!")
