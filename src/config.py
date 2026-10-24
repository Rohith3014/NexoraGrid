"""
EcoGrid AI - Configuration Module
Loads and provides access to all configuration settings.
"""
import os
import yaml
from pathlib import Path

# Resolve project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent

def load_config(config_path: str = None) -> dict:
    """Load configuration from YAML file."""
    if config_path is None:
        config_path = PROJECT_ROOT / "config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config

_config = None

def get_config() -> dict:
    """Get cached configuration."""
    global _config
    if _config is None:
        _config = load_config()
    return _config

def get_data_path(subdir: str = "raw") -> Path:
    """Get absolute path to data subdirectory."""
    config = get_config()
    if subdir == "raw":
        return PROJECT_ROOT / config["data"]["raw_dir"]
    elif subdir == "processed":
        return PROJECT_ROOT / config["data"]["processed_dir"]
    elif subdir == "demo":
        return PROJECT_ROOT / config["data"]["demo_dir"]
    elif subdir == "predictions":
        return PROJECT_ROOT / config["data"]["predictions_dir"]
    else:
        return PROJECT_ROOT / "data" / subdir

def get_db_path() -> Path:
    """Get database file path."""
    config = get_config()
    return PROJECT_ROOT / config["data"]["database_path"]

def get_campus_constants() -> dict:
    """Get campus engineering constants."""
    return get_config()["campus_constants"]

def get_building_aliases() -> dict:
    """Get building name alias mapping."""
    return get_config()["building_aliases"]
