"""
data_source.py
--------------
Modular data abstraction layer.
Decouples telemetry ingestion from dashboard rendering and ML inference,
enabling direct plug-and-play migration from offline/simulated CSV data
to real-world IoT sensor networks (ESP32, MQTT, Inverter Modbus, REST APIs).
"""

from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
import pandas as pd

from utils.data_processing import load_dataset


class BaseDataSource(ABC):
    """
    Abstract Base Class for Telemetry Data Sources.
    Any future sensor integration (ESP32, Arduino, MQTT, HTTP) should implement this contract.
    """

    @abstractmethod
    def get_source_name(self) -> str:
        """Human-readable name of the source."""
        pass

    @abstractmethod
    def is_simulated(self) -> bool:
        """Returns True if the data stream is synthetic/simulated."""
        pass

    @abstractmethod
    def fetch_latest_reading(self) -> Dict[str, Any]:
        """Fetch the most recent single telemetry record."""
        pass

    @abstractmethod
    def fetch_historical_data(
        self, start_time: Optional[datetime] = None, end_time: Optional[datetime] = None
    ) -> pd.DataFrame:
        """Fetch a batch or historical range of telemetry data."""
        pass


class CSVDataSource(BaseDataSource):
    """
    CSV-backed data source simulating time-series data streams.
    """

    def __init__(self, file_path: str = "data/renewable_data.csv"):
        self.file_path = file_path
        self._df: Optional[pd.DataFrame] = None
        self._load_cache()

    def _load_cache(self) -> None:
        self._df = load_dataset(self.file_path)

    def get_source_name(self) -> str:
        return f"Simulated CSV Dataset ({Path(self.file_path).name})"

    def is_simulated(self) -> bool:
        return True

    def get_dataframe(self) -> pd.DataFrame:
        if self._df is None:
            self._load_cache()
        return self._df.copy()

    def fetch_latest_reading(self) -> Dict[str, Any]:
        df = self.get_dataframe()
        if df.empty:
            return {}
        latest = df.iloc[-1].to_dict()
        return latest

    def fetch_historical_data(
        self, start_time: Optional[datetime] = None, end_time: Optional[datetime] = None
    ) -> pd.DataFrame:
        df = self.get_dataframe()
        if start_time is not None:
            df = df[df["timestamp"] >= pd.to_datetime(start_time)]
        if end_time is not None:
            df = df[df["timestamp"] <= pd.to_datetime(end_time)]
        return df.reset_index(drop=True)


class MQTTDataSource(BaseDataSource):
    """
    Template for future IoT Broker Integration (e.g., Mosquitto, HiveMQ, AWS IoT).
    Subscribes to topics:
      - plant/solar/telemetry (irradiance, ambient_temp, module_temp, pv_power)
      - plant/wind/telemetry (wind_speed, wind_direction, wind_power)
    """

    def __init__(self, broker_url: str = "mqtt://localhost:1883", client_id: str = "ren-ai-sub"):
        self.broker_url = broker_url
        self.client_id = client_id
        self._connected = False

    def get_source_name(self) -> str:
        return f"Live MQTT Broker ({self.broker_url})"

    def is_simulated(self) -> bool:
        return False

    def fetch_latest_reading(self) -> Dict[str, Any]:
        # Implementation contract for paho-mqtt listener or redis ring buffer
        raise NotImplementedError("Connect physical ESP32 or MQTT broker to activate live stream.")

    def fetch_historical_data(
        self, start_time: Optional[datetime] = None, end_time: Optional[datetime] = None
    ) -> pd.DataFrame:
        # Implementation contract for TimeScaleDB or InfluxDB backend
        raise NotImplementedError("Connect database backend for live MQTT history.")


class RESTAPIDataSource(BaseDataSource):
    """
    Template for Inverter API integration (e.g., SolarEdge, Fronius, SMA WebConnect).
    """

    def __init__(self, api_endpoint: str, api_key: str):
        self.api_endpoint = api_endpoint
        self.api_key = api_key

    def get_source_name(self) -> str:
        return f"Inverter REST API ({self.api_endpoint})"

    def is_simulated(self) -> bool:
        return False

    def fetch_latest_reading(self) -> Dict[str, Any]:
        raise NotImplementedError("Connect inverter API credentials to activate live stream.")

    def fetch_historical_data(
        self, start_time: Optional[datetime] = None, end_time: Optional[datetime] = None
    ) -> pd.DataFrame:
        raise NotImplementedError("Connect inverter API credentials to activate history.")
