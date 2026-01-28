"""
Configuration Management System - JSON Only
Loads and merges configuration from JSON files
"""
import json
import os
from pathlib import Path
from typing import Dict, Any, Optional
import logging
from copy import deepcopy


class ConfigurationManager:
    """
    Central configuration management using JSON files ONLY.
    Supports environment-specific overrides.
    """
    
    _instance = None
    _config: Dict[str, Any] = {}
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not self._config:
            self.logger = logging.getLogger(__name__)
            self._load_configuration()
    
    def _load_configuration(self):
        """Load configuration from JSON files"""
        # Navigate from src/application/config to project root
        config_dir = Path(__file__).parent.parent.parent.parent / "config"
        
        # Load base configuration
        base_config_path = config_dir / "config.json"
        if not base_config_path.exists():
            raise FileNotFoundError(f"Base configuration not found: {base_config_path}")
        
        with open(base_config_path, 'r') as f:
            self._config = json.load(f)
        
        # Load environment-specific configuration
        env = os.getenv('POTHOLE_ENV', 'development')
        env_config_path = config_dir / f"{env}.json"
        
        if env_config_path.exists():
            with open(env_config_path, 'r') as f:
                env_config = json.load(f)
                self._merge_config(env_config)
            self.logger.info(f"Loaded {env} environment configuration")
        else:
            self.logger.warning(f"Environment config not found: {env_config_path}")
        
        self.logger.info("Configuration loaded successfully")
    
    def _merge_config(self, override: Dict[str, Any]):
        """Recursively merge override configuration into base"""
        def merge_dict(base: Dict, override: Dict):
            for key, value in override.items():
                if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                    merge_dict(base[key], value)
                else:
                    base[key] = value
        
        merge_dict(self._config, override)
    
    def get(self, key_path: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation.
        
        Args:
            key_path: Dot-separated path (e.g., 'vision.inference.confidence_threshold')
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        keys = key_path.split('.')
        value = self._config
        
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        
        return value
    
    def get_section(self, section: str) -> Dict[str, Any]:
        """
        Get entire configuration section.
        
        Args:
            section: Section name (e.g., 'vision', 'accelerometer')
            
        Returns:
            Configuration section as dictionary
        """
        return deepcopy(self._config.get(section, {}))
    
    def get_all(self) -> Dict[str, Any]:
        """Get complete configuration"""
        return deepcopy(self._config)
    
    def set(self, key_path: str, value: Any):
        """
        Set configuration value (runtime only, not persisted).
        
        Args:
            key_path: Dot-separated path
            value: Value to set
        """
        keys = key_path.split('.')
        config = self._config
        
        for key in keys[:-1]:
            if key not in config:
                config[key] = {}
            config = config[key]
        
        config[keys[-1]] = value
    
    def reload(self):
        """Reload configuration from files"""
        self._config = {}
        self._load_configuration()


# Global configuration instance
config = ConfigurationManager()


def get_config() -> ConfigurationManager:
    """Get global configuration instance"""
    return config
