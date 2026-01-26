"""
Configuration Loader Module

Provides centralized configuration management with support for YAML files,
environment variable overrides, and runtime configuration updates.
"""

import os
import yaml
from pathlib import Path
from typing import Any, Dict, Optional
from dataclasses import dataclass, field


class ConfigLoader:
    """
    Singleton configuration loader that manages all system settings.
    
    Supports:
        - YAML configuration file loading
        - Environment variable overrides
        - Nested key access via dot notation
        - Runtime configuration updates
    """
    
    _instance: Optional['ConfigLoader'] = None
    _config: Dict[str, Any] = {}
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not hasattr(self, '_initialized'):
            self._initialized = True
            self._config_path: Optional[Path] = None
    
    def load(self, config_path: str = "config/config.yaml") -> Dict[str, Any]:
        """
        Load configuration from YAML file.
        
        Args:
            config_path: Path to the YAML configuration file
            
        Returns:
            Loaded configuration dictionary
            
        Raises:
            FileNotFoundError: If config file doesn't exist
            yaml.YAMLError: If config file is invalid
        """
        # Resolve path relative to project root
        project_root = Path(__file__).parent.parent.parent.parent
        full_path = project_root / config_path
        
        if not full_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {full_path}")
        
        with open(full_path, 'r', encoding='utf-8') as f:
            self._config = yaml.safe_load(f)
        
        self._config_path = full_path
        self._apply_env_overrides()
        self._resolve_paths(project_root)
        
        return self._config
    
    def _apply_env_overrides(self):
        """Apply environment variable overrides with POTHOLE_ prefix."""
        for key, value in os.environ.items():
            if key.startswith('POTHOLE_'):
                config_key = key[8:].lower().replace('__', '.')
                self.set(config_key, self._parse_value(value))
    
    def _parse_value(self, value: str) -> Any:
        """Parse string value to appropriate type."""
        # Boolean
        if value.lower() in ('true', 'false'):
            return value.lower() == 'true'
        # Integer
        try:
            return int(value)
        except ValueError:
            pass
        # Float
        try:
            return float(value)
        except ValueError:
            pass
        # String
        return value
    
    def _resolve_paths(self, project_root: Path):
        """Convert relative paths to absolute paths."""
        if 'paths' in self._config:
            for key, value in self._config['paths'].items():
                if isinstance(value, str) and not os.path.isabs(value):
                    self._config['paths'][key] = str(project_root / value)
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation.
        
        Args:
            key: Configuration key (e.g., "vision.training.epochs")
            default: Default value if key not found
            
        Returns:
            Configuration value or default
        """
        keys = key.split('.')
        value = self._config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def set(self, key: str, value: Any):
        """
        Set configuration value using dot notation.
        
        Args:
            key: Configuration key (e.g., "vision.training.epochs")
            value: Value to set
        """
        keys = key.split('.')
        config = self._config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
    
    def get_vision_config(self) -> Dict[str, Any]:
        """Get vision pipeline configuration."""
        return self._config.get('vision', {})
    
    def get_accel_config(self) -> Dict[str, Any]:
        """Get accelerometer pipeline configuration."""
        return self._config.get('accelerometer', {})
    
    def get_fusion_config(self) -> Dict[str, Any]:
        """Get fusion engine configuration."""
        return self._config.get('fusion', {})
    
    def get_paths(self) -> Dict[str, str]:
        """Get all path configurations."""
        return self._config.get('paths', {})
    
    @property
    def config(self) -> Dict[str, Any]:
        """Get full configuration dictionary."""
        return self._config
    
    def reload(self):
        """Reload configuration from file."""
        if self._config_path:
            self.load(str(self._config_path))


# Global config instance
_config_instance: Optional[ConfigLoader] = None


def get_config(config_path: str = "config/config.yaml") -> ConfigLoader:
    """
    Get or create the global configuration instance.
    
    Args:
        config_path: Path to configuration file (only used on first call)
        
    Returns:
        ConfigLoader instance
    """
    global _config_instance
    
    if _config_instance is None:
        _config_instance = ConfigLoader()
        _config_instance.load(config_path)
    
    return _config_instance
