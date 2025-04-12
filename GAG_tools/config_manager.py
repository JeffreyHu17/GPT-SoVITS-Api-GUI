import json
import os
import threading
from copy import deepcopy
from typing import Dict, Any


class ConfigDict:
    def __init__(self, config_manager):
        self._config_manager = config_manager

    def __getitem__(self, key):
        return self._config_manager.get_value(key)

    def __setitem__(self, key, value):
        self._config_manager.set_value(key, value)

    def get(self, key, default=None):
        return self._config_manager.get_value(key, default)

    def __delitem__(self, key):
        self._config_manager.delete_value(key)


class ConfigManager:
    _instance_lock = threading.Lock()
    _instance = None

    def __new__(cls):
        if not cls._instance:
            with cls._instance_lock:
                if not cls._instance:
                    cls._instance = super(ConfigManager, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if not hasattr(self, '_initialized'):
            self.config_file = os.path.abspath("GAG_config.json")
            self._config = {}
            self._config_dict = ConfigDict(self)
            self._initialized = True

            self._default_config = {
                'api_url': 'http://127.0.0.1:9880',
                'python_path': os.path.join('runtime', 'python.exe'),
                'autostart_api': True,
                'presets': {
                    'Default': {
                        'text_lang': 'all_zh',
                        'ref_audio_path': '',
                        'aux_ref_audio_paths': [],
                        'prompt_text': '',
                        'prompt_lang': 'all_zh',
                        'top_k': 5,
                        'top_p': 1.0,
                        'temperature': 1.0,
                        'text_split_method': 'cut1',
                        'batch_size': 1,
                        'batch_threshold': 0.75,
                        'split_bucket': False,
                        'return_fragment': False,
                        'speed_factor': 1.0,
                        'streaming_mode': False,
                        'seed': -1,
                        'parallel_infer': False,
                        'repetition_penalty': 1.35,
                        'gpt_model': '',
                        'sovits_model': '',
                        'no_prompt': False,
                        "sample_steps": 32,
                        "super_sampling": False,
                    }
                },
                'current_preset': 'Default',
                'save_directory': '',
                'batch_tts_save_directory': '',
                'batch_tts_segment_size': 100,
                'batch_tts_tasks': [],
            }

            self.load_config()

    def load_config(self) -> None:
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    loaded_config = json.load(f)
                    self._config = self._deep_merge(deepcopy(self._default_config), loaded_config)
            else:
                self._config = deepcopy(self._default_config)
                self.save_config()
        except Exception as e:
            print(f"Error loading config: {e}")
            self._config = deepcopy(self._default_config)
            self.save_config()

    def _deep_merge(self, base: Dict, update: Dict) -> Dict:
        for key, value in update.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                base[key] = self._deep_merge(base[key], value)
            else:
                base[key] = value
        return base

    def save_config(self) -> None:
        try:
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(self._config, f, indent=4, ensure_ascii=False)
        except Exception as e:
            print(f"Error saving config: {e}")

    def update_config(self, updates: Dict[str, Any], save: bool = True) -> None:
        self._config = self._deep_merge(self._config, updates)
        if save:
            self.save_config()

    @property
    def config(self) -> Dict:
        return self._config_dict

    def get_value(self, key: str, default: Any = None) -> Any:
        try:
            keys = key.split('.')
            value = self._config
            for k in keys:
                value = value[k]
            return deepcopy(value)
        except KeyError:
            return default

    def set_value(self, key: str, value: Any, save: bool = True) -> None:
        keys = key.split('.')
        current = self._config
        for k in keys[:-1]:
            if k not in current:
                current[k] = {}
            current = current[k]
        current[keys[-1]] = deepcopy(value)
        if save:
            self.save_config()

    def delete_value(self, key: str, save: bool = True) -> bool:
        try:
            keys = key.split('.')
            current = self._config
            for k in keys[:-1]:
                current = current[k]
            if keys[-1] in current:
                del current[keys[-1]]
                if save:
                    self.save_config()
                return True
            return False
        except (KeyError, TypeError):
            return False