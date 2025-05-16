from dataclasses import dataclass, field
from pathlib import Path
from typing import Union
from train.utils import convert_to_path

default_freq_bands = {
	"Delta": (1, 4),
	"Theta": (4, 8),
	"Alpha": (8, 13),
	"Beta": (13, 30),
	"Gamma": (30, 100)
}


@dataclass
class Config:
	save_path_root: [Union[str, Path]] = '../'
	DE_method: str = 'auto'
	window_length: int = None
	step_length: int = None
	nice_number: int = None
	sample_rate: [int] = 500
	is_filtered: bool = True
	_freq_bands: dict = field(default_factory=lambda: default_freq_bands)
	
	_presets = {
		7: (2000, 2000, '../Result_feature/feature_7'),
		8: (500, 500, '../Result_feature/feature_8'),
		
	}
	
	def __post_init__(self):
		self.save_path_root = convert_to_path(self.save_path_root)
	
	@property
	def freq_bands(self):
		"""Provides read-only access to the frequency bands."""
		return self._freq_bands
	
	@freq_bands.setter
	def freq_bands(self, value):
		raise AttributeError("Modification of freq_bands is strongly discouraged. If you do want to change it, "
		                     "please modify the source code directly which in eeg_process/eeg_filtering/config.py."
		                     "Make sure it's the same as in eeg_process/eeg_filtering/config.py.")
	
	@classmethod
	def from_preset(cls, preset: int):
		if preset in cls.presets:
			return cls(*cls.presets[preset])
		else:
			raise ValueError(f"Preset {preset} is not available.")
	
	@classmethod
	def get_preset(cls, preset: int):
		"""
		Get the configuration parameters of a specified preset.
	
		Parameters:
		- preset: Preset number
	
		Returns:
		- tuple: Corresponding window_length, step_length, and save_path
		"""
		if preset in cls._presets:
			return cls._presets[preset]
		else:
			raise ValueError(f"Preset {preset} is not available.")
	
	@classmethod
	def get_all_presets(cls):
		"""
		Get all available presets.
	
		Returns:
		- dict: Dictionary of all preset values.
		"""
		return cls._presets
