import warnings
from dataclasses import dataclass, field
from typing import List, Tuple, Union, Dict, Optional
from pathlib import Path
from train.utils import convert_to_path


@dataclass
class Config:
	"""
	Configure classes that store all preprocessing parameters and settings
	"""
	save_path_root: Union[str, Path]  # Save folder path, can be a string or Path object
	channels_drop: List[str] = field(default_factory=list)  # Default to an empty list
	eeg_freq_range: Tuple[int, int] = (1, 100)  # EEG frequency range
	notch_freq_range: Tuple[int, int] = (48, 52)  # Notch filter frequency range
	ecg_freq_range: Tuple[float, float] = (0.04, 40)  # ECG frequency range
	if_baseline: bool = False  # Whether to use baseline correction
	resampled_sfreq: int = None
	nice_number: int = None  # Setting the priority
	n_jobs: int = 12
	
	# Customizable preprocessing options
	channel_types: Optional[Dict[str, str]] = None  # Custom channel types
	ref_channels: Union[str, List[str]] = 'average'
	apply_channel_types = True
	apply_drop_channel: bool = True
	apply_filter_data: bool = True
	apply_drop_bad_channels: bool = True
	apply_reference_data: bool = True
	apply_ica: bool = True
	
	apply_drop_outside_annotations: bool = False
	apply_drop_bad_period: bool = False
	apply_resample: bool = False
	
	# Threshold
	threshold_drop_bad_channels: float = 2.0
	threshold_annotate_muscle_zscore: float = 4.0
	threshold_find_bads_muscle: float = 0.9
	threshold_pro_icalabel: float = 0.9
	
	# Artifact repair
	if_drop_bad_channels: bool = True
	if_annotate_muscle_zscore: bool = False
	if_find_bads_muscle: bool = True
	if_find_bads_ecgs: bool = True
	if_find_bads_eogs: bool = True
	only_brain: bool = False
	
	def __post_init__(self):
		self.save_path_root = convert_to_path(self.save_path_root)
		
		if self.resampled_sfreq is not None and not self.apply_resample:
			warnings.warn(
				"Resampling frequency (new_sfreq) is specified, but apply_resample is set to False. "
				"Resampling will not be applied."
			)
		if self.apply_resample and self.resampled_sfreq is None:
			self.resampled_sfreq = 100
			warnings.warn(
				"apply_resample is set to True, but no resampling frequency (new_sfreq) was specified. "
				"Setting new_sfreq to 100 Hz by default."
			)
		if self.channel_types is None:
			self.apply_channel_types = False
		
		if isinstance(self.channels_drop, str):
			self.channels_drop = [self.channels_drop]
		elif not isinstance(self.channels_drop, list):
			raise TypeError(f"'channels_drop' must be a list or a signal string, got: {type(self.channels_drop)}")
		
		if isinstance(self.ref_channels, str) and self.ref_channels != 'average':
			self.ref_channels = [self.ref_channels]
		elif not (self.ref_channels == 'average' or isinstance(self.ref_channels, list)):
			raise ValueError(
				"Invalid reference_method. It should be 'average', a channel name, or a list of channel names.")
