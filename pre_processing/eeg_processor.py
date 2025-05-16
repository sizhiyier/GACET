# signal_process/eeg_process/eeg_preprocessing/eeg_processor.py
import inspect
import numpy as np
import warnings
import mne
from mne.preprocessing import (find_bad_channels_lof, annotate_muscle_zscore, ICA)
from mne_icalabel import label_components
from pathlib import Path
from typing import List, Union, Optional

from signal_process.utils.file_utils import ensure_directory_exists

from config import Config
from record_manager import RecordManager


class EEGProcessor:
	"""
	EEG processing class, responsible for the preprocessing flow of a single EEG file.
	"""
	
	def __init__(self, config: Config, record_manager: RecordManager) -> None:
		self.config = config
		self.record_manager = record_manager
		self.task = ''
	
	def drop_channel(self, raw: mne.io.BaseRaw) -> mne.io.BaseRaw:
		"""Drop specified channels"""
		channels = self.config.channels_drop
		non_existent_channels = [ch for ch in channels if ch not in raw.ch_names]
		if non_existent_channels:
			warnings.warn(
				f"The following channels do not exist in the data and cannot be dropped: {non_existent_channels}")
		raw.drop_channels([ch for ch in channels if ch in raw.ch_names])
		return raw
	
	def drop_outside_annotations(self, raw: mne.io.Raw, first_label: Optional[str] = None,
	                             last_label: Optional[str] = None,
	                             seconds_before_first: float = 0.0, seconds_after_last: float = 0.0) -> mne.io.Raw:
		"""
		Drop data outside specific annotations in the raw data.

		Parameters:
		raw (mne.io.Raw): The raw EEG data.
		first_label (str, optional): The label of the first annotation to keep.
		last_label (str, optional): The label of the last annotation to keep.
		seconds_before_first (float, optional): Seconds to keep before the first annotation. Defaults to 0.
		seconds_after_last (float, optional): Seconds to keep after the last annotation. Defaults to 0.

		Returns:
		mne.io.Raw: The cropped raw data.
		"""
		annotations = raw.annotations
		
		# Get the onset times of annotations with the given labels
		if first_label is None:
			first_onset = annotations.onset[0]  # First annotation onset
		else:
			first_onsets = [ann.onset for ann in annotations if ann['description'] == first_label]
			if not first_onsets:
				raise ValueError(f"Label '{first_label}' not found in annotations.")
			first_onset = first_onsets[0]
		
		if last_label is None:
			last_onset = annotations.onset[-1] + annotations.duration[-1]  # End of the last annotation
		else:
			last_onsets = [ann.onset for ann in annotations if ann['description'] == last_label]
			if not last_onsets:
				raise ValueError(f"Label '{last_label}' not found in annotations.")
			last_onset = last_onsets[-1] + annotations[last_onsets[-1]].duration
		
		# Calculate crop start and end times
		crop_start = first_onset - seconds_before_first
		crop_end = last_onset + seconds_after_last
		
		# Ensure crop times are within data limits
		crop_start = max(crop_start - 0.1, 0)
		crop_end = min(crop_end + 0.1, raw.times[-1])
		
		# Crop the raw data
		raw_cropped = raw.copy().crop(tmin=crop_start, tmax=crop_end)
		# Crop the annotations
		cropped_annotations = annotations.copy().crop(tmin=crop_start, tmax=crop_end)
		raw_cropped.set_annotations(cropped_annotations)
		
		return raw_cropped
	
	def filter_data(self, raw: mne.io.BaseRaw) -> mne.io.BaseRaw:
		"""Apply band-pass and notch filtering to raw data"""
		raw.filter(l_freq=self.config.eeg_freq_range[0], h_freq=self.config.eeg_freq_range[1], picks='eeg')
		raw.notch_filter(freqs=self.config.notch_freq_range, picks='eeg')
		if 'ecg' in raw.get_channel_types():
			raw.filter(l_freq=self.config.ecg_freq_range[0], h_freq=self.config.ecg_freq_range[1], picks='ecg')
		return raw
	
	def drop_bad_channels(self, raw: mne.io.BaseRaw) -> mne.io.BaseRaw:
		"""Identify and interpolate bad channels"""
		if self.config.if_drop_bad_channels:
			try:
				bad_channels = find_bad_channels_lof(raw, threshold=self.config.threshold_drop_bad_channels,
				                                     picks='eeg')
				raw.info['bads'].extend(bad_channels)
				raw.interpolate_bads()
				self.record_manager.update_record({'bad_channels': [len(bad_channels)]})
			except Exception as e:
				print(f"Error in drop_bad_channels: {e}")
				raise
		return raw
	
	def reference_data(self, raw: mne.io.BaseRaw) -> mne.io.BaseRaw:
		"""Apply reference"""
		raw.set_eeg_reference(ref_channels=self.config.ref_channels, ch_type='eeg')
		return raw
	
	def resample_data(self, raw: mne.io.BaseRaw) -> mne.io.BaseRaw:
		raw.resample(sfreq=self.config.resampled_sfreq)
		return raw
	
	def apply_ica(self, raw: mne.io.BaseRaw) -> mne.io.BaseRaw:
		"""Apply ICA to remove artifacts"""
		try:
			ica = ICA(method='infomax', fit_params=dict(extended=True), max_iter="auto", random_state=42)
			ica.fit(raw)
			exclude_idx = self.identify_artifacts(raw, ica)
			ica.exclude = exclude_idx
			ica.apply(raw)
			self.record_manager.update_record({'excluded_components': [len(exclude_idx)]})
		except Exception as e:
			print(f"Error in apply_ica: {e}")
			raise
		return raw
	
	def identify_artifacts(self, raw: mne.io.BaseRaw, ica: ICA) -> List[int]:
		"""Identify artifact components"""
		exclude_idx = []
		if self.config.only_brain:
			ic_labels = label_components(raw, ica, method="iclabel")
			brain_idx = [
				idx for idx, (prob, lbl) in enumerate(zip(ic_labels['y_pred_proba'], ic_labels['labels']))
				if lbl == "brain"
			]
			all_components = list(range(ica.n_components_))
			exclude_idx = [idx for idx in all_components if idx not in brain_idx]
		else:
			if self.config.if_find_bads_muscle:
				muscle_idx, _ = ica.find_bads_muscle(raw, threshold=self.config.threshold_find_bads_muscle)
				exclude_idx.extend(muscle_idx)
				self.record_manager.update_record({'excluded_muscle': [len(muscle_idx)]})
			if self.config.if_find_bads_ecgs:
				ecg_idx, _ = ica.find_bads_ecg(raw, ch_name='ECG1')
				exclude_idx.extend(ecg_idx)
				self.record_manager.update_record({'excluded_ecg': [len(ecg_idx)]})
			if self.config.if_find_bads_eogs:
				ic_labels = label_components(raw, ica, method="iclabel")
				eog_idx = [
					idx for idx, (prob, lbl) in enumerate(zip(ic_labels['y_pred_proba'], ic_labels['labels']))
					if lbl == "eye blink" and prob > self.config.threshold_pro_icalabel
				]
				exclude_idx.extend(eog_idx)
				self.record_manager.update_record({'excluded_eog': [len(eog_idx)]})
		return list(set(exclude_idx))
	
	def drop_bad_period(self, raw: mne.io.BaseRaw) -> mne.io.BaseRaw:
		"""Annotate and remove muscle artifacts"""
		if self.config.if_annotate_muscle_zscore:
			try:
				annot_muscle, _ = annotate_muscle_zscore(
					raw,
					threshold=self.config.threshold_annotate_muscle_zscore,
					ch_type='eeg',
					filter_freq=(1, 100)
				)
				raw.set_annotations(annot_muscle)
				raw = raw.copy().crop(annotations='BAD_muscle', invert=True)
				bad_durations = round(np.sum(annot_muscle.duration), 2)
				self.record_manager.update_record({'total_bad_duration': [bad_durations]})
			except Exception as e:
				print(f"Error in drop_bad_period: {e}")
				raise
		return raw
	
	def save_processed_data(self, raw: mne.io.BaseRaw, file_name: str, save_path: Path) -> None:
		"""Save processed EEG data, and ECG data if present."""
		
		# Save EEG data
		raw_eeg = raw.copy().pick_types(eeg=True)
		save_path_matlab = save_path / 'matlab'
		
		eeg_fif_path = save_path / f'{file_name}_eeg.fif'
		raw_eeg.save(eeg_fif_path, overwrite=True)
		print(f"Saved EEG data to {eeg_fif_path}")
		
		eeg_set_path = save_path_matlab / f'{file_name}_eeg.set'
		raw_eeg.export(eeg_set_path, overwrite=True)
		print(f"Exported EEG data to {eeg_set_path}")
		
		# Check if ECG channel exists, if yes, save ECG data
		if 'ecg' in raw.get_channel_types():
			raw_ecg = raw.copy().pick(picks='ecg')
			
			# Save ECG fif file
			ecg_fif_path_new = [part if part != 'eeg' else 'ecg' for part in save_path.parts]
			save_path_ecg_fif = Path(*ecg_fif_path_new)
			ensure_directory_exists(save_path_ecg_fif)
			ecg_fif_path = save_path_ecg_fif / f'{file_name}_ecg.fif'
			raw_ecg.save(ecg_fif_path, overwrite=True)
			print(f"Saved ECG data to {ecg_fif_path}")
			
			# Save ECG set file
			ecg_set_path_new = [part if part != 'eeg' else 'ecg' for part in save_path_matlab.parts]
			save_path_ecg_set = Path(*ecg_set_path_new)
			ensure_directory_exists(save_path_ecg_set)
			ecg_set_path = save_path_ecg_set / f'{file_name}_ecg.set'
			raw_ecg.export(ecg_set_path, overwrite=True)
			print(f"Exported ECG data to {ecg_set_path}")
		else:
			print("No ECG channels found in the raw data.")
	
	def process_raw(self, raw: mne.io.BaseRaw, file_name: str, save_path: Union[str, Path]) -> None:
		"""Process the raw EEG data through a series of preprocessing steps"""
		try:
			# List of tuples containing config attributes and corresponding methods to call
			processing_steps = [
				(self.config.apply_channel_types, lambda: raw.set_channel_types(self.config.channel_types)),
				(self.config.apply_drop_channel, lambda: self.drop_channel(raw)),
				(self.config.apply_drop_outside_annotations, lambda: self.drop_outside_annotations(raw)),
				(self.config.apply_filter_data, lambda: self.filter_data(raw)),
				(self.config.apply_drop_bad_channels, lambda: self.drop_bad_channels(raw)),
				(self.config.apply_reference_data, lambda: self.reference_data(raw)),
				(self.config.apply_resample, lambda: self.resample_data(raw)),
				(self.config.apply_ica, lambda: self.apply_ica(raw)),
				(self.config.apply_drop_bad_period, lambda: self.drop_bad_period(raw))
			]
			
			for condition, method in processing_steps:
				if condition:
					method_name = inspect.getsource(method).strip()
					raw = method()
			
			self.save_processed_data(raw, file_name, save_path)
			print(f"File {file_name} processed and saved successfully.")
		except Exception as e:
			print(f"Error occurred while processing file {file_name}: {e}")
			raise
