import numpy as np
import mne

from .base import FeatureCalculator, PSDResult
from .config import Config


class PSDCalculator(FeatureCalculator):
	"""
	Power Spectral Density (PSD) calculator.
	"""
	
	def __init__(self, sample):
		self.sample = sample
	
	def calculate(self, data: dict) -> PSDResult:
		"""
		Calculate power spectral density.

		Parameters:
		- data: Filtered EEG data, shape [n_example, n_channel, n_sample].

		Returns:
		- Dictionary containing average PSD and raw PSD:
		  {
			  'mean': np.ndarray,  # Averaged PSD, shape [n_example, total_features]
			  'raw': list          # Raw PSD list, one element per frequency band
		  }
		"""
		freq_bands = Config().freq_bands
		psd_all = []
		psd_all_raw = []
		try:
			for band_name, band_data in data.items():
				n_fft = 512

				psd = mne.time_frequency.psd_array_welch(
					band_data,
					sfreq=self.sample,
					n_fft=n_fft,
					fmin=freq_bands[band_name][0],
					fmax=freq_bands[band_name][1],
					# n_per_seg=512
				)
				print(f"PSD calculated successfully with n_fft: {n_fft}")
				psd_mean = np.mean(psd[0], axis=2)  # Average over frequencies
				psd_all.append(psd_mean)
				psd_all_raw.append(psd[0])
			return PSDResult({
				'mean': np.concatenate(psd_all, axis=1),
				'raw': psd_all_raw
			}, have_raw=True)
		except Exception as e:
			print(f"Error in PSDCalculator: {e}")
			raise
