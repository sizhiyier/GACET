
from base import FeatureCalculator
import numpy as np

def differential_entropy_frequency_domain(signal, axis=-1):
	"""
	h(X) = 0.5 * log(power_spectrum) + 0.5 * log(2πe/N)
	"""
	signal = np.asarray(signal)
	signal = np.moveaxis(signal, axis, -1)
	N = signal.shape[-1]
	fft_values = np.fft.fft(signal, axis=-1)
	power_spectrum = np.mean(np.abs(fft_values )**2, axis=-1)
	entropy = 0.5 * np.log(power_spectrum) + 0.5 * np.log(2 * np.pi * np.e / N)
	
	return entropy




class DECalculator(FeatureCalculator):
	"""
	Differential Entropy (DE) calculator.
	"""
	
	def __init__(self, method: str):
		self.method = method
	
	def calculate(self, data: dict) -> np.ndarray:
		"""
		Calculate differential entropy.

		Parameters:
		- data: EEG data with shape [n_example, n_channel, n_sample].

		Returns:
		- DE data with shape [n_example, n_channel].
		"""
		try:
			results = []
			for band_name, band_data in data.items():
				# band_data.shape=(N,num_electrodes,window_duration)
				results.append(differential_entropy_frequency_domain(band_data, axis=-1))
			return np.concatenate(results, axis=-1)
		
		except Exception as e:
			print(f"Error in DECalculator: {e}")
			raise
