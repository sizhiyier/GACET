import os
import warnings

import numpy as np
import mne
import random
from pathlib import Path
from typing import Union, Tuple, List
import pickle
from torch.utils.data import Dataset
from torch.utils.data import Subset, ConcatDataset


def set_seed(seed=42):
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False
	os.environ['PYTHONHASHSEED'] = str(seed)
	torch.backends.cuda.matmul.allow_tf32 = False
	torch.backends.cudnn.allow_tf32 = False
	os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
	torch.use_deterministic_algorithms(True)

def convert_to_path(file_path):
	"""
	Convert a file path to a Path object.

	Parameters:
	file_path (Union[str, Path]): The file path to convert.

	Returns:
	Path: The converted Path object.

	Raises:
	TypeError: If the input is neither a string nor a Path object.
	"""
	if isinstance(file_path, str):
		return Path(file_path)
	elif isinstance(file_path, Path):
		return file_path
	else:
		raise TypeError("file_path must be a string or a Path object")

def load_pkl(save_path: Union[str, Path], if_raw: bool = False) -> Union[dict, None]:
	"""
	Load data from a pickle file.

	Parameters:
	- save_path (Union[str, Path]): The path to the pickle file.
	- if_raw (bool): A flag to determine which data to return if 'have_raw' is True.
					 If True, returns 'raw' data, otherwise returns 'mean' data. Default is False.

	Returns:
	- The loaded data, which could be 'raw', 'mean', or the complete content depending on the conditions.
	"""
	save_path = convert_to_path(save_path)

	with open(save_path, 'rb') as f:
		all_data = pickle.load(f)
	# Check if the loaded data has the 'have_raw' attribute (for PSD)
	have_raw = getattr(all_data, 'have_raw', False)
	if not have_raw:
		return all_data
	return all_data.get('raw') if if_raw else all_data.get('mean')


class DualSourceDataset(Dataset):
	"""
	A combined dataset that merges data from two sources.
	
	Data from each source is loaded from multiple days for each task
	according to the given task order. It assumes that the labels for each task
	are identical across both data sources.
	"""
	
	def __init__(self, data_path_1: Path, data_path_2: Path, days: list, task_order: list):
		"""
		Initialize the dataset with data from two sources combined into one dataset.

		Parameters:
		- data_path_1 (Path): Path to the first data source.
		- data_path_2 (Path): Path to the second data source.
		- days (List[int]): List of days to include.
		- task_order (List[str]): List of task file names, e.g., ['MATBeasy_eeg.pkl', 'MATBmed_eeg.pkl', ...].
		"""
		# Load data for each source using the provided days and task order
		data_group_1 = self.get_data_for_days(data_path_1, days, task_order)
		data_group_2 = self.get_data_for_days(data_path_2, days, task_order)
		
		# Create datasets for each source and merge them
		dataset_1 = self.load_dataset(data_group_1)
		dataset_2 = self.load_dataset(data_group_2)
		
		self.data = (dataset_1["data"], dataset_2["data"])
		# Assumes that labels are identical between the two sources.
		self.labels = dataset_1["labels"]
	
	def get_data_for_days(self, base_path: Path, days: list, task_order: list):
		"""
		Retrieve file paths for the specified tasks across multiple days.

		Parameters:
		- base_path (Path): Base directory of the data.
		- days (List[int]): List of days.
		- task_order (List[str]): List of task file names.
		
		Returns:
		- A list of tuples where each tuple contains a list of file paths for a task and the corresponding label.
		"""
		datasets = []
		for i, task in enumerate(task_order):
			task_files = []
			for day in days:
				# Construct the session key and corresponding path.
				ses_key = f"ses-S{day}"
				day_path = base_path / ses_key
				if not day_path.exists() or not day_path.is_dir():
					raise ValueError(f"Directory not found for day {day}: {day_path}")
				
				task_file = day_path / task
				if not task_file.exists():
					raise ValueError(f"Task file not found: {task_file}")
				task_files.append(task_file)
			datasets.append((task_files, i))
		return datasets
	
	def load_dataset(self, datasets_with_labels: list):
		"""
		Load and combine data for the given datasets.

		Parameters:
		- datasets_with_labels (list): A list of tuples, each containing a list of file paths and the task label.

		Returns:
		- A dictionary containing:
			- "data": A torch.Tensor with all data concatenated.
			- "labels": A torch.Tensor containing labels repeated for each data sample.
		"""
		combined_data = []
		combined_labels = []
		for data_paths, label in datasets_with_labels:
			# Load data from each file and concatenate along axis 0.
			loaded_data = np.concatenate([load_pkl(p) for p in data_paths], axis=0)
			tensor_data = torch.tensor(loaded_data, dtype=torch.float32)
			combined_data.append(tensor_data)
			combined_labels.extend([label] * len(tensor_data))
		
		combined_data = torch.cat(combined_data, dim=0)
		combined_labels = torch.tensor(combined_labels, dtype=torch.long)
		return {"data": combined_data, "labels": combined_labels}
	
	def trim_to_length(self, max_len: int):
		"""
		Trim the dataset so that it does not exceed the specified max_len.
		"""
		self.data = (self.data[0][:max_len], self.data[1][:max_len])
		self.labels = self.labels[:max_len]
	
	def __len__(self):
		return len(self.labels)
	
	def __getitem__(self, idx):
		"""
		Return the data sample at the specified index.

		Returns:
			A tuple: ((data_source_1_sample, data_source_2_sample), label)
		"""
		return (self.data[0][idx], self.data[1][idx]), self.labels[idx]

class DualSourceDataSplitter:
	"""
	A class to split two dual-source datasets into training and validation subsets,
	and then combine them into single training and validation datasets.
	"""
	
	def __init__(self, dataset1, dataset2, train_idx, val_idx):
		"""
		Initialize the splitter with two datasets and the corresponding train/validation indices.

		Parameters:
		- dataset1: A PyTorch Dataset instance from the first data source.
		- dataset2: A PyTorch Dataset instance from the second data source.
		- train_idx: List or array of indices for the training subset.
		- val_idx: List or array of indices for the validation subset.
		"""
		# Create training subsets for both datasets
		train_subset_1 = Subset(dataset1, train_idx)
		train_subset_2 = Subset(dataset2, train_idx)
		
		# Create validation subsets for both datasets
		val_subset_1 = Subset(dataset1, val_idx)
		val_subset_2 = Subset(dataset2, val_idx)
		
		# Concatenate subsets for combined train and validation datasets
		self.train_dataset = ConcatDataset([train_subset_1, train_subset_2])
		self.val_dataset = ConcatDataset([val_subset_1, val_subset_2])


import torch
from torch.utils.data import Dataset, Subset

class StandardizedDataset(Dataset):
	"""
	A dataset wrapper to standardize features for dual-source datasets.
	It is designed for datasets where each sample is structured as ((data1, data2), label).
	When mean and std are not provided, they will be computed from the training data.
	"""
	
	def __init__(self, dataset, is_train=True, mean=None, std=None):
		"""
		Initialize the standardized dataset.

		Parameters:
		- dataset: The original dataset (e.g., output of DualSourceDataSplitter) where each sample is ((data1, data2), label).
		- is_train (bool): Indicates if the dataset is used for training. If True and mean/std are not provided, compute them.
		- mean: A list of means for each feature (data source). If None, computed from the dataset.
		- std: A list of standard deviations for each feature. If None, computed from the dataset.
		"""
		self.dataset = dataset
		self.is_train = is_train
		
		if mean is None or std is None:
			# Extract original data for each feature from the dataset
			original_data = self._get_original_data(dataset)
			# Compute mean and std independently for each data source
			self.mean = [torch.mean(feature_data, dim=0) for feature_data in original_data]
			self.std = [torch.std(feature_data, dim=0) for feature_data in original_data]
		else:
			self.mean = mean
			self.std = std
	
	def _get_original_data(self, dataset):
		"""
		Retrieve original feature data from the dataset.
		Handles both full datasets and Subset datasets.
		
		For each sample assumed as ((data1, data2), label), this method stacks
		all data1 into one tensor and all data2 into another.
		"""
		if isinstance(dataset, Subset):
			# Recursively call for the underlying dataset and slice with indices.
			original_data = self._get_original_data(dataset.dataset)
			return [feature_data[dataset.indices] for feature_data in original_data]
		else:
			# Assume every sample is structured as ((data1, data2), label)
			num_features = len(dataset[0][0])
			return [
				torch.stack([sample[0][i] for sample in dataset])
				for i in range(num_features)
			]
	
	def get_mean_std(self):
		"""
		Return the computed (or provided) mean and standard deviation.
		"""
		return self.mean, self.std
	
	def __len__(self):
		return len(self.dataset)
	
	def __getitem__(self, idx):
		"""
		For a given index, return the standardized sample.
		
		Standardizes each feature (data source) independently.
		
		Returns:
			A tuple: ((standardized_data1, standardized_data2), label)
		"""
		(data_tuple, label) = self.dataset[idx]
		# Standardize each data source independently
		standardized_data = tuple(
			(x - m) / s for x, m, s in zip(data_tuple, self.mean, self.std)
		)
		return standardized_data, label

class ChannelPositionManager:
	# 预定义的电极列表
	_PRESET_CHANNEL_LISTS = {
		'data_1': [
			'Fp1','Fz','F3','F7','FT9','FC5','FC1','C3','T7','CP5','CP1','Pz',
			'P3','P7','O1','Oz','O2','P4','P8','TP10','CP6','CP2','FCz','C4',
			'T8','FT10','FC6','FC2','F4','F8','Fp2','AF7','AF3','AFz','F1',
			'F5','FT7','FC3','C1','C5','TP7','CP3','P1','P5','PO7','PO3','POz',
			'PO4','PO8','P6','P2','CPz','CP4','TP8','C6','C2','FC4','FT8','F6',
			'AF8','AF4','F2'
		],
		'data_2': [
			'FP1','FPZ','FP2','AF3','AF4','F7','F5','F3','F1','FZ','F2','F4','F6','F8',
			'FT7','FC5','FC3','FC1','FCZ','FC2','FC4','FC6','FT8','T7','C5','C3','C1','CZ','C2','C4','C6','T8',
			'TP7','CP5','CP3','CP1','CPZ','CP2','CP4','CP6','TP8','P7','P5','P3','P1','PZ','P2','P4','P6','P8',
			'PO7','PO5','PO3','POZ','PO4','PO6','PO8','O1','OZ','O2'
		]
	}
	
	def __init__(self, montage_name: str = 'standard_1020'):

		self.montage_name = montage_name
		montage = mne.channels.make_standard_montage(montage_name)
		self._pos_dict = montage.get_positions()['ch_pos']
		self._upper_map = {name.upper(): name for name in self._pos_dict.keys()}
	
	def get_positions(self,
	                  source: str = None,
	                  ch_names: list[str] = None
	                  ) -> np.ndarray:

		if source:
			key = source.lower()
			if key not in self._PRESET_CHANNEL_LISTS:
				valid = list(self._PRESET_CHANNEL_LISTS.keys())
				raise ValueError(f"Unknown source '{source}', choose from {valid} or provide ch_names")
			channels = self._PRESET_CHANNEL_LISTS[key]
		elif ch_names:
			channels = ch_names
		else:
			raise ValueError("Must provide either 'source' or 'ch_names'")
		
		coords = []
		for nm in channels:
			key = nm.upper()
			if key in self._upper_map:
				real = self._upper_map[key]
				x, y, _ = self._pos_dict[real]
				coords.append([x, y])
			else:
				warnings.warn(f"Channel '{nm}' not found in montage '{self.montage_name}'")
				coords.append([np.nan, np.nan])
		
		coords_array = np.array(coords)
		# 确保返回行数与输入通道数一致
		assert coords_array.shape[0] == len(channels), (
			f"Returned positions {coords_array.shape[0]} rows but expected {len(channels)}"
		)
		return coords_array
