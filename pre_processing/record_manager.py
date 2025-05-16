
import os
import warnings
import pandas as pd
from pathlib import Path
from dataclasses import asdict
from dataclasses import is_dataclass
from config import Config


class RecordManager:
	"""
	Record management class, responsible for storing data generated during preprocessing
	and saving it as CSV and TXT files.
	"""
	
	def __init__(self, config):
		self.record_dict = {}  # Dictionary to store records
		# Ensure config is an instance of the dataclass and specifically of type Config
		if not (is_dataclass(config) and isinstance(config, Config)):
			raise TypeError("config must be specifically of type Config.")
		self.config: Config = config
		self.save_path = self.config.save_path_root
	
	def update_record(self, local_record):
		"""Update the record dictionary"""
		for key, value in local_record.items():
			self.record_dict.setdefault(key, []).extend(value)
	
	def save_to_csv(self):
		"""Save records as an Excel file, with separate sheets for records and averages"""
		averages = {}
		for key, values in self.record_dict.items():
			numeric_values = []
			for v in values:
				try:
					numeric_values.append(float(v))
				except ValueError:
					warnings.warn(f"Warning: Key '{key}' contains non-numeric value '{v}', skipped.")
					continue
			if numeric_values:
				averages[key] = sum(numeric_values) / len(numeric_values)
			else:
				averages[key] = None
		
		record_df = pd.DataFrame.from_dict(self.record_dict, orient='index').reset_index()
		record_df.columns = ['Key'] + [f'Value_{i + 1}' for i in range(record_df.shape[1] - 1)]
		
		averages_df = pd.DataFrame(list(averages.items()), columns=['Key', 'Average'])
		
		excel_path = Path(os.path.join(self.save_path, 'record.xlsx'))
		
		with pd.ExcelWriter(excel_path, engine='xlsxwriter') as writer:
			record_df.to_excel(writer, sheet_name='record', index=False, header=False)
			averages_df.to_excel(writer, sheet_name='average', index=False, header=False)
	
	def save_to_txt(self):
		"""Save configuration information to a text file"""
		# Convert the dataclass instance to a dictionary
		config_dict = asdict(self.config)
		
		# Define the path for saving the file
		file_path = Path(os.path.join(self.save_path, 'attribute.txt'))
		
		# Write the configuration data to a text file
		with open(file_path, 'w', encoding='utf-8') as file:
			for key, value in config_dict.items():
				file.write(f"{key}: {value}\n")
