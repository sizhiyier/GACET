import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau

class EarlyStopping:
	def __init__(self, patience=5, delta=0.001, window=3, mode='max', verbose=True):
		"""
		Early Stopping mechanism to halt training when the evaluation metric has not improved
		significantly over a specified number of epochs.

		Parameters:
		- patience (int): Number of consecutive epochs without significant improvement before stopping.
		- delta (float): Minimum change in the monitored metric to qualify as an improvement.
		- window (int): Number of recent epochs to consider for calculating the average metric.
		- mode (str): 'max' if the metric should be maximized, 'min' if it should be minimized.
		- verbose (bool): Whether to print early stopping related messages.
		"""
		self.patience = patience
		self.delta = delta
		self.window = window
		self.mode = mode
		self.verbose = verbose
		
		self.best_score = None
		self.counter = 0
		self.score_history = []
		self.early_stop = False
		
		if self.mode == 'max':
			self.compare = lambda current, best: current > best + self.delta
		elif self.mode == 'min':
			self.compare = lambda current, best: current < best - self.delta
		else:
			raise ValueError("mode should be 'max' or 'min'")
	
	def __call__(self, current_score):
		"""
		Update the early stopping status based on the current evaluation score.

		Parameters:
		- current_score (float): The current epoch's evaluation metric value.

		Returns:
		- bool: True if training should be stopped, False otherwise.
		"""
		if self.best_score is None:
			self.best_score = current_score
			if self.verbose:
				print(f"Initialized best score: {self.best_score:.4f}")
			return False
		
		# Update score history
		self.score_history.append(current_score)
		if len(self.score_history) > self.window:
			removed_score = self.score_history.pop(0)
			if self.verbose:
				print(f"Removed oldest score: {removed_score:.4f}")
		
		# Calculate the average score over the sliding window
		current_avg = sum(self.score_history) / len(self.score_history)
		if self.verbose:
			print(f"Current sliding window average ({len(self.score_history)}): {current_avg:.4f}")
		
		# Check for significant improvement
		if self.compare(current_avg, self.best_score):
			self.best_score = current_avg
			self.counter = 0
			if self.verbose:
				print(f"Significant improvement detected, updating best score to: {self.best_score:.4f}")
		else:
			self.counter += 1
			if self.verbose:
				print(f"No significant improvement, patience counter: {self.counter}/{self.patience}")
			if self.counter >= self.patience:
				if self.verbose:
					print("Patience limit reached, triggering early stopping.")
				self.early_stop = True
		
		return self.early_stop
	
class Trainer:
	def __init__(self, model, train_loader, val_loader, test_loader, criterion, optimizer, device, num_epochs=200):
		"""
		Trainer class for training a model with early stopping and a learning rate scheduler.

		Parameters:
			model (torch.nn.Module): The model to be trained.
			train_loader (DataLoader): DataLoader for the training dataset.
			val_loader (DataLoader): DataLoader for the validation dataset.
			test_loader (DataLoader): DataLoader for the test dataset.
			criterion (Loss): Loss function.
			optimizer (Optimizer): Optimizer.
			device (torch.device): Device to perform training on (CPU or GPU).
			num_epochs (int): Maximum number of epochs for training.
			patience (int): Patience for early stopping.
		"""
		self.model = model
		self.train_loader = train_loader
		self.val_loader = val_loader
		self.test_loader = test_loader
		self.criterion = criterion
		self.optimizer = optimizer
		self.device = device
		self.num_epochs = num_epochs
		
		self.scheduler = ReduceLROnPlateau(
			optimizer,
			mode='max',
			factor=0.1,
			patience=5,
			threshold=0.01
		)
		
		self.early_stopping = EarlyStopping(
			patience=10,
			delta=0.01,
			window=5,
			mode='max',
			verbose=False
		)
		self.best_val_accuracy = 0.0
		self.best_model_state = None
	
	def train_epoch(self):
		"""
		Perform one epoch of training.

		Returns:
			epoch_loss (float): Average training loss.
			epoch_accuracy (float): Training accuracy.
		"""
		self.model.train()
		running_loss = 0.0
		correct = 0
		total = 0
		
		for inputs, labels in self.train_loader:
			# Move data to the device
			inputs = tuple(input_.to(self.device, dtype=torch.float32) for input_ in inputs)
			labels = labels.to(self.device, dtype=torch.long)
			
			self.optimizer.zero_grad()
			outputs = self.model(inputs)
			loss = self.criterion(outputs, labels)
			loss.backward()
			self.optimizer.step()
			
			running_loss += loss.item() * labels.size(0)
			_, predicted = torch.max(outputs, 1)
			correct += (predicted == labels).sum().item()
			total += labels.size(0)
		
		epoch_loss = running_loss / total
		epoch_accuracy = correct / total
		return epoch_accuracy
	
	def validate(self):
		"""
		Validate the model on the validation dataset.

		Returns:
			val_loss (float): Average validation loss.
			val_accuracy (float): Validation accuracy.
		"""
		self.model.eval()
		running_loss = 0.0
		correct = 0
		total = 0
		
		with torch.no_grad():
			for inputs, labels in self.val_loader:
				inputs = tuple(input_.to(self.device, dtype=torch.float32) for input_ in inputs)
				labels = labels.to(self.device, dtype=torch.long)
				outputs = self.model(inputs)
				loss = self.criterion(outputs, labels)
				running_loss += loss.item() * labels.size(0)
				_, predicted = torch.max(outputs, 1)
				correct += (predicted == labels).sum().item()
				total += labels.size(0)
		
		val_accuracy = correct / total
		return val_accuracy
	
	def test(self):
		"""
		Evaluate the model on the test dataset.

		Returns:
			test_loss (float): Average test loss.
			test_accuracy (float): Test accuracy.
		"""
		self.model.eval()
		running_loss = 0.0
		correct = 0
		total = 0
		
		with torch.no_grad():
			for inputs, labels in self.test_loader:
				inputs = tuple(input_.to(self.device, dtype=torch.float32) for input_ in inputs)
				labels = labels.to(self.device, dtype=torch.long)
				outputs = self.model(inputs)
				loss = self.criterion(outputs, labels)
				running_loss += loss.item() * labels.size(0)
				_, predicted = torch.max(outputs, 1)
				correct += (predicted == labels).sum().item()
				total += labels.size(0)
		
		test_accuracy = correct / total
		return test_accuracy
	
	def train(self):
		"""
		Train the model with early stopping and learning rate scheduler.

		This method runs the training loop, validates after each epoch, checks for early stopping,
		and finally prints the final validation and test accuracy.

		"""
		for epoch in range(self.num_epochs):
			train_accuracy = self.train_epoch()
			val_accuracy = self.validate()
			if val_accuracy > self.best_val_accuracy:
				self.best_val_accuracy = val_accuracy
				self.best_model_state = self.model.state_dict()
			self.scheduler.step(val_accuracy)
			should_stop = self.early_stopping(val_accuracy)
			if should_stop:
				break
		print(f"The Best Validation Accuracy: {self.best_val_accuracy * 100:.2f}%")
		
		# Load the best model state from training
		if self.best_model_state is not None:
			self.model.load_state_dict(self.best_model_state)
			print("Loaded best model state from training.")
		
		# Evaluate and print test accuracy
		test_accuracy = self.test()
		print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
		
		return test_accuracy
