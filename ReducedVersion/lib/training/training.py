from datetime import datetime

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

from ReducedVersion.lib.MovesAndPositions import MovesAndPositions
from ReducedVersion.lib.model.Model import Model


def get_device() -> torch.device:
	return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def get_data_loader(moves_and_positions: MovesAndPositions, batch_size=32) -> DataLoader:
	# transfer all data to GPU if available
	device = get_device()
	all_positions = np.array(moves_and_positions.positions)
	all_moves = np.array(moves_and_positions.moves)
	positions = torch.from_numpy(np.asarray(all_positions)).to(device)
	moves = torch.from_numpy(np.asarray(all_moves)).to(device)

	training_set = TensorDataset(positions, moves)

	# Create data loaders for our datasets; shuffle for training, not for validation
	return DataLoader(training_set, batch_size=batch_size, shuffle=True)


def train_one_epoch(
		model: Model,
		optimizer: torch.optim.SGD,
		loss_fn: torch.nn.CrossEntropyLoss,
		training_loader: DataLoader) -> float:
	running_loss = 0.
	last_loss = 0.

	# Here, we use enumerate(training_loader) instead of
	# iter(training_loader) so that we can track the batch
	# index and do some intra-epoch reporting
	for i, data in enumerate(training_loader):

		# Every data instance is an input + label pair
		inputs, labels = data

		# Zero your gradients for every batch!
		optimizer.zero_grad()

		# Make predictions for this batch
		outputs = model(inputs)

		# Compute the loss and its gradients
		loss = loss_fn(outputs, labels)
		loss.backward()

		# Adjust learning weights
		optimizer.step()

		# Gather data and report
		running_loss += loss.item()
		if i % 1000 == 999:
			# loss per batch
			last_loss = running_loss / 1000
			# print('  batch {} loss: {}'.format(i + 1, last_loss))
			running_loss = 0.

	return last_loss


def save_best_model(vloss: float, path: str, output_folder: str) -> None:
	f = open(output_folder + "/bestModel.txt", "w")
	f.write(str(vloss))
	f.write("\n")
	f.write(path)
	f.close()
	print("NEW BEST MODEL FOUND WITH LOSS:", vloss)


def run_training(moves_and_positions: MovesAndPositions, output_folder: str,
				 epochs=500, learning_rate=0.001, momentum=0.9):
	best_loss = 10000000
	epoch_number = 0

	training_loader = get_data_loader(moves_and_positions)

	model = Model()
	loss_fn = torch.nn.CrossEntropyLoss()
	optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	model.to(device)

	best_vloss = 1_000_000.

	for _ in range(epochs):

		# Make sure gradient tracking is on, and do a pass over the data
		model.train(True)
		last_loss = train_one_epoch(model, optimizer, loss_fn, training_loader)

		running_vloss = 0.0
		# Set the model to evaluation mode, disabling dropout and using population
		# statistics for batch normalization.

		model.eval()

		# Disable gradient computation and reduce memory consumption.
		with torch.no_grad():
			for i, vdata in enumerate(training_loader):
				vinputs, vlabels = vdata
				voutputs = model(vinputs)

				vloss = loss_fn(voutputs, vlabels)
				running_vloss += vloss

		avg_vloss = running_vloss / (i + 1)

		# Track the best performance, and save the model's state
		if avg_vloss < best_vloss:
			best_vloss = avg_vloss

			# if better than previous best loss from all models created, save it
			if best_loss > best_vloss:
				timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
				model_path = output_folder + '/model_{}_{}'.format(timestamp, epoch_number)
				torch.save(model.state_dict(), model_path)
				save_best_model(best_vloss, model_path, output_folder)
				best_loss = best_vloss

		epoch_number += 1
		print(f"Epoch {epoch_number}")

	print("\n\nBEST VALIDATION LOSS FOR ALL MODELS: ", best_loss)
