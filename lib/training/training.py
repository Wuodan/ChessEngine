import logging
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from lib.MovesAndPositions import MovesAndPositions
from lib.model.Model import Model
import time

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')


def get_device() -> torch.device:
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	logging.info(f"Using device={device}")
	return device


def get_data_loader(moves_and_positions: MovesAndPositions, batch_size, num_workers=4) -> DataLoader:
	all_positions = np.array(moves_and_positions.positions)
	all_moves = np.array(moves_and_positions.moves)
	positions = torch.from_numpy(np.asarray(all_positions))
	moves = torch.from_numpy(np.asarray(all_moves))

	training_set = TensorDataset(positions, moves)

	# Create data loaders for our datasets; shuffle for training, not for validation
	return DataLoader(training_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)


def train_one_epoch(
		model: torch.nn.Module, optimizer: torch.optim.SGD, loss_fn: torch.nn.CrossEntropyLoss,
		training_loader: DataLoader, device: torch.device) -> float:
	running_loss = 0.
	last_loss = 0.

	i = 0
	for batch_data, batch_labels in training_loader:
		start_time = time.time()

		# Transfer batch data to GPU
		batch_data = batch_data.to(device, non_blocking=True)
		batch_labels = batch_labels.to(device, non_blocking=True)

		# Make predictions for this batch
		outputs = model(batch_data)

		# Zero your gradients for every batch!
		optimizer.zero_grad()

		# Compute the loss and its gradients
		loss = loss_fn(outputs, batch_labels)
		loss.backward()

		# Adjust learning weights
		optimizer.step()

		# Gather data and report
		running_loss += loss.item()

		# Profiling each batch
		batch_time = time.time() - start_time
		if i % training_loader.batch_size == training_loader.batch_size - 1:
			print(f"Batch {i + 1}: Time taken: {batch_time:.4f}s, Loss: {running_loss / 100:.4f}")
			running_loss = 0.

		i += 1

	return last_loss


def save_best_model(vloss: float, path: str, output_folder: str) -> None:
	with open(output_folder + "/bestModel.txt", "w") as f:
		f.write(str(vloss))
		f.write("\n")
		f.write(path)
	logging.info(f"NEW BEST MODEL FOUND WITH LOSS: {vloss}")


def run_training(
		moves_and_positions: MovesAndPositions, output_folder: str,
		epochs=500, learning_rate=0.001, momentum=0.9, num_workers=4, batch_size=4096):
	best_loss = 10000000

	training_loader = get_data_loader(moves_and_positions, batch_size=batch_size, num_workers=num_workers)
	logging.info("Got data-loader")

	model = Model()
	device = get_device()

	if torch.cuda.device_count() > 1:
		logging.info(f"Using {torch.cuda.device_count()} GPUs")
		model = torch.nn.DataParallel(model)

	model.to(device)

	loss_fn = torch.nn.CrossEntropyLoss()
	optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)

	best_vloss = 1_000_000.

	for epoch_number in range(epochs):
		epoch_start_time = time.time()

		# Make sure gradient tracking is on, and do a pass over the data
		model.train(True)
		last_loss = train_one_epoch(model, optimizer, loss_fn, training_loader, device)

		running_vloss = 0.0
		model.eval()

		with torch.no_grad():
			i = 0
			for batch_data, batch_labels in training_loader:
				# for i, vdata in enumerate(training_loader):
				# vinputs, vlabels = batch_data
				batch_data = batch_data.to(device, non_blocking=True)
				batch_labels = batch_labels.to(device, non_blocking=True)
				voutputs = model(batch_data)
				vloss = loss_fn(voutputs, batch_labels)
				running_vloss += vloss
				i += 1

		avg_vloss = running_vloss / i

		if avg_vloss < best_vloss:
			best_vloss = avg_vloss

			if best_loss > best_vloss:
				model_path = output_folder + '/model'
				torch.save(model.state_dict(), model_path)
				save_best_model(best_vloss, model_path, output_folder)
				best_loss = best_vloss

		epoch_time = time.time() - epoch_start_time
		logging.info(f"Epoch {epoch_number}: Time taken: {epoch_time:.2f}s, Average loss: {avg_vloss:.4f}")

	logging.info(f"\n\nBEST VALIDATION LOSS FOR ALL MODELS: {best_loss}")
