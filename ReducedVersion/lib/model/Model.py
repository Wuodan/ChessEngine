import chess
import numpy as np
import torch
from ReducedVersion.lib.encode_decode.encode import encode_board
from ReducedVersion.lib.encode_decode.decode import decode_move


class Model(torch.nn.Module):

	def __init__(self):
		super(Model, self).__init__()
		self.INPUT_SIZE = 896
		# self.INPUT_SIZE = 7*7*13 # NOTE changing input size for using CNNs
		self.OUTPUT_SIZE = 4672  # = number of unique moves (action space)

		# can try to add CNN and pooling here (calculations taking into account spacial features)

		# input shape for sample is (8,8,14), flattened to 1d array of size 896
		# self.cnn1 = nn.Conv3d(4,4,(2,2,4), padding=(0,0,1))

		self.activation = torch.nn.Tanh()
		# self.activation = torch.nn.ReLU()

		self.linear1 = torch.nn.Linear(self.INPUT_SIZE, 1000)
		self.linear2 = torch.nn.Linear(1000, 1000)
		self.linear3 = torch.nn.Linear(1000, 1000)
		self.linear4 = torch.nn.Linear(1000, 200)
		self.linear5 = torch.nn.Linear(200, self.OUTPUT_SIZE)
		self.softmax = torch.nn.Softmax(1)  # use softmax as prob for each move, dim 1 as dim 0 is the batch dimension

	def forward(self, x):  # x.shape = (batch size, 896)
		x = x.to(torch.float32)
		# x = self.cnn1(x) # for using CNNs
		x = x.reshape(x.shape[0], -1)
		x = self.linear1(x)
		x = self.activation(x)
		x = self.linear2(x)
		x = self.activation(x)
		x = self.linear3(x)
		x = self.activation(x)
		x = self.linear4(x)
		x = self.activation(x)
		x = self.linear5(x)
		# x = self.softmax(x) # do not use softmax since you are using cross entropy loss
		return x

	def predict(self, board: chess.Board):
		# takes in a chess board and returns a move-object.
		# NOTE: this function should definitely be written better, but it works for now
		with torch.no_grad():
			encoded_board = encode_board(board)
			encoded_board = encoded_board.reshape(1, -1)
			encoded_board = torch.from_numpy(encoded_board)
			res = self.forward(encoded_board)
			probs = self.softmax(res)

			probs = probs.numpy()[0]  # do not want tensor anymore, 0 since it is a 2d array with 1 row

			# verify that move is legal and can be decoded before returning
			while len(probs) > 0:  # try max 100 times, if not throw an error
				move_idx = probs.argmax()
				try:  # TODO should not have try here, but was a bug with idx 499 if it is black to move
					uci_move = decode_move(move_idx, board)
					if uci_move is None:  # could not decode
						probs = np.delete(probs, move_idx)
						continue
					move = chess.Move.from_uci(str(uci_move))
					if move in board.legal_moves:  # if legal, return, else: loop continues after deleting the move
						return move
				except Exception as e:
					print(f"something seriously went wrong: {e}")
					pass
				# remove the move, so it's not chosen again next iteration
				# TODO probably better way to do this, but it is not too time critical as it is only for predictions
				probs = np.delete(probs, move_idx)

			# return random move if model failed to find move
			moves = board.legal_moves
			if moves.count() > 0:
				print(f"Returning one of {moves.count()} moves")
				return np.random.choice(np.array(list(moves)))
			print("Your predict function could not find any legal/decodable moves")
			# if no legal moves found, return None
			# TODO raise Exception("Your predict function could not find any legal/decodable moves")
			return None
