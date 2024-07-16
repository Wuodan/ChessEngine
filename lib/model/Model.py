import time

import chess
import numpy as np
import torch
from gym_chess.alphazero.board_encoding import BoardHistory
from torch import Tensor

from lib.encode_decode.decode import decode_move


class Model(torch.nn.Module):

	def __init__(self) -> None:
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

		self.random_number_generator = np.random.default_rng(int(time.time()))
		self.board_history = BoardHistory(0)

	def forward(self, x: Tensor) -> Tensor:  # x.shape = (batch size, 896)
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

	def predict(self, board: chess.Board) -> chess.Move | None:
		# takes in a chess board and returns a move-object.
		# NOTE: this function should definitely be written better, but it works for now
		with torch.no_grad():
			encoded_position = self.board_history.encode(board)
			encoded_position = encoded_position.reshape(1, -1)
			tensor_board = torch.from_numpy(encoded_position)
			res = self.forward(tensor_board)
			probs = self.softmax(res)

			probs = probs.numpy()[0]  # do not want tensor anymore, 0 since it is a 2d array with 1 row

			legal_moves = board.legal_moves
			self.print_legal_moves(legal_moves)

			# verify that move is legal and can be decoded before returning
			while len(probs) > 0:  # try max 100 times, if not throw an error
				move_idx = int(probs.argmax())
				try:  # TODO should not have try here, but was a bug with idx 499 if it is black to move
					uci_move = decode_move(move_idx, board)
					if uci_move is None:  # could not decode
						probs = np.delete(probs, move_idx)
						continue
					move = chess.Move.from_uci(str(uci_move))
					if move in legal_moves:  # if legal, return, else: loop continues after deleting the move
						self.print_move(legal_moves, move, "Got legal move from model: ")
						return move

					# todo debugging
					if move is None:
						print(f"Why is move None? uci_move = {uci_move}")

				except IndexError:
					# this happens when chess.Move.from_uci(str(uci_move) fails, printing uci_move will produce an error
					# print(f"IndexError with index {move_idx} and uci_move {uci_move}: {ie}")
					# print(f"IndexError with index {move_idx}: {ie}")
					# todo remove debug
					# uci_move = decode_move(move_idx, board)
					# move = chess.Move.from_uci(str(uci_move))
					# print(f"uci_move={uci_move}")
					# self.print_move(legal_moves, uci_move, "uci_move=")
					pass
				except Exception as e:
					print(f"something seriously went wrong with index {move_idx} and move {move}: {e}")
					pass
				# remove the move, so it's not chosen again next iteration
				# TODO probably better way to do this, but it is not too time critical as it is only for predictions
				probs = np.delete(probs, move_idx)

			# return random move if model failed to find move
			if legal_moves.count() > 0:
				move = self.random_number_generator.choice(np.array(list(legal_moves)))
				self.print_move(legal_moves, move, "Returning random move: ")
				return move

			# print("Your predict function could not find any legal/decodable moves")
			# if no legal moves found, return None
			# TODO raise Exception("Your predict function could not find any legal/decodable moves")
			return None

	@staticmethod
	def print_legal_moves(legal_moves: chess.LegalMoveGenerator) -> None:
		sans = ", ".join(legal_moves.board.lan(move) for move in legal_moves)
		print(f"Legal moves are:\n<LegalMoveGenerator at {id(legal_moves):#x} ({sans})>")

	@staticmethod
	def print_move(legal_moves: chess.LegalMoveGenerator, move: chess.Move, prefix="") -> None:
		sans = legal_moves.board.lan(move)
		print(f"{prefix}{sans}")
