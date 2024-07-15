import numpy as np


# holds encoded moves and positions of one or several games
class MovesAndPositions:
	moves = []
	positions = []

	def __init__(self, moves: [int], positions: [np.array]):
		self.moves = moves
		self.positions = positions

	def __repr__(self):
		return f"{[self.moves, self.positions]}"
