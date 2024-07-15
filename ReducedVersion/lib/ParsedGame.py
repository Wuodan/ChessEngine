# holds moves and positions of one game
class ParsedGame:
	moves = []
	positions = []

	def __init__(self, moves: [str], positions: [str]):
		self.moves = moves
		self.positions = positions

	def __repr__(self):
		return f"{[self.moves, self.positions]}"
