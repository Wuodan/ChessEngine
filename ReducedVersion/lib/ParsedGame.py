# holds moves and positions of one game
# TODO obsolete -> remove
class ParsedGame:
	moves = []
	positions = []

	def __init__(self, moves: [str], positions: [str]) -> None:
		self.moves = moves
		self.positions = positions

	def __repr__(self) -> str:
		return f"{[self.moves, self.positions]}"
