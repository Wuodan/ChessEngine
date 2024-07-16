import chess.pgn
from ReducedVersion.lib.ParsedGame import ParsedGame


def pgn_to_moves_and_positions(pgn_file: str) -> [ParsedGame]:
	parsed_games = []

	with open(pgn_file, "r") as pgn:
		moves = []
		positions = []
		while True:
			game = chess.pgn.read_game(pgn)

			if game is None:
				break

			board = game.board()

			for move in game.mainline_moves():
				moves.append(move.uci())
				board.push(move)
				positions.append(board.fen())

			assert len(moves) == len(positions), "Moves and positions of a game must be of same length"
			parsed_game = ParsedGame(moves, positions)
			parsed_games.append(parsed_game)

	return parsed_games
