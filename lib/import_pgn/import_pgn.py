import chess.pgn


def pgn_to_moves_and_positions(pgn_file: str) -> [[str]]:
	parsed_games = []

	with open(pgn_file, "r") as pgn:
		while True:
			game = chess.pgn.read_game(pgn)

			if game is None:
				break

			game_moves = [move.uci() for move in game.mainline_moves()]
			parsed_games.append(game_moves)

	return parsed_games
