import chess.pgn


def pgn_to_moves_and_positions(pgn_file):
	moves_list = []
	positions_list = []

	with open(pgn_file, "r") as pgn:
		while True:
			game = chess.pgn.read_game(pgn)
			if game is None:
				break

			board = game.board()

			for move in game.mainline_moves():
				moves_list.append(move.uci())
				board.push(move)
				positions_list.append(board.fen())

	assert len(moves_list) == len(positions_list), "moves and positions must be of same length"
	return [moves_list, positions_list]
