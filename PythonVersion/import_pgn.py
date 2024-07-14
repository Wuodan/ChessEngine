import chess.pgn
# import json
from libs.Training import *

def pgn_to_moves_and_positions(pgn_file):
	# all_games_moves_positions = []

	with open(pgn_file, "r") as pgn:
		while True:
			game = chess.pgn.read_game(pgn)
			if game is None:
				break

			board = game.board()
			moves_list = []
			# positions_list = [board.fen()]  # Initial position
			positions_list = []

			for move in game.mainline_moves():
				moves_list.append(move.uci())
				board.push(move)
				positions_list.append(board.fen())

			# all_games_moves_positions.append((moves_list, positions_list))
			saveData(moves_list, positions_list)

	# return all_games_moves_positions

# Usage
pgn_file = "../data/pgn_import/ficsgamesdb_2023_chess2000_nomovetimes_394057.pgn"
# all_games_moves_positions = pgn_to_moves_and_positions(pgn_file)
pgn_to_moves_and_positions(pgn_file)

# Print the moves and positions arrays for each game
# for i, (moves, positions) in enumerate(all_games_moves_positions):
# 	print(f"Game {i + 1}:")
# 	print("Moves:", moves)
# 	print("Positions:", positions)
# 	print()

# Optionally, save the arrays to JSON files for each game
# for i, (moves, positions) in enumerate(all_games_moves_positions):
# 	with open(f"moves_game_{i + 1}.json", "w") as moves_file:
# 		json.dump(moves, moves_file)
#
# 	with open(f"positions_game_{i + 1}.json", "w") as positions_file:
# 		json.dump(positions, positions_file)
