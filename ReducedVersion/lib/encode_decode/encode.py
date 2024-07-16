import chess
from gym_chess import Chess
from gym_chess.alphazero.board_encoding import BoardHistory
from gym_chess.alphazero.move_encoding import MoveEncoding

from ReducedVersion.lib.MovesAndPositions import MovesAndPositions
from ReducedVersion.lib.ParsedGame import ParsedGame


# TODO obsolete -> remove
def encode_moves_and_positions(parsed_games: [[ParsedGame]]) -> MovesAndPositions:
	all_moves = []
	all_positions = []

	env = Chess()

	move_encoding = MoveEncoding(env)
	board_history = BoardHistory(0)

	for i_game in range(len(parsed_games)):
		env.reset()
		print(f"Encoding game {i_game}")
		# for game in parsed_games:
		game = parsed_games[i_game]

		for i_move in range(len(game.moves)):
			print(f"Encoding game {i_game}, move {i_move}")
			uci_move = game.moves[i_move]

			move = chess.Move.from_uci(uci_move)
			print(f"Encoding game {i_game}, move {i_move}: {move}")
			encoded_move = move_encoding.encode(move)
			board, reward, done, foo = env.step(move)
			print(f"result of last step() is done={done}")
			# openai terminates the game when a draw can be claimed, prevent this
			if done:
				env._ready = True
			encoded_position = board_history.encode(board)

			# encoded_move = encode_move(uci_move, board)
			# encoded_position = encode_board_from_fen(fen_position)

			all_moves.append(encoded_move)
			all_positions.append(encoded_position)

	return MovesAndPositions(all_moves, all_positions)
