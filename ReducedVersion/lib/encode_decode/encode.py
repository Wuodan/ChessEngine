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
		game = parsed_games[i_game]

		for i_move in range(len(game.moves)):
			uci_move = game.moves[i_move]

			move = chess.Move.from_uci(uci_move)
			encoded_move = move_encoding.encode(move)
			board, _, done, _ = env.step(move)

			# openai terminates the game when a draw can be claimed, prevent this
			if done:
				env._ready = True

			encoded_position = board_history.encode(board)

			all_moves.append(encoded_move)
			all_positions.append(encoded_position)

	return MovesAndPositions(all_moves, all_positions)
