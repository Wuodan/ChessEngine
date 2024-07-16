import chess
from gym_chess import Chess, MoveEncoding
from gym_chess.alphazero.board_encoding import BoardHistory
from lib.MovesAndPositions import MovesAndPositions
from multiprocessing import Pool, cpu_count


def encode_game_moves(uci_moves):
	moves = []
	positions = []

	env = Chess()
	move_encoding = MoveEncoding(env)
	board_history = BoardHistory(0)

	env.reset()
	board = chess.Board()

	for uci_move in uci_moves:
		move = chess.Move.from_uci(uci_move)
		encoded_move = move_encoding.encode(move)
		encoded_position = board_history.encode(board)

		board, _, done, _ = env.step(move)
		# openai terminates the game when a draw can be claimed, prevent this
		if done:
			env._ready = True

		moves.append(encoded_move)
		positions.append(encoded_position)

	return moves, positions


def encode_moves_and_positions(parsed_games: [[str]]) -> MovesAndPositions:
	num_workers = cpu_count()

	with Pool(processes=num_workers) as pool:
		results = pool.map(encode_game_moves, parsed_games)

	all_moves = []
	all_positions = []
	for moves, positions in results:
		all_moves.extend(moves)
		all_positions.extend(positions)

	return MovesAndPositions(all_moves, all_positions)
