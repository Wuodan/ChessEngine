import numpy as np
import chess
from gym_chess.alphazero.move_encoding import utils


# fixing encoding funcs from openai

def encode_knight(move: chess.Move):
	_NUM_TYPES: int = 8

	# Starting point of knight moves in last dimension of 8 x 8 x 73 action array.
	_TYPE_OFFSET: int = 56

	# Set of possible directions for a knight move, encoded as
	# (delta rank, delta square).
	_DIRECTIONS = utils.IndexedTuple(
		(+2, +1),
		(+1, +2),
		(-1, +2),
		(-2, +1),
		(-2, -1),
		(-1, -2),
		(+1, -2),
		(+2, -1),
	)

	from_rank, from_file, to_rank, to_file = utils.unpack(move)

	delta = (to_rank - from_rank, to_file - from_file)
	is_knight_move = delta in _DIRECTIONS

	if not is_knight_move:
		return None

	knight_move_type = _DIRECTIONS.index(delta)
	move_type = _TYPE_OFFSET + knight_move_type

	action = np.ravel_multi_index(
		multi_index=(from_rank, from_file, move_type),
		dims=(8, 8, 73)
	)

	return action


def encode_queen(move: chess.Move):
	_NUM_TYPES: int = 56  # = 8 directions * 7 squares max. distance
	_DIRECTIONS = utils.IndexedTuple(
		(+1, 0),
		(+1, +1),
		(0, +1),
		(-1, +1),
		(-1, 0),
		(-1, -1),
		(0, -1),
		(+1, -1),
	)

	from_rank, from_file, to_rank, to_file = utils.unpack(move)

	delta = (to_rank - from_rank, to_file - from_file)

	is_horizontal = delta[0] == 0
	is_vertical = delta[1] == 0
	is_diagonal = abs(delta[0]) == abs(delta[1])
	is_queen_move_promotion = move.promotion in (chess.QUEEN, None)

	is_queen_move = (
			(is_horizontal or is_vertical or is_diagonal)
			and is_queen_move_promotion
	)

	if not is_queen_move:
		return None

	direction = tuple(np.sign(delta))
	distance = np.max(np.abs(delta))

	direction_idx = _DIRECTIONS.index(direction)
	distance_idx = distance - 1

	move_type = np.ravel_multi_index(
		multi_index=([direction_idx, distance_idx]),
		dims=(8, 7)
	)

	action = np.ravel_multi_index(
		multi_index=(from_rank, from_file, move_type),
		dims=(8, 8, 73)
	)

	return action


def encode_under(move):
	_NUM_TYPES: int = 9  # = 3 directions * 3 piece types (see below)
	_TYPE_OFFSET: int = 64
	_DIRECTIONS = utils.IndexedTuple(
		-1,
		0,
		+1,
	)
	_PROMOTIONS = utils.IndexedTuple(
		chess.KNIGHT,
		chess.BISHOP,
		chess.ROOK,
	)

	from_rank, from_file, to_rank, to_file = utils.unpack(move)

	is_under_promotion = (
			move.promotion in _PROMOTIONS
			and from_rank == 6
			and to_rank == 7
	)

	if not is_under_promotion:
		return None

	delta_file = to_file - from_file

	direction_idx = _DIRECTIONS.index(delta_file)
	promotion_idx = _PROMOTIONS.index(move.promotion)

	promotion_type = np.ravel_multi_index(
		multi_index=([direction_idx, promotion_idx]),
		dims=(3, 3)
	)

	move_type = _TYPE_OFFSET + promotion_type

	action = np.ravel_multi_index(
		multi_index=(from_rank, from_file, move_type),
		dims=(8, 8, 73)
	)

	return action


def encode_move(move: str, board) -> int:
	move = chess.Move.from_uci(move)
	if board.turn == chess.BLACK:
		move = utils.rotate(move)

	action = encode_queen(move)

	if action is None:
		action = encode_knight(move)

	if action is None:
		action = encode_under(move)

	if action is None:
		raise ValueError(f"{move} is not a valid move")

	return action


def encode_board(board: chess.Board) -> np.array:
	"""Converts a board to numpy array representation."""

	array = np.zeros((8, 8, 14), dtype=int)

	for square, piece in board.piece_map().items():
		rank, file = chess.square_rank(square), chess.square_file(square)
		piece_type, color = piece.piece_type, piece.color

		# The first six planes encode the pieces of the active player,
		# the following six those of the active player's opponent. Since
		# this class always stores boards oriented towards the white player,
		# White is considered to be the active player here.
		offset = 0 if color == chess.WHITE else 6

		# Chess enumerates piece types beginning with one, which we have
		# to account for
		idx = piece_type - 1

		array[rank, file, idx + offset] = 1

	# Repetition counters
	array[:, :, 12] = board.is_repetition(2)
	array[:, :, 13] = board.is_repetition(3)

	return array


def encode_board_from_fen(fen: str) -> np.array:
	board = chess.Board(fen)
	return encode_board(board)