import numpy as np
import chess
from chess import Move
from gym_chess.alphazero.move_encoding import utils
from typing import Optional


# decoding moves from idx to uci notation
def decode_knight(action: int) -> Optional[chess.Move]:
	"""
	Decodes the given action into a knight move in the chess game.

	Args:
		action (int): The action to decode.

	Returns:
		Optional[chess.Move]: The decoded knight move as a chess.Move object, or None if the action is not a valid knight move
	"""

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

	from_rank, from_file, move_type = np.unravel_index(action, (8, 8, 73))

	is_knight_move = (
			_TYPE_OFFSET <= move_type < _TYPE_OFFSET + _NUM_TYPES
	)

	if not is_knight_move:
		return None

	knight_move_type = move_type - _TYPE_OFFSET

	delta_rank, delta_file = _DIRECTIONS[knight_move_type]

	to_rank = from_rank + delta_rank
	to_file = from_file + delta_file

	move = utils.pack(from_rank, from_file, to_rank, to_file)
	return move


def decode_queen(action: int) -> Optional[chess.Move]:
	_NUM_TYPES: int = 56  # = 8 directions * 7 squares max. distance

	# Set of possible directions for a queen move, encoded as
	# (delta rank, delta square).
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
	from_rank, from_file, move_type = np.unravel_index(action, (8, 8, 73))

	is_queen_move = move_type < _NUM_TYPES

	if not is_queen_move:
		return None

	direction_idx, distance_idx = np.unravel_index(
		indices=move_type,
		shape=(8, 7)
	)

	direction = _DIRECTIONS[direction_idx]
	distance = distance_idx + 1

	delta_rank = direction[0] * distance
	delta_file = direction[1] * distance

	to_rank = from_rank + delta_rank
	to_file = from_file + delta_file

	move = utils.pack(from_rank, from_file, to_rank, to_file)
	return move


def decode_under_promotion(action: int) -> chess.Move | None:
	_NUM_TYPES: int = 9  # = 3 directions * 3 piece types (see below)

	# Starting point of under_promotions in last dimension of 8 x 8 x 73 action
	# array.
	_TYPE_OFFSET: int = 64

	# Set of possible directions for an under_promotion, encoded as file delta.
	_DIRECTIONS = utils.IndexedTuple(
		-1,
		0,
		+1,
	)

	# Set of possible piece types for an under_promotion (promoting to a queen
	# is implicitly encoded by the corresponding queen move).
	_PROMOTIONS = utils.IndexedTuple(
		chess.KNIGHT,
		chess.BISHOP,
		chess.ROOK,
	)

	from_rank, from_file, move_type = np.unravel_index(action, (8, 8, 73))

	is_under_promotion = (
			_TYPE_OFFSET <= move_type < _TYPE_OFFSET + _NUM_TYPES
	)

	if not is_under_promotion:
		return None

	under_promotion_type = move_type - _TYPE_OFFSET

	direction_idx, promotion_idx = np.unravel_index(
		indices=under_promotion_type,
		shape=(3, 3)
	)

	direction = _DIRECTIONS[direction_idx]
	promotion = _PROMOTIONS[promotion_idx]

	to_rank = from_rank + 1
	to_file = from_file + direction

	move = utils.pack(from_rank, from_file, to_rank, to_file)
	move.promotion = promotion

	return move


# primary decoding function, the ones above are just helper functions
def decode_move(action: int, board: chess.Board) -> Move | None:
	move = decode_queen(action)
	is_queen_move = move is not None

	if not move:
		move = decode_knight(action)

	if not move:
		move = decode_under_promotion(action)

	if not move:
		raise ValueError(f"{action} is not a valid action")

	# Actions encode moves from the perspective of the current player. If
	# this is the black player, the move must be reoriented.
	turn = board.turn

	if not turn:  # black to move
		move = utils.rotate(move)

	# Moving a pawn to the opponent's home rank with a queen move
	# is automatically assumed to be queen under_promotion. However,
	# since queen-moves has no reference to the board and can thus not
	# determine whether the moved piece is a pawn, we have to add this
	# information manually here
	if is_queen_move:
		to_rank = chess.square_rank(move.to_square)
		is_promoting_move = (
				(to_rank == 7 and turn) or
				(to_rank == 0 and not turn)
		)

		piece = board.piece_at(move.from_square)
		if piece is None:  # NOTE I added this, not entirely sure if it's correct
			return None
		is_pawn = piece.piece_type == chess.PAWN

		if is_pawn and is_promoting_move:
			move.promotion = chess.QUEEN

	return move
