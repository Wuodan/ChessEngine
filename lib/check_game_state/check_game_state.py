from enum import Enum

import chess


class GameOutcome(Enum):
	DRAW = "draw"
	WHITE = "white"
	BLACK = "black"


class GameState(Enum):
	ONGOING = "Game is still ongoing"
	WHITE_WINS = "White wins (checkmate)"
	BLACK_WINS = "Black wins (checkmate)"
	DRAW_CAN_CLAIM = "Draw (can claim)"
	DRAW_STALEMATE = "Draw (stalemate)"
	DRAW_INSUFFICIENT_MATERIAL = "Draw (insufficient material)"
	DRAW_REPETITION = "Draw (fivefold repetition)"
	DRAW_FIFTY_MOVES = "Draw (seventy-five moves without pawn movement or capture)"

	def is_game_ongoing(self) -> bool:
		return self == GameState.ONGOING

	def draw_or_who_won(self) -> GameOutcome | None:
		if self in {GameState.DRAW_CAN_CLAIM,
					GameState.DRAW_STALEMATE,
					GameState.DRAW_INSUFFICIENT_MATERIAL,
					GameState.DRAW_REPETITION,
					GameState.DRAW_FIFTY_MOVES}:
			return GameOutcome.DRAW
		elif self == GameState.WHITE_WINS:
			return GameOutcome.WHITE
		elif self == GameState.BLACK_WINS:
			return GameOutcome.BLACK
		return None  # For ongoing games


def check_game_state(board: chess.Board) -> GameState:
	if board.is_checkmate():
		if board.turn:  # True for white's turn
			return GameState.BLACK_WINS
		else:
			return GameState.WHITE_WINS
	elif board.can_claim_draw():  # Check if a draw can be claimed
		return GameState.DRAW_STALEMATE  # Using stalemate for all draws for simplicity
	elif board.is_stalemate():
		return GameState.DRAW_STALEMATE
	elif board.is_insufficient_material():
		return GameState.DRAW_INSUFFICIENT_MATERIAL
	elif board.is_fivefold_repetition():
		return GameState.DRAW_REPETITION
	elif board.is_seventyfive_moves():
		return GameState.DRAW_FIFTY_MOVES
	else:
		return GameState.ONGOING
