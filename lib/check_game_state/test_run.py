import chess
from check_game_state import check_game_state


def run(fen: str) -> None:
	board = chess.Board(fen)
	state = check_game_state(board)
	# Check game state
	if state.is_game_ongoing():
		print("The game is still ongoing.")
		print(state)
		print(state.draw_or_who_won())
	else:
		print(state)
		print(state.draw_or_who_won())
		result = state.draw_or_who_won()
		print(f"The game ended in: {result.value}")  # Outputs "draw", "white", or "black"


# Example usage
run("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")  # ongoing
run("2r2r2/5p1k/6b1/1p1Q4/1P6/8/2n3PP/4q1K1 w - - 0 35")  # black wins
run("R1k5/1p4pp/3BpN2/2pp4/3P4/3B1P1P/1P3P2/3QK2R b K - 1 24")  # white wins
