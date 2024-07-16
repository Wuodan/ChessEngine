from print_chess_board_with_pieces import print_chess_board_with_pieces
import chess
import time


# Function to update the board with a move
def update_board(board: chess.Board, move: chess.Move) -> None:
	if move in board.legal_moves:
		board.push(move)
		print_chess_board_with_pieces(board)
	else:
		print("Illegal move!")


def main() -> None:
	# Create a new board
	board = chess.Board()
	# Print the initial board
	print_chess_board_with_pieces(board)

	# Simulate some moves with a delay
	moves = ["e2e4", "e7e5", "g1f3", "b8c6"]
	for move in moves:
		time.sleep(1)
		update_board(board, chess.Move.from_uci(move))


if __name__ == "__main__":
	main()
