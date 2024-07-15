from print_chess_board_with_pieces import print_chess_board_with_pieces
import chess
import time

# Create a new board
board = chess.Board()


# Function to update the board with a move
def update_board(_board, _move):
	if _move in _board.legal_moves:
		_board.push(_move)
		print_chess_board_with_pieces(_board)
	else:
		print("Illegal move!")


# Print the initial board
print_chess_board_with_pieces(board)

# Simulate some moves with a delay
moves = ["e2e4", "e7e5", "g1f3", "b8c6"]
for move in moves:
	time.sleep(1)
	update_board(board, chess.Move.from_uci(move))
