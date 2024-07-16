import chess
import os


def clear_console() -> None:
	# Clear the console screen based on the operating system
	os.system('cls' if os.name == 'nt' else 'clear')


def print_chess_board_with_pieces(board: chess.Board) -> None:
	pieces = {
		'r': '♜', 'n': '♞', 'b': '♝', 'q': '♛', 'k': '♚', 'p': '♟',
		'R': '♖', 'N': '♘', 'B': '♗', 'Q': '♕', 'K': '♔', 'P': '♙',
		'.': ' '
	}

	# Clear the console before printing the board
	clear_console()

	# Define the top border
	print("  +---+---+---+---+---+---+---+---+")
	for row in range(8):
		# Print the row number
		print(f"{8 - row} ", end="|")
		for col in range(8):
			square = chess.square(col, 7 - row)
			piece = board.piece_at(square)
			piece_char = pieces[piece.symbol()] if piece else pieces['.']
			# Print the piece or empty square
			print(f" {piece_char} |", end="")
		# Print the row border
		print("\n  +---+---+---+---+---+---+---+---+")

	# Print the column labels
	print("    a   b   c   d   e   f   g   h")
