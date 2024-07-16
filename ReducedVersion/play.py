from pathlib import Path

import chess
import torch
from stockfish import Stockfish
from tqdm import tqdm

from ReducedVersion.lib.check_game_state.check_game_state import check_game_state, GameState
from ReducedVersion.lib.model.Model import Model


def read_best_model_file(best_model_path: str) -> str | None:
	if Path(best_model_path).is_file():
		file = open(best_model_path)
		loss = file.readline()
		model_path = file.readline()
		file.close()
		print(f"Found model at path {model_path} with loss {loss}")
		return model_path
	return None


def get_model(output_folder: str) -> Model:
	model_path = read_best_model_file(output_folder + "/bestModel.txt")
	if model_path is None:
		model_path = input("Please enter the path to the model file:").strip()
	else:
		model_path = input(f"Please enter the path to the model file ({model_path}):").strip() or model_path

	if not Path(model_path).is_file():
		raise FileNotFoundError(f"No model file exists at {model_path}")

	model = Model()
	model.load_state_dict(torch.load(model_path))

	return model


def init_stockfish(stockfish_path: str) -> Stockfish:
	elo = int(input("Please enter an integer of elo you want to play against (100):").strip() or "100")
	stockfish = Stockfish(path=stockfish_path)
	stockfish.reset_engine_parameters()
	stockfish.set_elo_rating(elo)
	stockfish.set_skill_level(0)
	return stockfish


def end_game(board: chess.Board) -> GameState:
	game_state = check_game_state(board)
	print(f"Found no legal move, game-state is {game_state}")
	if not game_state.is_game_ongoing():
		return game_state


def main() -> GameState:
	output_folder = 'data/savedModels'
	stockfish_path = r"../stockfish/stockfish-windows-x86-64-avx2.exe"
	max_number_of_moves = 150

	model = get_model(output_folder)
	stockfish = init_stockfish(stockfish_path)

	board = chess.Board()
	all_moves = []  # list of strings for saving moves for setting pos for stockfish

	for i in tqdm(range(max_number_of_moves)):  # set a limit for the game
		# first my artificial intelligence move
		# try:
		move = model.predict(board)

		if move is None:
			return end_game(board), all_moves

		board.push(move)
		# add so stockfish can see
		stockfish.make_moves_from_current_position([move])
		# all_moves.append(str(move))
		# except Exception as e:
		# 	if str(e) == "object of type 'LegalMoveGenerator' has no len()":
		# 		print("game over. You lost in", i, "moves")
		# 	else:
		# 		print("An error occurred:", str(e))
		# 	break

		# #then get stockfish move
		# stockfish.set_position(all_moves)
		stockfish_move = stockfish.get_best_move_time(1)

		if move is None:
			return end_game(board), all_moves

		all_moves.append(stockfish_move)
		stockfish_move = chess.Move.from_uci(stockfish_move)
		board.push(stockfish_move)


if __name__ == "__main__":
	game_state, all_moves = main()
	print(f"Game ended with {game_state}")
	print(f"Moves were {all_moves}")
