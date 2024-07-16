import logging
from pathlib import Path

import chess
import torch
from chess import pgn

from lib.check_game_state.check_game_state import check_game_state, GameState
from lib.model.Model import Model
from stockfish import Stockfish

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S.%f')


def read_best_model_file(best_model_path: str) -> str | None:
	if Path(best_model_path).is_file():
		file = open(best_model_path)
		loss = file.readline()
		model_path = file.readline()
		file.close()
		logging.info(f"Found model at path {model_path} with loss {loss}")
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
	logging.info(f"Found no legal move, game-state is {game_state}")
	if not game_state.is_game_ongoing():
		return game_state


def main() -> chess.Board:
	output_folder = 'data/savedModels'
	stockfish_path = r"stockfish/stockfish-windows-x86-64-avx2.exe"
	max_number_of_moves = 150

	model = get_model(output_folder)
	stockfish = init_stockfish(stockfish_path)

	board = chess.Board()

	# set a limit for the game
	for _ in range(max_number_of_moves):
		# first my artificial intelligence move
		move = model.predict(board)

		if move is None:
			return board

		board.push(move)

		# add so stockfish can see
		stockfish.make_moves_from_current_position([move.uci()])

		# #then get stockfish move
		stockfish_move = stockfish.get_best_move_time(1)

		if move is None:
			return board

		stockfish_move = chess.Move.from_uci(stockfish_move)

		stockfish.make_moves_from_current_position([stockfish_move.uci()])
		board.push(stockfish_move)

	game_state = end_game(board)
	logging.info(f"Game ended with {game_state}")

	game = pgn.Game.from_board(board)
	exporter = chess.pgn.StringExporter(headers=False, variations=False, comments=False)
	pgn_string = game.accept(exporter)

	logging.info(f"PGN is:\n{pgn_string}")


if __name__ == "__main__":
	main()
