import logging

from lib.encode_decode.encode import encode_moves_and_positions
from lib.import_pgn.import_pgn import pgn_to_moves_and_positions
from lib.training.training import run_training

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S.%f')


def main() -> None:
	default_file = "data/pgn_import/3_games.pgn"
	pgn_file = input(f"Please enter the path to the PGN file ({default_file}): ").strip() or default_file

	logging.info(f"Importing PGN file {pgn_file}")

	parsed_games = pgn_to_moves_and_positions(pgn_file)
	logging.info(f"Found {len(parsed_games)} games")

	moves_and_positions = encode_moves_and_positions(parsed_games)
	logging.info("Encoded all games")

	output_folder = 'data/savedModels'
	run_training(moves_and_positions, output_folder=output_folder)

	logging.info("Done")


if __name__ == "__main__":
	main()
