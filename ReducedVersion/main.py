from ReducedVersion.lib.training.training import run_training
from lib.import_pgn.import_pgn import pgn_to_moves_and_positions
from lib.encode_decode.encode import encode_moves_and_positions


def main():
	# create_main_screen()
	default_file = "data/pgn_import/3_games.pgn"
	pgn_file = input(f"Please enter the path to the PGN file ({default_file}): ").strip() or default_file
	print(f"Importing PGN file {pgn_file}")
	parsed_games = pgn_to_moves_and_positions(pgn_file)
	print(f"Found {len(parsed_games)} games")

	print(parsed_games)

	moves_and_positions = encode_moves_and_positions(parsed_games)

	run_training(moves_and_positions, output_folder='data/savedModels')


if __name__ == "__main__":
	main()
