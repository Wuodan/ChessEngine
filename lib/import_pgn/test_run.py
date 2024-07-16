from import_pgn import pgn_to_moves_and_positions


pgn_file = "../../data/pgn_import/3_games.pgn"
parsed_games = pgn_to_moves_and_positions(pgn_file)

print(parsed_games)
