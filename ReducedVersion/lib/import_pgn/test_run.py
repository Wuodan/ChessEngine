from import_pgn import pgn_to_moves_and_positions


pgn_file = "../../data/pgn_import/3_games.pgn"
[moves_list, positions_list] = pgn_to_moves_and_positions(pgn_file)

print(moves_list)
print(positions_list)
