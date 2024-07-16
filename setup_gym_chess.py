import logging
import re
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')


def replace_np_int() -> None:
	pattern = r'\bnp\.int\b'
	files_to_fix = ["venv/Lib/site-packages/gym_chess/alphazero/board_encoding.py"]
	for file_path in files_to_fix:
		if Path(file_path).is_file():
			with open(file_path, 'r') as file:
				content = file.read()

			if re.search(pattern, content):
				# Replace 'np.int' with 'int'
				new_content = re.sub(pattern, 'int', content)

				with open(file_path, 'w') as file:
					file.write(new_content)

				logging.info(f"Replaced 'np.int' with 'int' in file {file_path}")

			else:
				logging.info(f"Nothing to replaced in file {file_path}")


def main():
	replace_np_int()


if __name__ == "__main__":
	main()
