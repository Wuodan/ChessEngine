import subprocess

import torch


def get_nvcc_version():
	try:
		result = subprocess.run(['nvcc', '--version'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
		if result.returncode == 0:
			print(result.stdout)
		else:
			print(f"Error: {result.stderr}")
	except FileNotFoundError:
		print("nvcc not found. Make sure CUDA is installed and nvcc is in your PATH.")


def get_nvidia_smi():
	try:
		result = subprocess.run(['nvidia-smi'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
		if result.returncode == 0:
			print("nvidia-smi output:\n", result.stdout)
		else:
			print(f"Error: {result.stderr}")
	except FileNotFoundError:
		print("nvidia-smi not found. Make sure NVIDIA drivers are installed and nvidia-smi is in your PATH.")


print(f"torch.__version__={torch.__version__}")
print(f"torch.version.cuda={torch.version.cuda}")
print(f"torch.backends.cudnn.version()={torch.backends.cudnn.version()}")
print(f"torch.cuda.is_available()={torch.cuda.is_available()}")
print()

get_nvcc_version()
print()

get_nvidia_smi()
