import torch

print("CUDA tersedia:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("Nama GPU yang akan dipakai:", torch.cuda.get_device_name(0))
else:
    print("GPU belum terdeteksi, sistem masih membaca CPU!")