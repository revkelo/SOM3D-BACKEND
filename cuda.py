import torch
print("CUDA disponible:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("Usando:", torch.cuda.get_device_name(0))
