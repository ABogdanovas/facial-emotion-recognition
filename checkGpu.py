import torch, sys
print("python:", sys.executable)
print("torch:", torch.__version__)
print("torch cuda available:", torch.cuda.is_available())
print("torch cuda version:", torch.version.cuda)
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0))
