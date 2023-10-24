import torch

print(f"The torch version is: {torch.__version__}")
print(f"The torch CUDA version is: {torch.version.cuda}")
print(f"The torch cuDNN version is: {torch.backends.cudnn.version()}")
print(f"Is CUDA available for torch?: {torch.cuda.is_available()}")
print(f"Number of CUDA devices available: {torch.cuda.device_count()}")

if torch.cuda.is_available():
    print(f"Name of the first CUDA device: {torch.cuda.get_device_name(0)}")
    print(f"Current active CUDA device: {torch.cuda.current_device()}")
    print(f"Information on the first CUDA device: {torch.cuda.device(0)}")
    print(f"Information on 'cuda:0' device: {torch.cuda.device('cuda:0')}")
    print(f"Information on the current active CUDA device: {torch.cuda.device(torch.cuda.current_device())}")
else:
    print("CUDA is not available on this machine.")
