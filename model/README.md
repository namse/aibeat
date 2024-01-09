# Install

```bash

# Install cuda 12.1 https://developer.nvidia.com/cuda-12-1-1-download-archive
# Below code is for ubuntu.
wget https://developer.download.nvidia.com/compute/cuda/12.1.1/local_installers/cuda_12.1.1_530.30.02_linux.run
sudo sh cuda_12.1.1_530.30.02_linux.run

# install torch 2.1.0 which supported by tch crate.
pip install torch==2.1.0 torchvision torchaudio

echo 'export LD_LIBRARY_PATH=~/.local/lib/python3.10/site-packages/torch/lib:$LD_LIBRARY_PATH' >> ~/.bashrc
```
