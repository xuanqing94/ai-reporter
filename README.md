## Investigating Heterogeneities of Live Mesenchymal Stromal Cells Using AI-based Label-free Imaging


### Hardware requirements
Recommended hardware: 2 X 2080Ti or higher, 64GiB memory or higher, CPU with >= 8 cores. 
Nvidia CUDA version: >= 10.2
Nvidia CUDNN version: >= 8.0

### Install
Our software is cross-platform, it will work on Linux (tested), Windows and macOS.

Install dependencies (with pip and anaconda or miniconda):
1. First, create a clean conda environment with python3.7 installed.
```bash
conda create --name ai_reporter python=3.7
conda activate ai_reporter
```

2. Second, install python packages
```bash
pip install numpy scipy scikit-learn torch torchvision imageio tifffile imagecodec opencv-python
```

3. Then, install our software with
```bash
cd code
pip install -e .
```

### Run our example
```bash
bash train_example.sh
```

### Acknowledgement
> We thank Takuya Matsumoto, Eri Harada, Alex Hofmann, Gregory R. Johnson and Roy Wallman for insightful discussions. This work was supported by the UCLA SPORE in Prostate Cancer grant (P50 CA092131), the Eli and Edythe Broad Center of Regenerative Medicine and Stem Cell Research at UCLA and California NanoSystems Institute at UCLA Planning Award grant and the National Science Foundation grant (IIS-1901527).

> Source code is modified from [PyTorch-pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix)
