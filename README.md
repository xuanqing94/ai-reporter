## AI Reporter

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
