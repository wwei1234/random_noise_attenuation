# random_noise_attenuation

This repository contains research code for random noise attenuation in seismic and ground penetrating radar data. The code includes deep learning models, conventional denoising methods, data processing scripts, visualization utilities, and manuscript-related LaTeX files.

## Project Contents

- `Black Bridge/code/`: Code for the Black Bridge dataset experiments.
- `Marmousi/code/`: Code for Marmousi synthetic data experiments.
- `Stratton/code/`: Code for Stratton dataset experiments.
- `shandong/code/`: Code for Shandong dataset experiments.
- `shengbei/code/`: Code for Shengbei dataset experiments.
- `山东院项目/软件/源码/`: Source code for the iDeRaNS software package.
- `Manuscript/` and `Revised_Manuscript/`: LaTeX manuscript files and references.

## Main Code Types

The repository includes scripts for:

- U-Net and U-Net 3+ model training
- CBAM-enhanced U-Net models
- Noise-to-noise dataset construction
- Random noise synthesis
- Seismic and radar denoising
- F-K and F-X domain processing
- Band-pass filtering
- Quantitative evaluation
- Data visualization

## Notes About Data and Models

Large datasets, trained model weights, generated figures, packed software outputs, and result files are intentionally excluded from Git by `.gitignore`. This keeps the GitHub repository lightweight and avoids GitHub file size limits.

Excluded file types include:

- seismic/radar data: `.sgy`, `.segy`, `.dat`, `.DT1`, `.HD`
- arrays and matrices: `.npy`, `.npz`, `.mat`
- model weights: `.pth`, `.pt`, `.ckpt`
- archives and packaged outputs: `.rar`, `.zip`, `dist/`, `build/`
- generated images and documents: `.png`, `.jpg`, `.pdf`, `.docx`, `.pptx`

## Basic Usage

Install the required Python dependencies according to the script you want to run. Common dependencies include:

```bash
pip install numpy scipy matplotlib torch torchvision segyio scikit-image
```

Then run the target script from the corresponding experiment folder, for example:

```bash
python train.py
python test.py
```

Some scripts contain hard-coded local data paths. Update those paths before running experiments on another machine.

## Repository

GitHub: https://github.com/wwei1234/random_noise_attenuation
