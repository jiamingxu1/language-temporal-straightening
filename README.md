# The human language processing system straightens natural speech

Code accompanying:

**Xu et al. — “The human language processing system straightens natural speech.”**

This repository contains code for the fMRI, WavLM, and simulation analyses reported in the manuscript.

## Usage

### 1. Clone the repository

```bash
git clone https://github.com/jiamingxu1/language-temporal-straightening.git
cd language-temporal-straightening
```

### 2. Create a Python environment

Python 3.11 is recommended.

```bash
conda create -n language-straightening python=3.11 -y
conda activate language-straightening
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Download the fMRI data

The anonymized fMRI dataset is large and is therefore hosted separately on Box rather than in this repository. The fMRI data required to reproduce the analyses are available here:

https://utexas.box.com/s/ca5ydunq876o5ru7u8yhf3yyt9l77mad

Download the dataset and place the downloaded `fsmat/`, `mri/`, `rois/`, and `surfaces/` folders inside the repository's `data/` directory.

The resulting directory structure should be:

```text
data/
├── audios/
├── fsmat/
├── mri/
├── rois/
└── surfaces/
```

The audio files used for the WavLM analyses are already included in `data/audios/`.

The `surfaces/` directory contains the pycortex surface databases and transforms used for the fMRI analyses, including the `fsaverage` template used for the group-level cortical map.

### 5. Run the analysis notebooks

The notebooks are organized by analysis type under `analyses/`.

#### AR(1) and curvature simulations

`analyses/simulations/AR-curv.ipynb`

Simulates AR(1) population trajectories across a range of autoregressive coefficients and measurement-noise levels and evaluates direct and indirect curvature estimation.

Supporting functions are contained in:

`src/AR_sim_core.py`

`analyses/simulations/timescale_curv.ipynb`

A simple two-dimensional simulation illustrating how signal timescale and measurement noise affect trajectory curvature.

#### fMRI analyses

`analyses/fmri/fmri_main_analysis.ipynb`

Runs analyses for an example participant, including direct and indirect trajectory-curvature estimates across cortical regions and voxelwise visualization of implied curvature.

`analyses/fmri/fmri_group_level.ipynb`

Runs the group-level analyses across participants and stories, including statistical analyses, temporal-downsampling analyses, voxel-count and SNR controls, and the group-average cortical map of implied trajectory curvature.

Reusable fMRI analysis functions are contained in:

`src/fmri_core.py`

#### WavLM analyses

`analyses/wavlm/wavLM.ipynb`

Runs the WavLM analyses for natural speech, stimulus perturbation controls, randomly initialized models, comparison of direct and indirect curvature estimates, and temporal-downsampling analyses.

The pretrained WavLM-large model (`microsoft/wavlm-large`) is downloaded automatically from Hugging Face the first time this notebook is run.

## Repository structure

```text
analyses/
├── simulations/
│   ├── AR-curv.ipynb
│   └── timescale_curv.ipynb
├── fmri/
│   ├── fmri_main_analysis.ipynb
│   └── fmri_group_level.ipynb
└── wavlm/
    └── wavLM.ipynb

spiking/
├── geometry/
│   ├── compute_curvature.m
│   ├── compute_trajectory_pixel.m
├── statistics/
│   ├── compute_CMP.m
│   ├── compute_VST.m
│   ├── epl_kernel.m
│   ├── mean_var_CMP.m
│   ├── rbf_kernel.m
└── main_simulation.m

src/
├── AR_sim_core.py
└── fmri_core.py

data/
├── audios/
├── fsmat/
├── mri/
├── rois/
└── surfaces/

requirements.txt
```

## Spiking-neuron simulations

Code for the spiking-neuron population simulations are in the `spiking/` directory. 
