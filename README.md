# MAGPO
This repository provides the official implementation of **Multi-Agent Guided Policy Optimization (MAGPO)**, as introduced in our [paper](http://arxiv.org/abs/2505.15418).

Our implementation is based on [Mava](https://github.com/instadeepai/Mava), and follows its concise single-file JAX implementation style. Please refer to the original Mava repository for general infrastructure details and design philosophy.

## 📌 TODO

- [x] Release MAGPO code
- [ ] Release the CoordSum environment used in the paper
- [ ] Release all experimental configurations and results
- [ ] Add support for HAPPO and the heterogeneous version of MAGPO


## 🛠️ Installation

The installation process is the same as in Mava. We recommend using [uv](https://docs.astral.sh/uv/) for dependency management.

```bash
# Clone the repository
git clone https://github.com/instadeepai/Mava.git
cd Mava
# Create a virtual environment and install all dependencies
uv sync
# Activate the virtual environment
source .venv/bin/activate
```

To install with a GPU or TPU aware version of JAX

```bash
uv sync --extra cuda12  # GPU aware JAX
uv sync --extra tpu  # TPU aware JAX
```

Alternatively with pip, create a virtual environment and then:

```bash
pip install -e ".[cuda12]"  # GPU aware JAX (leave out the [cuda12] if you don't have a GPU or are on Mac)
```

For more detailed installation options, including Docker builds, please refer to [Mava's detailed installation guide](https://github.com/instadeepai/Mava/blob/develop/docs/DETAILED_INSTALL.md).

## 🚀 Training
To train a multi-agent system with MAGPO, run one of the system files. For example:
```
python mava/systems/gpo/anakin/rec_magpo.py
```
We use [Hydra](https://github.com/facebookresearch/hydra) for config management. 
Default configurations can be found in `mava/configs/` directory. 
To run on a specific environment, use command-line overrides. Example: training on Level-based Foraging:
```
python mava/systems/gpo/anakin/rec_magpo.py env=lbf
```
Training on RWARE with a specific scenario:
```
python mava/systems/gpo/anakin/rec_magpo.py  env=rware env/scenario=tiny-4ag
```
More examples can be found in [Mava's Quickstart notebook](https://github.com/instadeepai/Mava/blob/develop/examples/Quickstart.ipynb).

## 📖 Citation
If you find this repository or GPO useful in your research, please consider citing our paper:
```bibtex
@article{
}
```
