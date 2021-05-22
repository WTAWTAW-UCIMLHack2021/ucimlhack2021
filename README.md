# ucimlhack2021
Our team's submission for the 2021 UCI Machine Learning Hackathon

## Setup

1. Make sure you have [Anaconda](https://www.anaconda.com/products/individual) installed
2. In a terminal where you can run Anaconda commands, run `conda env create --file environment.yaml` to make the Conda virtual environment
3. Next, activate the conda environment `conda activate uciml`
4. Finally, to link the virtual environment with Jupyter Notebook/Lab, run `python -m ipykernel install --user --name=uciml`
5. Start JupyterLab by running `jupyter lab`

All done!

To update the conda environment, run `conda env update --name uciml --file .\environment.yaml`