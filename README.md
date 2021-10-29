# How do players experience CEM-driven Non-Player Characters?
A repository for the code related to my Master's thesis.

# Setup
## Prerequisites
- [Conda](https://conda.io/projects/conda/en/latest/index.html)

## Setting up for the first time
1. Clone the repository
2. Create a new Conda environment for the project by running this command in the repository: `conda env create -f environment.yml`
3. Activate the environment with `conda activate cem-experiments`

## Updating the environment
If there have been changes to the environment since the last time you ran it, you can update it by running the following command in the project root: `conda env update --name cem-experiments --file environment.yml --prune`. The `--prune` option removes packages that are not found in the `environment.yml`. Omit the option if you do not wish this behavior.

# Run
Once you have the environment active, run `python main.py`.

# Controls
Move with WASD, skip turn with Q, melee attack the character in front of you with SPACE, ranged attack three tiles in front of you with E, heal the character in front of you with H.