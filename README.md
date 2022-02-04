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
Move with WASD, attack the character in front of you or ahead of you with SPACE, heal the character in front of you with H.

Print the empowerment heatmaps with P. This takes a while, your system's memory consumption going up and down is the best indicator of it doing some job.

# Configuration
The program can be configured in the `game_conf.yaml` file.

Comprehensive instructions will be here once the format has established and all options are available through the config file.