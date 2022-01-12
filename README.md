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
The program can be configured by changing the `game_conf.ini` file.
Here's an example:
```
[DEFAULT]
GriddlyDescription = testbed1.yaml
Teams = [[1,2],[3]]
MaxHealth = 2
CEMAgents = [2,3]
EmpowermentPairs = [[[2,2],[2,1],[1,1]],[[3,3],[3,1],[1,1]]]
EmpowermentWeights = [[0.3,0.2,0.9],[0.2,0.2,-1.0]]
AgentActions = [["idle", "move", "heal"],["idle", "move", "heal", "attack"],["idle", "move", "attack"]]
NStep = 1
```
## GriddlyDescription
Which Griddly environment is loaded from `griddly_descriptions` directory. This file includes all the rules and level of the game, including player count.

## Teams
Which players belong to the same team.

## MaxHealth
Max health of each player. Make sure this matches with what is given in Griddly description.

## CEMAgents
Which players are controlled by CEM algorithm.

## Empowerment pairs
For each CEM agent, define which empowerment pairs should be considered. The contents are 1-based player indices. Pair with two of the same index mean empowerment of that agent; if the numbers are different it is transfer empowerment.

## Empowerment weights
For each CEM agent, define the weight of each empowerment pair in its decision making.

## AgentActions
For each agent, give the actions that are available for that player.

## NStep
N-step CEM algorithm is applied.