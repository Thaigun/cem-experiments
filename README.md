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

# Running the tests
Test runner is designed to be run on shared resources: It automatically scales so that it uses 30% of memory or 50% of processing power that is not used by other processes. This assumes that the server it's running on has a lot of memory; one test instance can take up to 11GB. If you do not want to run parallel tests, you can set `PARALLEL=False` in `main.py`.

## Setup
The results are saved in Firebase Realtime Database. You have to create a new database where the experiment results are stored.
1. Create a new Firebase project and a Realtime Database in it.
2. In Firebase navigate through Project settings -> Service Accounts -> Firebase Admin SDK -> Generate new private key.
3. Save the key in this project folder `cert/`
4. Copy the file `.env.example` and name it `.env`
5. Set value `FIREBASE_KEY_FILE` in `.env` to the filename of your new key file.
6. Set value `FIREBASE_DATABASE_URL` as the URL of your Firebase Realtime database.

## Running tests
Call `python main.py` to run the tests. Mind that this can be slow. A single test run may take from 40 seconds to 30 minutes or more. I recommend you have hundreds of GB of memory and you run the tests for a couple of days. Alternatively, speed up the algorithm with a touch of quality programming.

## Included results
In the `results/` folder, there are results from two different experiments, one with generated action sets and one with fixed.

# Plotting the results
In Firebase console, select the object that is named after whatever you have in your `.env` file as a value for `DB_ROOT`. Download the results as a JSON-file and save the file in `results/`. Run `python result_handler.py` and follow the instructions in the console.

## Replaying individual game runs
Make sure that `PLAYBACK_ROOT` in `.env` matches the name of `DB_ROOT`. Then, run `python play_back_tool.py` and follow instructions.

# Play with keyboard
Once you have the environment active, run `python keyboard_game.py` to test the environment manually. See the console for controls. In the file `keyboard_game.py`, you can change the value of variable `USE_CONF`. The configuration with the given name will be loaded from `game_conf.yaml`. 

## Configuration
The program can be configured in the `game_conf.yaml` file. See the configuration file for examples of how to define a new test environment.

# Notes about creating new environments
In some places the code assumes custom shaders to show health bars of agent. To avoid errors, include `ObjectVariables` called `health` and `max_health` in the Griddly environment. If you want to show health bars, add similarly named variables to the agent object (note that the object which needs the healthbar should be alphabetically the first object, use "avatar" or "agent"). 

    ...
    Environment:
        Observers:
            Sprite2D:
            Shader:
                ObjectVariables: [ health, max_health ]
    ...
    Objects:
  - Name: avatar
    MapCharacter: P
    Variables:
      - Name: health
        InitialValue: 1
      - Name: max_health
        InitialValue: 1
    ...