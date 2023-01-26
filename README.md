# Little-Go-Agent

This project is a demonstration of reinforcement learning techniques applied to a simple problem. 
The goal of the project is to train an agent to play a 5x5 grid Little Go game and defeat a random player. 

![LittleGo banner](littleGo.png?raw=true "Little Banner")

## Details
The project includes the following files:

1. `GoGame.py`: This is the main file that runs the simulation and trains the agent.
    - Initializes Q-Learning Agent and Random Player.
    - Simulates "battle" with alternate player turns and calculating score.
    - Calculates winning rate of the agent. 
    
2. `GoAgent.py`: This file contains the implementation of the agent and its Q-Learning Algorithm.
    - Initializes Q-values for initial board state.
    - Contains implementation of player moves and Q-learning.
    - Finds out best possible move to play.
    - Loads and Saves Q-values in the **Qval.txt**.
    
3. `GoBoard.py`: This file contains the implementation of Little-Go environment.
    - Visualizing and resetting the board.
    - Finding neighbors and liberty of the pieces.
    - Removing dead pieces form the board.
    - Calculating scores for the players and checking winner.
    
4.  `GoRandom.py`: This file contains the implementation of a random player.

5.  `Qval.txt`: This file is generated after the agent has learnt and contains Q-Values for each state.

## Note
The project is intended for educational and research purposes.

We hope you enjoy the project and learn something new about reinforcement learning!
