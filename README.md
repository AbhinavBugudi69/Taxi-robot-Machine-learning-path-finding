# Taxi Pathfinding with Q-Learning

## Module: 6ELEN018W - Applied Robotics

## Lecturer: Dr. Dimitris C. Dracopoulos

## Student: Abhinava Sai Bugudi (ID: W1947458)

-----

## Overview

This project implements a **Q-Learning agent** to solve the classic **Taxi-v3 environment** from OpenAI Gym. The agent learns to navigate a taxi efficiently within a grid-world, pick up a passenger, and drop them off at their designated destination. This repository showcases the Q-Learning algorithm, its training process, and performance evaluation through various scenarios. It also includes the generation of a dataset from the Q-table, which can be used for potential future work with neural networks.

-----

## Table of Contents

1.  [Introduction](https://www.google.com/search?q=%23introduction)
2.  [Environment: Taxi-v3](https://www.google.com/search?q=%23environment-taxi-v3)
3.  [Q-Learning Algorithm](https://www.google.com/search?q=%23q-learning-algorithm)
4.  [Project Structure](https://www.google.com/search?q=%23project-structure)
5.  [Setup and Installation](https://www.google.com/search?q=%23setup-and-installation)
6.  [Usage](https://www.google.com/search?q=%23usage)
      * [Training the Agent](https://www.google.com/search?q=%23training-the-agent)
      * [Generating Data for Neural Networks](https://www.google.com/search?q=%23generating-data-for-neural-networks)
      * [Simulating a Random Configuration](https://www.google.com/search?q=%23simulating-a-random-configuration)
      * [Testing Extreme Configurations](https://www.google.com/search?q=%23testing-extreme-configurations)
      * [Plotting Training Results](https://www.google.com/search?q=%23plotting-training-results)
7.  [Results](https://www.google.com/search?q=%23results)
8.  [Future Improvements](https://www.google.com/search?q=%23future-improvements)
9.  [License](https://www.google.com/search?q=%23license)
10. [Contact](https://www.google.com/search?q=%23contact)

-----

## Introduction

Reinforcement Learning (RL) allows an agent to learn by interacting with an environment, aiming to maximize cumulative rewards. This project applies **Q-Learning**, a model-free, off-policy RL algorithm, to the `Taxi-v3` problem. The agent constructs a Q-table, which stores the expected future rewards for each state-action pair, thereby learning an optimal policy.

The agent's decision-making balances **exploration** (trying new actions) and **exploitation** (choosing known best actions) using an $\\epsilon$-greedy strategy. $\\epsilon$ gradually decays, enabling the agent to explore more initially and then focus on optimal actions as it gains experience.

-----

## Environment: Taxi-v3

The `Taxi-v3` environment is a well-known discrete Reinforcement Learning problem from OpenAI Gym.

  * **Grid World:** A 5x5 grid represents the taxi's operational area.
  * **State Space:** The environment's state is defined by the taxi's location (row, column), the passenger's location, and the destination. There are 404 valid states, considering various combinations.
  * **Action Space:** The taxi has 6 possible discrete actions:
      * `0`: Move South
      * `1`: Move North
      * `2`: Move East
      * `3`: Move West
      * `4`: Pickup passenger
      * `5`: Drop off passenger
  * **Rewards:**
      * `+20`: Successful passenger drop-off at the correct destination.
      * `-10`: Invalid pickup or drop-off attempt.
      * `-1`: For each step taken.
  * **Goal:** The primary objective is for the agent to learn the most efficient sequence of actions to pick up and drop off the passenger.

-----

## Q-Learning Algorithm

Q-Learning learns an action-value function, $Q(s, a)$, which estimates the total expected reward for taking action $a$ in state $s$ and following an optimal policy thereafter.

The Q-table is iteratively updated using the Bellman equation:

$Q(s, a) \\leftarrow Q(s, a) + \\alpha [R\_{t+1} + \\gamma \\max\_{a'} Q(s', a') - Q(s, a)]$

Where:

  * $Q(s, a)$: Current Q-value for state $s$ and action $a$.
  * $\\alpha$ (learning rate): Dictates how much new information influences the existing Q-value (range: 0 to 1).
  * $R\_{t+1}$: The immediate reward received from the environment.
  * $\\gamma$ (discount factor): Balances the importance of immediate vs. future rewards (range: 0 to 1).
  * $\\max\_{a'} Q(s', a')$: The maximum Q-value achievable from the next state $s'$.
  * $s'$: The state reached after performing action $a$.

-----

## Project Structure

```
.
├── taxi_q_learning.py   # Main script: Q-Learning agent, training, and evaluation
├── q_learning_data.csv  # Generated CSV: Q-table derived data (output after training)
├── README.md            # This documentation file
└── requirements.txt     # Python package dependencies
```

-----

## Setup and Installation

To run this project, ensure you have Python installed. Using a virtual environment is highly recommended.

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/AbhinavBugudi69/Taxi-Pathfinding-Q-Learning.git # Update with your actual repo URL if different
    cd Taxi-Pathfinding-Q-Learning
    ```

2.  **Create and activate a virtual environment:**

    ```bash
    python -m venv venv
    # On Windows: .\venv\Scripts\activate
    # On macOS/Linux: source venv/bin/activate
    ```

3.  **Install dependencies:**
    Create a `requirements.txt` file in your project's root directory with the following content:

    ```
    gymnasium
    numpy
    pandas
    matplotlib
    ```

    Then install them:

    ```bash
    pip install -r requirements.txt
    ```

-----

## Usage

Execute the `taxi_q_learning.py` script to initiate the agent's training, data generation, simulation, and result plotting.

```bash
python taxi_q_learning.py
```

### Training the Agent

The `TaxiV3QLearningAgent` class encapsulates the training logic. Key hyperparameters are configured in the `if __name__ == "__main__":` block:

  * `num_train_episodes`: Number of training iterations.
  * `alpha` (learning rate): Impact of new learning on existing knowledge.
  * `gamma` (discount factor): Preference for immediate vs. future rewards.
  * `epsilon`: Initial exploration vs. exploitation balance.
  * `epsilon_min`: The lowest exploration rate allowed.
  * `epsilon_decay`: Rate at which exploration decreases.

### Generating Data for Neural Networks

Upon completion of training, an `q_learning_data.csv` file is created. This CSV maps environment states (encoded taxi, passenger, and destination locations) to the optimal action determined by the trained Q-table. This dataset can serve as ground truth for training a supervised learning model, like a Neural Network, to predict optimal actions.

### Simulating a Random Configuration

The `process_random_config` function demonstrates the trained agent's behavior in a new, random scenario. It outputs the initial state's details and the sequence of actions taken, rewards received, and total steps until completion.

### Testing Extreme Configurations

The agent's performance is rigorously tested against two "extreme" pre-defined scenarios:

1.  Taxi starts farthest from the passenger.
2.  Passenger is picked up but farthest from the destination.

For each, the agent runs multiple test episodes (defaulting to 100), providing average rewards, steps, and penalties, which indicate the robustness of the learned policy.

### Plotting Training Results

Visual insights into the learning process are provided by two plots generated at the end of the script:

1.  **Rewards Per Episode:** Illustrates how the total reward evolves over training episodes.
2.  **Steps Per Episode:** Shows the number of steps taken by the agent in each training episode.

These plots should ideally show an increasing trend for rewards and a decreasing trend for steps, signifying effective learning.

-----

## Results

Executing `taxi_q_learning.py` will produce console output similar to this:

```
!Training the Q-learning agent!
!Generating data for neural network training!
Data saved to 'q_learning_data.csv'.

 Analyzing extreme configuration 1: {'taxi_row': 0, 'taxi_col': 0, 'passenger_index': 3, 'destination_index': 0}
Testing configuration: {'taxi_row': 0, 'taxi_col': 0, 'passenger_index': 3, 'destination_index': 0}
  Average Reward: 8.31
  Average Steps: 12.69
  Average Penalties: 0.00

 Analyzing extreme configuration 2: {'taxi_row': 4, 'taxi_col': 4, 'passenger_index': 0, 'destination_index': 3}
Testing configuration: {'taxi_row': 4, 'taxi_col': 4, 'passenger_index': 0, 'destination_index': 3}
  Average Reward: 8.03
  Average Steps: 12.97
  Average Penalties: 0.00

 Starting simulation for a random configuration :
  Taxi Initial Location: (row=2, col=4)
  Passenger Location: 0
  Destination: 3
Step 1: Action=3, Reward=-1
Step 2: Action=3, Reward=-1
Step 3: Action=3, Reward=-1
Step 4: Action=1, Reward=-1
Step 5: Action=1, Reward=-1
Step 6: Action=3, Reward=-1
Step 7: Action=4, Reward=-1
Step 8: Action=2, Reward=-1
Step 9: Action=0, Reward=-1
Step 10: Action=0, Reward=-1
Step 11: Action=2, Reward=-1
Step 12: Action=2, Reward=-1
Step 13: Action=0, Reward=-1
Step 14: Action=0, Reward=-1
Step 15: Action=5, Reward=20

Simulation complete. Total Reward: 6, Steps: 15
```

The accompanying plots will graphically illustrate the agent's learning curve, showing rewards generally increasing and steps decreasing over training episodes, indicating successful policy convergence and efficient problem-solving. A notable aspect is the consistent average penalties of `0.00` across extreme test configurations, confirming the agent's ability to avoid illegal actions.

-----

## Future Improvements

  * **Deep Q-Network (DQN) Implementation:** Extend this project by training a Neural Network using the generated `q_learning_data.csv`. This would allow for handling much larger or continuous state spaces where explicit Q-tables become unmanageable.
  * **Advanced Hyperparameter Tuning:** Implement automated hyperparameter optimization techniques (e.g., Grid Search, Random Search, or Bayesian Optimization) to systematically find the most effective `alpha`, `gamma`, and `epsilon` decay schedules.
  * **Custom Environment Design:** Apply the Q-Learning framework to a more complex, custom-designed grid-world or a simulated robotics environment to test its adaptability.
  * **Real-time Visualization:** Enhance the simulation by adding real-time visual rendering of the taxi's movements using `render_mode="human"` or external visualization libraries.
  * **Robust Evaluation:** Introduce more advanced evaluation metrics beyond simple averages, such as success rates, episode completion times, and convergence speed comparisons.

-----

## License

This project is open-source and available under the [MIT License](LICENSE.md).

-----

## Contact

For any questions, collaboration opportunities, or discussions about this project, please feel free to connect:

**Abhinava Sai Bugudi**

  * **Email**: abhinavasaibugudi04@gmail.com
  * **LinkedIn**: [bugudi-abhinava-sai](https://www.google.com/search?q=https://www.linkedin.com/in/bugudi-abhinava-sai/)
  * **GitHub**: [AbhinavBugudi69](https://www.google.com/search?q=https://github.com/AbhinavBugudi69)

-----