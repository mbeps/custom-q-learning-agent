This project implements a Q-learning algorithm to teach an agent how to play Pacman. 
The agent learns through experience to navigate the maze, collect food, and avoid ghosts. 
After training, the agent can make intelligent decisions without human intervention.

The implementation is based on the UC Berkeley Pacman AI framework and serves as a practical application of reinforcement learning concepts.

# Implementation (Techniques, Algorithms & Optimisations)

## Core Reinforcement Learning

- **Q-Learning Algorithm**: Standard Q-learning update formula is implemented
- **Dual Exploration Strategy**:
  - ε-Greedy: Takes random actions with probability ε
  - Count-Based Exploration: Adds an exploration bonus inversely proportional to visit frequency
- **Optimistic Initialisation**: Q-values start at positive values to encourage exploration
- **Parameter Adjustment**: Both learning rate (α) and exploration rate (ε) decrease as training progresses

## State Representation

- **Feature Extraction**: Custom abstraction of the game state focusing on relevant information
- **Relational Features**: Direction and distance to food/ghosts rather than absolute positions
- **BFS Pathfinding**: Efficient path calculation to important objects
- **Custom Hashing**: Allows similar states to be recognised as equivalent for better generalisation

## Optimisations

- **DefaultDict Usage**: Efficient automatic initialisation of new state-action pairs
- **Intelligent Tie-Breaking**: Prefers less-explored actions when Q-values are equal
- **Depth-Limited Search**: Prevents excessive computation in pathfinding
- **Type Hinting**: Comprehensive type annotations for better code quality

# Requirements

- Python 3.10 or above
- Optional: Poetry for dependency management

The project uses standard Python libraries and the Pacman framework included in the repository.

# Usage

Clone the repository and navigate to the project directory:
```bash
git clone https://github.com/yourusername/pacman-qlearning.git
cd pacman-qlearning
```

## Using Regular Python
### 1. Run the agent with default training parameters:
```bash
python3 pacman.py -p QLearnAgent -x 2000 -n 2010 -l smallGrid
```

This will:
- Train the agent for 2000 episodes (-x 2000)
- Run 10 additional test episodes (-n 2010)
- Use the small grid layout (-l smallGrid)

### 2. To modify learning parameters:
```bash
python3 pacman.py -p QLearnAgent -x 2000 -n 2010 -l smallGrid -a alpha=0.3,epsilon=0.1,gamma=0.9
```

### Using Poetry

### 1. Install dependencies:
```bash
poetry install
```

### 2. Run the agent:
```bash
poetry run python pacman.py -p QLearnAgent -x 2000 -n 2010 -l smallGrid
```

### 3. Or with custom parameters:
```bash
poetry run python pacman.py -p QLearnAgent -x 2000 -n 2010 -l smallGrid -a alpha=0.3,epsilon=0.1,gamma=0.9
```

## Additional Options

- `-l [layout]`: Change the game layout (e.g., mediumClassic, smallClassic)
- `-k 0`: Run without ghosts
- `-q`: Run in quiet mode without visuals (faster training)
- `-g DirectionalGhost`: Use different ghost behaviours
