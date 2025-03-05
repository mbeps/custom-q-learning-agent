# 6CCS3ML1 (Machine Learning)  
## Coursework 2  
### (Version 1.7)  

## 1 Overview  

For this coursework, you will have to implement the Q-learning algorithm. Your code will again be controlling Pacman, in the classic game, and the Q-learning algorithm will be helping Pacman choose how to move. Your Q-learning algorithm should be able, once it has done its learning, to play a pretty good game, and its ability to play a good game will be part of what we assess.  

This coursework is worth 10% of the marks for the module.  

**Note:** Failure to follow submission instructions will result in a deduction of 10% of the marks you earn for this coursework.  

### Figure 1: The smallGrid version of Pacman.  

---

## 2 Getting started  

### 2.1 Start with Pacman  

The Pacman code that we will be using for the 6CCS3ML1 coursework was originally developed at UC Berkeley for their AI course. Note that we will not be doing any of their projects.  

You should:  

1. Download `pacman-cw2.zip` from KEATS.  
2. Save that file to your account at KCL (or to your own computer) in a new folder.  
3. Unzip the archive. This will create a folder `pacman-cw2`.  
4. From the command line (you will need to use the command line in order to use the various options), switch to the folder `pacman-cw2`.  
5. Now type:  

   ```bash
   python3 pacman.py -p RandomAgent -n 10 -l smallGrid
   ```

   and watch it run.  

This command illustrates a few aspects of the Pacman code that you will need to understand:  

- `-n 10` runs the game 10 times.  
- `-l smallGrid` runs the very reduced game you see in Figure 1.  

This is not a very interesting game for a human to play, but it is moderately challenging for a reinforcement learning program to learn to play.  

- `-p RandomAgent` tells the `pacman.py` code to let Pacman be controlled by an object that is an instance of a class called `RandomAgent`.  

The program then searches through all files with a name that ends in `Agents.py` looking for this class. If the class isn’t in an appropriately named file, you will get the error:  

```python
Traceback (most recent call last):
File "pacman.py", line 679, in <module>
    args = readCommand( sys.argv[1:] ) # Get game components based on input
File "pacman.py", line 541, in readCommand
    pacmanType = loadAgent(options.pacman, noKeyboard)
File "pacman.py", line 608, in loadAgent
    raise Exception('The agent ' + pacman + ' is not specified in any *Agents.py.')
```

In this case, the class is found in `sampleAgents.py`.  

`RandomAgent` just picks actions at random. When there is no ghost, it will win a game eventually by eating all the food, but when a ghost is present, `RandomAgent` dies pretty quickly.  

---

### 2.2 Towards an RL Pacman  

Your task is to write code to learn how to play this small version of Pacman. To get you started, we have provided a skeleton of code that will do this, found in:  

```plaintext
mlLearningAgents.py
```

This file contains a class `QLearnAgent`, and that class includes several methods.  

#### `__init__()`  
- This is the constructor for `QLearnAgent`. It is called by the game when the game starts up (because the game starts up the learner).  
- The version of `__init__()` in `QLearnAgent` allows you to pass parameters from the command line. Some of these you know from the lectures:  

  - `alpha`, the learning rate  
  - `gamma`, the discount rate  
  - `epsilon`, the exploration rate  

  and you will use them in your implementation. The other:  

  - `numTraining`  

  allows you to run some games as training episodes and some as real games.  

#### `getAction()`  
This function is called by the game every time that it wants Pacman to make a move (which is every step of the game).  

- This is the core of code that controls Pacman.  
- It has access to information about the position of Pacman, the position of ghosts, the location of food, and the score.  
- The only bit that maybe needs some explanation is the food. Food is represented by a grid of letters, one for each square in the game grid.  

  - `F` (False) means there is no food in that square.  
  - `T` (True) means there is food in that square.  

The main job of `getAction()` is to decide what move Pacman should make, so it has to return an action. The current code shows how to do that but just makes the choice randomly.  

#### `final()`  
This function is called by the game when Pacman has been killed by a ghost or when Pacman eats the last food (and so wins).  

---

## 3 What you have to do (and what you aren’t allowed to do)  

### 3.1 Implementation  

You should fill out `QLearnAgent` with code that performs Q-learning. You should implement your code in `mlLearningAgents.py`.  

Because of the way that Pacman works, your learner needs to include two separate pieces:  

1. Something that learns, adjusting Q-values based on how well the learner plays the game.  
2. Something that chooses how to act. This decision can be based on the Q-values but also has to make sure that the learner does enough exploring.  

Your code will be tested with:  

```bash
python3 pacman.py -p QLearnAgent -x 2000 -n 2010 -l smallGrid
```

Your `QLearnAgent` needs to win around 8 of the games (which is very achievable).  

---

## 4 What you have to hand in  

Your submission should consist of a single ZIP file.  

The ZIP file **must** include a single Python file (`mlLearningAgents.py`).  

The ZIP file **must** be named:  

```plaintext
cw2-<lastname>-<firstname>.zip
```

**Do not include the entire `pacman-cw2` folder.**  

Submissions that do not follow these instructions will lose marks.  

---

## 5 How your work will be marked  

There will be six components of the mark for your work:  

### **1. Functionality**  
- Your code must run with `python3 pacman.py -p QLearnAgent -x 2000 -n 2010 -l smallGrid`.  
- We will look for evidence of Q-learning in your code.  
- Your `QLearnAgent` should win **8 out of 10 games**.  

### **2. Form**  
- Your code should follow standard good practice in software development.  
- Your work will be partly assessed by the comments you provide in your code.  

---

## **Version list**  

- **Version 1.0:** March 17th 2019  
- **Version 1.1:** March 27th 2019  
- **Version 1.2:** March 13th 2020  
- **Version 1.3:** February 22nd 2021  
- **Version 1.4:** March 8th 2022  
- **Version 1.5:** February 27th 2023  
- **Version 1.6:** February 20th 2024  
- **Version 1.7:** February 26th 2025  