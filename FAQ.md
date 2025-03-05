# 6CCS3ML1 CW2 FAQ  
*Oana Cocarascu, Helen Yannakoudakis, Zheng Yuan, Adrian Salazar, Ionut Moraru*  

---

## 1. Coursework Overview  
This coursework exercise requires you to implement Q-learning to control Pacman’s movement.  

---

## 2. Frequently Asked Questions  

### 2.1 What do I need to submit?  
Your submission should consist of a single ZIP file. (KEATS will be configured to only accept a single file.) This ZIP file must include a single Python file (`mlLearningAgents.py`). See coursework spec for more details.  

**DO NOT SUBMIT** the `mlLearningAgents.pyc` file! That is the compiled file, not the human-readable file.  

---

### 2.2 Which map should I use to test my agent?  
Your code will be evaluated on `smallGrid` and another secret map. Your agent needs to win **8 out of 10 games** on the `smallGrid`.  

---

### 2.3 I am worried that the uncertainty of the environment will affect my grade. What will you do about that?  
Reinforcement Learning (RL) approaches are specifically designed to behave well in environments with high uncertainty. A well-implemented Q-learning agent will win all **10 games most of the time** (the map is really small, and the win condition is easy to achieve).  

However, to ensure fairness, any agent that does not reach the requested win ratio will be **re-run**. If it does not reach the threshold the second time, then you will lose some marks.  

---

### 2.4 Why are you running it on another grid?  
This is done to check whether your agent’s behaviour is **hard-coded**. Hard-coded behaviour means that the agent follows a predefined set of actions instead of making decisions based on the Q-learning algorithm.  

---

### 2.5 What dimension will this other grid have?  
There are several other layouts in the source code that you can test to see if your Q-learning agent works. The goal is **not** for the agent to win in another grid, but to check if it generalises.  

The grid we’ll test with will be **similar to one of those available** in the source code. We will let the code run for several minutes.  

---

### 2.6 What is the `final()` method?  
When running several games in a row (i.e., using the `-n` parameter), the `__init__()` method is only called at the beginning, when the object is created.  

As such, you will need to use the `final()` method to **reset your values** for the next game to the initial values (if needed).  

---

### 2.7 What sections am I allowed to modify? Is it just the `*** Your code here ***` section, or can I modify the constructor of `QLearnAgent`?  
In addition to `*** Your code here ***`, you can **add code** to the constructor or the `final()` method and **add more helper methods**.  

---

### 2.8 What is the purpose of the `GameStateFeatures` class? Why do I need a wrapper around the state if I can get all the information needed from the state? Can we add more methods in `GameStateFeatures`?  
`GameStateFeatures` acts as a **wrapper**.  

- It provides a **simpler interface** where you extract only the necessary information for your Q-learning agent.  
- The `GameState` may have **more features/methods** than you need.  
- This class encourages students to **carefully select features** rather than using all information from `GameState`.  
- You **can** add any methods that you like in this class.  
- You **must** implement `__eq__` and `__hash__` methods.  

---

### 2.9 Can I remove the random action selection part in `getAction()`?  
Yes, you should **remove** the random action selection and replace it with your **own** code. The random selection is provided as a **starting point** that ensures the code runs.  

---

### 2.10 Can I use `generatePacmanSuccessor`?  
**No.** The Q-learning agent **does not** have direct access to the environment. It cannot query future states—it has to **learn from experience**.  

---

### 2.11 How will `explorationFn` be assessed?  
You can choose your **own approach** for exploration.  

- A common approach is **count-based exploration (`explorationFn`)**.  
- Another option is using **both epsilon-greedy and count-based exploration together**.  

The `explorationFn` function **must** use counts to return a value.  

- **If you only return the utility**, you **will not** get full marks because the function would not be checking the counts.  

---

### 2.12 Can I just implement epsilon-greedy exploration?  
If your solution **only** uses epsilon-greedy, you will **not** receive marks for the **count-based exploration** component.  

---

### 2.13 What libraries can I use?  
You can use the **standard Python libraries**.  

Additionally, you are allowed to use:  
- `numpy`  
- `pandas`  

---

### 2.14 How should my code be styled?  
- Your code should be **commented** and have a **consistent style** throughout.  
- Ensure a **good separation** of tasks across methods and classes.  

---

### 2.15 What is a good coding style?  
This depends on personal preferences, but the most important factor is **readability**.  

Good practices include:  
- **Descriptive variable names**  
- **Consistent use of whitespaces**  
- **Logical organisation** of code into functions and classes  

---

### 2.16 How can I get the maximum grade in the comments part?  
Good comments should:  
- Explain **how different parts of your implementation work**.  
- Provide **high-level explanations** with references to relevant **theory**.  
- If you use functions, **explain their input parameters** and expected outputs.  

---

### **Last Updated:**  
**February 2025**  

---
