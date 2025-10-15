# Assignment 1: Simple Reflex Agent

**Course:** COMP 237 - Artificial Intelligence  
**Topic:** Agent-based systems and environment simulation  
**Completed:** September 2022

---

## 📝 Assignment Overview

Implementation of a **simple reflex agent** (BlindDog) that navigates an environment (Park) using percept-action mapping. Demonstrates core AI concepts including agent architecture, environment simulation, and object-oriented design.

---

## 🎯 Learning Objectives

- Implement agent-based systems using OOP principles
- Design environment simulations with state management
- Create percept-action mappings for autonomous behavior
- Apply inheritance and polymorphism in AI contexts

---

## 🏗️ System Architecture

### **Core Classes:**
```
Thing (base class)
├── Agent
│   └── BlindDog (custom agent)
├── Food
├── Water
├── Tree
└── Person

Environment (base class)
└── Park (custom environment)
```

### **Agent Behavior:**

| Percept | Action |
|---------|--------|
| Food detected | Eat |
| Water detected | Drink |
| Person detected | Bark |
| Nothing detected | Move down |

---

## 🚀 How to Run

### Prerequisites
- Python 3.8 or higher
- No external dependencies required

### Execution
```bash
# Navigate to assignment folder
cd 00_AI_Fundamentals/Assignment1-Agents

# Run the agent simulation
python Matheus_agent.py
```

### Expected Output

The BlindDog agent will:
1. Start at location 1
2. Move through the park (locations 1-12)
3. Eat food at locations 5 and 9
4. Drink water at location 7
5. Bark at persons at locations 3 and 12
6. Terminate after 18 steps or when resources depleted

---

## 📊 Implementation Details

### **Requirements Implemented:**

1. ✅ **Requirement 1:** Added second food item at location 9
2. ✅ **Requirement 2:** Created `Person` class extending `Thing`
3. ✅ **Requirement 3:** Instantiated two Person objects (Matheus at location 3, Teixeira at location 12)
4. ✅ **Requirement 4:** Implemented bark action when agent encounters a Person
   - Modified `execute_action()` in Park class
   - Updated `is_done()` termination condition
   - Extended `program()` function with Person detection
5. ✅ **Requirement 5:** Executed simulation for 18 steps

### **Key Components:**

**Agent (BlindDog):**
- `location`: Current position in park (integer)
- `eat(thing)`: Consumes food and returns True
- `drink(thing)`: Consumes water and returns True
- `bark(thing)`: Responds to person and returns True
- `movedown()`: Increments location by 1

**Environment (Park):**
- `percept(agent)`: Returns list of Things at agent's location
- `execute_action(agent, action)`: Updates environment based on action
- `is_done()`: Checks termination conditions (no food/water/people or dead agents)

**Program Function:**
- Receives percepts from environment
- Returns appropriate action based on percept type
- Default action: "move down"

---

## 🔍 Code Structure
```
Assignment1-Agents/
├── Matheus_agent.py                      # Main implementation
├── Assignment Agents_ wdemo.pdf          # Assignment instructions
├── MatheusTeixeira_COMP237assignment1.docx  # Analysis report with class diagram
├── Drawing1.vsdx                         # Visio class diagram
├── 2022-09-22 15-01-20.mp4              # Demo video
├── requirements.txt                      # Dependencies (none required)
├── environment.yml                       # Conda environment (optional)
└── README.md                            # This file
```

---

## 📈 Results & Analysis

**Simulation Output:**
- Agent successfully navigated all 18 steps
- All resources (food, water, persons) were encountered and handled appropriately
- Termination condition met after all interactive objects processed

**OOP Principles Demonstrated:**
- **Inheritance:** BlindDog extends Agent, Park extends Environment
- **Polymorphism:** Method overriding (`execute_action`, `is_done`)
- **Encapsulation:** Agent state managed internally
- **Abstraction:** Environment provides abstract interface for agent interaction

---

## 🎓 Key Takeaways

1. **Agent Architecture:** Separation of percept, decision-making (program), and action
2. **Environment Simulation:** State management and object lifecycle
3. **Percept-Action Mapping:** Direct sensory input to behavioral output
4. **OOP Design Patterns:** Effective use of inheritance and composition

---

## 📚 References

- Assignment based on AI textbook agent framework
- Simple reflex agent: no internal state or history
- Percept → Action mapping without learning or planning

---

## 👨‍💻 Author

**Matheus Teixeira**  
Student ID: 301236904  
COMP 237 - Section 002  
September 2022

---

## 📝 Notes

This implementation uses only Python standard library (`collections`, `numbers`). No external dependencies required, making it easy to run on any Python 3.8+ installation.

For detailed analysis including class diagrams and execution screenshots, see `MatheusTeixeira_COMP237assignment1.docx`.