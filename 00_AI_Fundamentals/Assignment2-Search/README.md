# Assignment 2: Search Algorithms

Implementation of uninformed and informed search algorithms in Python for COMP 237 (AI Fundamentals).

## 📋 Overview

This assignment implements and compares various search algorithms:
- **Breadth-First Search (BFS)** - Finding shortest paths in social networks
- **Uniform Cost Search (UCS)** - Cost-optimal pathfinding
- **Greedy Best-First Search** - Heuristic-guided search
- **A* Search** - Optimal informed search

## 🎯 Assignment Objectives

1. **Exercise #1**: Social network graph search using BFS
2. **Exercise #2**: Campus navigation comparing UCS vs Greedy vs A*
3. **Exercise #3**: 8-puzzle heuristic analysis (written responses only)

## 📁 Project Structure

```
Assignment2-Search/
├── Exercise#1_Matheus/
│   └── Breadth_First_Search/
│       ├── BFS_Matheus.py          # BFS implementation
│       ├── GraphData.py            # Social network graph
│       ├── Node.py                 # Tree node structure
│       ├── State.py                # State representation
│       └── TreePlot.py             # Graph visualization
│
└── Exercise#2_Matheus/
    ├── Astar/                      # A* algorithm
    ├── Greedy_search_code/         # Greedy Best-First
    └── UCS/                        # Uniform Cost Search
        ├── [Algorithm].py
        ├── NavigationData.py       # Campus map data
        ├── Node.py
        ├── State.py
        └── TreePlot.py
```

## 🚀 Features

### Exercise #1: BFS Social Network
- Find shortest introduction path between students
- Bidirectional graph representation
- Visual tree exploration with color coding:
  - 🔴 Red: Current node being explored
  - 🔵 Blue: Frontier nodes (in queue)
  - 🟢 Green: Visited/explored nodes
- Error handling for invalid inputs

### Exercise #2: Search Algorithm Comparison
- **Campus Navigation**: Find path from Bus Stop to AI Lab
- **Distance-based costs**: Using Euclidean distance
- **Heuristic evaluation**: Straight-line distance to goal
- **Performance metrics**: Nodes expanded, path length, search depth

## 🛠️ Technical Implementation

### Data Structures
- **Queue**: FIFO for BFS
- **Priority Queue**: For UCS, Greedy, A*
- **Dictionary**: Graph adjacency list
- **List**: Visited nodes tracking

### Key Algorithms

**BFS (Breadth-First Search)**
```
1. Start from initial node
2. Add to queue and mark visited
3. Dequeue node, check if goal
4. Add unvisited neighbors to queue
5. Repeat until goal found or queue empty
```

**UCS (Uniform Cost Search)**
- Priority: Cumulative path cost from root
- Guarantees optimal solution

**Greedy Best-First**
- Priority: Heuristic (distance to goal)
- Fast but not guaranteed optimal

**A* Search**
- Priority: f(n) = g(n) + h(n)
- g(n): Cost from start
- h(n): Heuristic to goal
- Optimal and complete

## 📊 Algorithm Comparison

| Algorithm | Optimality | Completeness | Time Complexity | Space Complexity |
|-----------|------------|--------------|-----------------|------------------|
| BFS       | Yes*       | Yes          | O(b^d)          | O(b^d)           |
| UCS       | Yes        | Yes          | O(b^(C*/ε))     | O(b^(C*/ε))      |
| Greedy    | No         | No           | O(b^m)          | O(b^m)           |
| A*        | Yes**      | Yes          | O(b^d)          | O(b^d)           |

*BFS optimal for unweighted graphs  
**A* optimal with admissible heuristic

### Complexity Notation Glossary

**Variables:**
- **b** = Branching factor (average number of children per node)
- **d** = Depth of shallowest solution
- **m** = Maximum depth of search tree
- **C*** = Cost of optimal solution
- **ε** (epsilon) = Minimum cost of any action (smallest step cost)

**Examples:**
```
If b=3 (each node has 3 children) and d=4:
- BFS explores: 3^0 + 3^1 + 3^2 + 3^3 + 3^4 = 121 nodes (worst case)

If b=10 and d=5:
- BFS explores: up to 111,111 nodes
- This grows exponentially!
```

**Practical Interpretation:**
- **O(b^d)**: Exponential - doubles with each level deeper
- **O(b^(C*/ε))**: Depends on solution cost vs step size
- **O(b^m)**: Can explore entire tree if no solution

**When to Use Each:**

| Scenario | Best Algorithm | Why |
|----------|----------------|-----|
| Unweighted graph (social network) | **BFS** | Finds shortest path, simple |
| Different edge costs (road map) | **UCS** | Guarantees cheapest path |
| Need fast answer, optimal not critical | **Greedy** | Faster, uses heuristic |
| Need optimal + efficient | **A*** | Best performance with good heuristic |

## 🎨 Visualization

All algorithms generate visual search trees using Graphviz:
- **Node labels**: State name + heuristic/cost value
- **Color coding**: Exploration status
- **Tree structure**: Parent-child relationships
- **Final path**: Highlighted in red

## ⚙️ Installation

### Prerequisites
- Python 3.11
- Graphviz executable (required by pydot)

### Install Graphviz

**Windows**:
```bash
# Download from: https://graphviz.org/download/
# Add to PATH: C:\Program Files\Graphviz\bin
```

**macOS**:
```bash
brew install graphviz
```

**Linux**:
```bash
sudo apt-get install graphviz
```

### Install Python Dependencies
```bash
pip install -r requirements.txt
```

Or using conda:
```bash
conda env create -f environment.yml
conda activate ai-search
```

## 🏃 Usage

### Exercise #1: BFS Social Network

```python
# Run BFS to find path
python BFS_Matheus.py

# Example calls in code:
BFS_Matheus("connections", "Dolly", "Matheus")
BFS_Matheus("connections", "George", "Bob")
```

**Output**:
- Console: Path from start to goal
- Visual: Search tree diagram (graph.png)

### Exercise #2: Search Comparison

```python
# Run each algorithm
cd Exercise#2_Matheus/UCS
python UCS.py

cd ../Greedy_search_code
python GreedySearch.py

cd ../Astar
python AStar.py
```

**Compare Results**:
- Number of nodes expanded
- Search depth
- Path cost
- Algorithm efficiency

## 📝 Key Findings

### Exercise #2 Analysis

**Campus Map**: Bus Stop → AI Lab

**UCS Results**:
- Optimal path guaranteed
- Explores all lower-cost paths first
- Higher node expansion

**Greedy Results**:
- Fast exploration
- May not find optimal path
- Fewer nodes expanded
- Can get stuck in local optima

**A* Results**:
- Best of both worlds
- Optimal with admissible heuristic
- Efficient node expansion
- Balances cost and heuristic

## 🧪 Test Cases

### BFS Tests
```python
# Valid path
BFS_Matheus("connections", "Dolly", "Matheus")
# Output: Dolly -> Bob -> Adam -> Matheus

# Invalid graph name
BFS_Matheus("invalid_graph", "A", "B")
# Output: Error message

# Invalid node
BFS_Matheus("connections", "Invalid", "Bob")
# Output: Appropriate error message
```

## 🎓 Learning Outcomes

1. **Algorithm Implementation**: Hands-on experience with search algorithms
2. **Heuristic Design**: Understanding admissibility and consistency
3. **Performance Analysis**: Comparing time/space complexity
4. **Graph Theory**: Working with different graph representations
5. **Visualization**: Using Graphviz for algorithm debugging

## 🔍 Code Highlights

### Dynamic Graph Loading
```python
# GraphData.py stores multiple graphs
graphs = {
    "connections": {...},
    "map": {...}
}

# Access any graph dynamically
BFS_Matheus(graph_name, start, goal)
```

### Robust Error Handling
- Graph existence check
- Node validation
- Path availability verification
- User-friendly error messages

### Modular Design
- Separate State, Node, Data modules
- Reusable TreePlot visualization
- Easy to extend with new algorithms

## 📚 References

- Russell, S., & Norvig, P. (2020). *Artificial Intelligence: A Modern Approach* (4th ed.)
- COMP 237 Course Materials - Centennial College
- Graphviz Documentation: https://graphviz.org/

## 🎯 Assignment Requirements Met

✅ BFS implementation for social network
✅ Error handling for all edge cases
✅ Visual tree exploration
✅ UCS implementation and comparison
✅ Written analysis of algorithm performance
✅ Comprehensive documentation

## 👨‍💻 Author

**Matheus Teixeira**
- Course: COMP 237 - AI Fundamentals
- Institution: Centennial College
- Term: Fall 2023

---

*Demonstrates fundamental understanding of search algorithms and their applications in AI problem-solving.*