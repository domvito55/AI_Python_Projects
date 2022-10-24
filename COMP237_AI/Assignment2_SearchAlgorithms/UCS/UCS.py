'''
@author: Devangini Patel
'''
from State import State
from Node import Node
import queue
from TreePlot import TreePlot
    

def UCS():
    """
    This method performs A* search
    """
    
    #create queue
    pqueue = queue.PriorityQueue()
    #To create visited list
    visited = [] 
    
    #create root node
    initialState = State()
    root = Node(initialState, None)
    
    #show the search tree explored so far
    treeplot = TreePlot()
    treeplot.generateDiagram(root, root)
    
    #add to priority queue
    #ignoring the heuristic, looking to cost only
    pqueue.put((root.costFromRoot, root))
    
    #check if there is something in priority queue to dequeue
    while not pqueue.empty(): 
        # print("###############################") #debug purposes
        
        #dequeue nodes from the priority Queue
        _, currentNode = pqueue.get()

        #If the node was already visited, skip this one
        #This helps to keep a clean plot, if you already foud
        #a minimum path to a node, you don't need to add it to
        #another branch of the tree
        if currentNode.state.place in visited:
            continue

        visited.append(currentNode.state.place)
        
        #remove from the fringe
        currentNode.fringe = False
        
        
        #check if it has goal State
        # print ("-- dequeue --", currentNode.state.place) #debug purposes
        
        #check if this is goal state
        if currentNode.state.checkGoalState():
            print ("reached goal state")
            #print the path
            print ("----------------------")
            print ("Path")
            currentNode.printPath()
            
            #show the search tree explored so far
            treeplot = TreePlot()
            treeplot.generateDiagram(root, currentNode)
            break
            
        #get the child nodes 
        childStates = currentNode.state.successorFunction()
        # print ("-- childstates", childStates) #debug purposes
        # print ("-- visited", visited) #debug purposes

        
        for childState in childStates:
            # print(childState) #debug purposes
            
            #Logic complementing the lines 43 and 44
            #If the node was already visited, skip this one
            #This helps to keep a clean plot, if you already foud
            #a minimum path to a node, you don't need to add it to
            #another branch of the tree
            if childState not in visited:
                childNode = Node(State(childState), currentNode)
                
                #add to tree and queue
                # print((childNode.costFromRoot, childNode))
                pqueue.put((childNode.costFromRoot, childNode))
            
        #show the search tree explored so far
        treeplot = TreePlot()
        treeplot.generateDiagram(root, currentNode)
        
                
    #print tree
    print ("----------------------")
    print ("Tree")
    root.printTree()
    
UCS()