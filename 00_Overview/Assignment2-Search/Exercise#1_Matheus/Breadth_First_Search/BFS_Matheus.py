'''
@author: Devangini Patel

edited by Matheus
'''

from Node import Node
from State import State
from collections import deque
from GraphData import *  #___edited
from TreePlot import TreePlot

####____ Requirement 4 ____####
####____ Requirement 5 ____####
def BFS_Matheus(graph_name, start, goal): #____edited
    """
    This function performs BFS search using a queue
    """
    
    ####____ Requirement 7 ____####
    ####____ Requirement 8 ____####
    if graph_name not in graphs.keys():
        print("\n\t***************************************")
        print("\t*The given graph is not a dictionary  *")
        print("\t*or don't exist. Check your GraphData *")
        print("\t*file                                 *")
        print("\t***************************************")
        return
    graph = graphs[graph_name]
    error = False
    if start not in graph.keys():
        print("\n\t***************************************")
        print("\t*The given start point does not exist.*")
        print("\t*Check your spelling.                 *")
        print("\t***************************************")
        error = True
    if goal not in graph.keys():
        print("\n\t**************************************")
        print("\t*The given goal point does not exist.*")
        print("\t*Check your spelling.                *")
        print("\t**************************************")
        error = True
    if error:
        return -1

    #create queue
    queue = deque([]) 
    #since it is a graph, we create visited list
    visited = [] 
    ####____ Requirement 5 ____####
    #create root node
    initialState = State(start) #____edited
    root = Node(initialState)

    ####____ Requirement 9 ____####
    #show the search tree explored so far
    treeplot = TreePlot()
    treeplot.generateDiagram(root, root)


    #add to queue and visited list
    queue.append(root)    
    visited.append(root.state.name)
    # check if there is something in queue to dequeue
    while len(queue) > 0:
        
        #get first item in queue
        currentNode = queue.popleft()
        
        ####____ Requirement 3 ____####
        #currently selected for exploration
        currentNode.frontier = False

        
#        print (("-- dequeue --"), currentNode.state.name) #___edited
        
        ####____ Requirement 5 ____####
        #check if this is goal state
        if currentNode.state.checkGoalState(goal):
            ####____ Requirement 9 ____####
            print ("reached goal state")
            #print the path
            print ("----------------------")
            print ("Path")
            currentNode.printPath()

            ####____ Requirement 9 ____####
            #show the search tree explored so far
            treeplot = TreePlot()
            treeplot.generateDiagram(root, currentNode)

            break           

        ####____ Requirement 4 ____####
        #get the child nodes 
        childStates = currentNode.state.successorFunction(graph)  #___edited
        # print(childStates)
        
        for childState in childStates:
            # print(childState) #debugging purposes
            childNode = Node(State(childState))
            
            #check if node is not visited
            if childNode.state.name not in visited:
                
                #add this node to visited nodes
                visited.append(childNode.state.name)
                
                #add to tree and queue
                currentNode.addChild(childNode)
                queue.append(childNode)

        ####____ Requirement 6 ____####
        if len(queue) == 0: #___edited
            print ("\n\t*********************************")
            print("\t*Sorry no relationship was found*")
            print ("\t*********************************")
        ####____ Requirement 9 ____####
        #show the search tree explored so far
        treeplot = TreePlot()
        treeplot.generateDiagram(root, currentNode)

    ####____ Requirement 9 ____####
    # print tree
    print ("----------------------")
    print ("Tree")
    root.printTree()

####____ Requirement 9 ____####
BFS_Matheus("connections", "Dolly", "Matheus") #___edited
BFS_Matheus("connections", "George", "Bob") #___edited
