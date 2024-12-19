'''
@author: Devangini Patel
'''

from NavigationData import *
import math

class Node:
    '''
    This class represents a node in the search tree
    '''
    # this is responsible for the count
    # Lines 14 to 28 are addicions to help the Priority Queue
    # solve which node has higher priority if there is a tie
    # on their distances. The older node will have higher preference
    counter = 0

    def __lt__(self, other):
        if self.id<other.id:
           return self

    def __init__(self, state, parentNode):
        """
        Constructor
        """
        Node.counter += 1
        self.id = Node.counter

        self.state = state
        self.depth = 0
        self.children = []
        #self.parent
        self.setParent(parentNode)
        self.fringe = True
        #self.costFromRoot
        self.computeCost()
        
        
    def setParent(self, parentNode):
        """
        This method adds a node under another node
        """
        if parentNode != None:
            parentNode.children.append(self)
            self.parent = parentNode
            self.depth = parentNode.depth + 1
        else:
            self.parent = None
    
    def printTree(self):
        """
        This method prints the tree
        """
        print (self.depth , " - " , self.state.place)
        for child in self.children:
            child.printTree()
            
            
    def printPath(self):
        """
        This method prints the path from initial state to goal state
        """
        if self.parent != None:
            self.parent.printPath()
        print ("-> ", self.state.place)
        
        
    def computeDistance(self, location1, location2):
        """
        This method computes distance between two places
        """
        #difference in x coordinates
        dx = location1[0] - location2[0]
        #difference in y coordinates
        dy = location1[1] - location2[1]
        #distance
        distance = math.sqrt(dx ** 2 + dy ** 2)
        return distance
    
        
    def computeCost(self):
        """
        This method computes distance of current node from root node
        """
        
        if self.parent != None:
            #find distance from current node to parent
            #I have added the round fuction to make a cleaner plot.
            distance = round(self.computeDistance(location[self.state.place], \
                location[self.parent.state.place]),1)
            #cost = parent cost + distance
            self.costFromRoot = self.parent.costFromRoot + distance
        else:
            self.costFromRoot = 0
    
        
