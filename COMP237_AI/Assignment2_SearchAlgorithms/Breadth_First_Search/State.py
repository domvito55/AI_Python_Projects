'''
@author: Devangini Patel

edited by Matheus.
'''

class State:
    '''
    This class retrieves state information for social connection feature
    '''
    
    def __init__(self, name = None):
        if name == None:
            #create initial state
            self.name = self.getInitialState()
        else:
            self.name = name
    
    def getInitialState(self):
        """
        This method returns me.
        """
        initialState = "Matheus"
        return initialState

    ####____ Requirement 5 ____####
    def successorFunction(self, connections):  #___edited
        """
        This is the successor function. It finds all the persons connected to the
        current person
        """
        return connections[self.name]
        
    ####____ Requirement 5 ____####
    def checkGoalState(self, goalName): #____edited
        """
        This method checks whether the person is the input goal.
        """ 
        return self.name == goalName #____edited