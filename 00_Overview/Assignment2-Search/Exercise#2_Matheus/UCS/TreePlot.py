'''
@author: Devangini Patel
'''
import pydot 
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

class TreePlot:
    """
    This class creates tree plot for search tree
    """
    counter = 0 #added just to make possible save more the one image
    
    def __init__(self):
        """
        Constructor
        """
        TreePlot.counter += 1 #added just to make possible save more the one image

        self.graph = pydot.Dot(graph_type='graph', dpi = 300)
        self.index = 0
        
    
    def createGraph(self, node, currentNode):
        """
        This method creates pydot graph from search tree
        Similar to printTree() of Node class
        """
        
        # As the are multiples paths to the same node, I have add the "and condition"
        # so the plot will only color red the node in the minimum path
        if (node.costFromRoot == currentNode.costFromRoot) and (node.state.place == currentNode.state.place):
            color = "#ee0011"
        elif node.fringe:
            color = "#0011ee"
        else:
            color = "#00ee11"
            
        #create node
        # USC looks only to the coast not the heuristic.
        parentGraphNode = pydot.Node(str(self.index) + " " + \
            node.state.place, style="filled", \
            fillcolor = color, xlabel = node.costFromRoot)
        # print(node.state.place, node.costFromRoot)
        self.index += 1
        
        #add node
        self.graph.add_node(parentGraphNode)
        
        #call this method for child nodes
        for childNode in node.children:
            childGraphNode = self.createGraph(childNode, currentNode)
            
            #create edge
            edge = pydot.Edge(parentGraphNode, childGraphNode)
            
            #add edge
            self.graph.add_edge(edge)
            
        return parentGraphNode
        
    
    def generateDiagram(self, rootNode, currentNode):
        """
        This method generates diagram
        """
        self.createGraph(rootNode, currentNode)
        
        #show the diagram
        #changed to make possible save more the one image
        self.graph.write_png('graph' + str(TreePlot.counter) + '.png')
        img=mpimg.imread('graph' + str(TreePlot.counter) + '.png')
        plt.imshow(img)
        plt.axis('off')
#        mng = plt.get_current_fig_manager()
#        mng.window.state('zoomed')
        plt.show()
