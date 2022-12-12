'''
@author: Matheus Teixiera

'''
####____ Requirement 1 ____####
connections = {}

####____ Requirement 5 ____####
#dictionary holding the names of all availables dictionaries
graphs = {}
graphs["connections"] = connections

#create a dictionary with all the mappings
connections["Matheus"] = ["George", "Frank", "Adam"]
connections["George"] = ["Matheus"]
connections["Frank"] = ["Matheus"]
connections["Adam"] = ["Matheus", "Ema", "Bob"]
connections["Ema"] = ["Adam", "Bob", "Dolly"]
connections["Bob"] = ["Adam", "Ema", "Dolly"]
connections["Dolly"] = ["Bob", "Ema"]





