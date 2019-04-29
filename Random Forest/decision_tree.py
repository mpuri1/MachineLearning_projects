#Author: Manish Puri
#Note: Portions of the code have been completed using materials provided on HW assignment and discussions on Piazza

from util import entropy, information_gain, partition_classes
import numpy as np 
import ast

class DecisionTree(object):
    
    def __init__(self):
        # Initializing the tree as an empty dictionary
        #self.tree = []
        self.tree = {}
        self.depth_limit = 50


    # implementing functions for information gain
    def maxaggrY(self,argt):
        aggr, freq = np.unique(argt, return_counts  = True)
        max_index = np.argmax(freq)
        val = aggr[max_index]
        return val
        
    def constantaggrY(self,argt):
        aggr, freq =np.unique(argt, return_counts  = True)
        l1 = len(aggr)
        if l1==1:
            return True
        else:
            return False


    def find_dest(self, val1,val2, height):
        if height>= self.depth_limit:
            return self.maxaggrY(val2)

        if self.constantaggrY(val2):
            return val2[0]
        split_col,split_val = self.divide_tree(val1,val2)

        X_left, X_right, y_left, y_right = partition_classes(val1,val2,split_col,split_val)
        l_x = len(X_left)
        r_x = len(X_right)
        if l_x ==0 or r_x ==0:
            return self.maxaggrY(val2)
        else:
            split_dest = {}
            split_dest[split_col] = [split_val, self.find_dest(X_left,y_left,height+1),self.find_dest(X_right,y_right,height+1)]
            return split_dest

    def divide_tree(self,val1,val2):
        val_x = val1[0][0]
        val_y, max_info_gain = 0,0
        
        for i in range(len(val1[0])):
            caller_row = [row[i] for row in val1]
            caller_row = set(caller_row)
            for j in caller_row:
                y_l,y_r = partition_classes(val1,val2,i,j)[2:4]
                if information_gain(val2,[y_l,y_r])>max_info_gain:
                    max_info_gain = information_gain(val2,[y_l,y_r])
                    val_y, val_x = i,j
        return val_y, val_x
        


    def learn(self, X, y):
        # TODO: Train the decision tree (self.tree) using the the sample X and labels y
        # You will have to make use of the functions in utils.py to train the tree
        
        # One possible way of implementing the tree:
        #    Each node in self.tree could be in the form of a dictionary:
        #       https://docs.python.org/2/library/stdtypes.html#mapping-types-dict
        #    For example, a non-leaf node with two children can have a 'left' key and  a 
        #    'right' key. You can add more keys which might help in classification
        #    (eg. split attribute and split value)
  
        depth_limit=1   
        self.tree = self.find_dest(X,y,depth_limit)
        
        

    def classify(self, record):
        # TODO: classify the record using self.tree and return the predicted label
        decision_tree = self.tree
        while isinstance(decision_tree,dict):
            
            attr_val = list(decision_tree.keys())[0]
            rec_val = decision_tree[attr_val][0]

            if type(rec_val) == str:
                if record[attr_val] == rec_val:
                    decision_tree = decision_tree[attr_val][1]
                else:
                    decision_tree = decision_tree[attr_val][2]
            else:
                if record[attr_val] <= rec_val:
                    decision_tree = decision_tree[attr_val][1]
                else:
                    decision_tree = decision_tree[attr_val][2]
        return decision_tree   
            