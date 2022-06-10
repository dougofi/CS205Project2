import pandas as pd
import numpy as np
import math


def main(): 
    #welcome and introduction
    print("""In this application we will be able to determine which combination of
features work best for our nearest neighbour classifier \n""")

    name = input("Kindly enter the name of your data file: " ) #input for file 
    choice = input(""" Which Search Pattern do you wish to use?  
    1. Forward Selection Algorithm
    2. Backwards Elimination Algorithm
    
    Kindly Enter 1 or 2 to make a selection: """ )#input for algo choice
    #fetching the file and using loadtxt from numpy for faster reading with support for lines with missing values(if they exist)
    try: 
        with open(name) as d: 
            items = pd.read_csv(name)
            num_items = len(items) #number of instances
            dataset = np.loadtxt(name) #numpy method for loading row-wise data from an ascii file with 8 point floating number
    except: 
        print("File name entered is not valid or does not exist, please restart program and try again")

    # printing dataset details
    print(f"This dataset has {len(dataset[0]) -1 } features(not including the class attribute) with {num_items} instances ")
    print("Beggining Search")
    if choice == "1": 
        forward_chaining(dataset)
    else: 
        backward_chaining(dataset)
 
  
#metric to be used in determining closest neighbours. sqrt not done here but within forward and backward functions due for weird numpy array issue i faced
def euclidean_distance(vectorA, vectorB, hold_out): 
    distance = 0
    for i in hold_out: 
        distance += (vectorA[i] - vectorB[i]) ** 2  
    return distance

#this function when called centers the values of the measured features around the mean
def center(dataset): 
    avg = np.mean(dataset) 
    std = np.std(dataset)
    dataset = (dataset - avg) / std
    return dataset

def forward_chaining(dataset): 
    #initializing placeholder variables for accuracy measurements and arrays for containing feature sets
    dataset = center(dataset)
    current_set_of_features = [] 
    best_so_far_accuracy = 0 
    best_features = []
    #loops once outer for each feature
    for i in range(1, len(dataset[0])):
        print(f"Current Best: {best_so_far_accuracy} {best_features} ") #putting this her so for runs that take long, we will have the latest recorded values
        print(f"On level {i} of the tree ")
        feature_to_add_at_this_level = []
       
        best_accuracy_current = 0
        #loops same number of times for during each outer loop
        for j in range(1, len(dataset[0])): 
            if j not in current_set_of_features: 
                print(f"--Considering adding the {j} feature")
                
                hold_out = list(current_set_of_features) #creating a working copy of current list of features for testing
                hold_out.append(j) #adding newest entry and testing how that affects accuracy in the search
                hits = 0 #successful classifications
                shortest_distance = math.inf #large value so its guaranteed to be lower than our metric of choice(euclidean)
                prediction_class = 0 
                #comparing each of the rows as vectors and passing it through the euclidaen distance functoin 
                for m in dataset: 
                    # print(m)
                    # exit()
                    shortest_distance = math.inf
                    for n in dataset: 
                        same = (m == n).all() 
                        if not same: #make sure the numpy arrays aren't the same
                            value = euclidean_distance(m , n , hold_out) 
                            if math.sqrt(value) < shortest_distance: #square root is done here instead of in the ED function cos i was experiencing a size-1 error
                                shortest_distance = math.sqrt(value)
                                prediction_class = n[0]
                    if prediction_class == m[0]: 
                        hits += 1 #recording successful classifications
                accuracy = hits / (len(dataset) )  #measuring accuracy
                  
                #checking if current stored accuracy should be updated to reflect on the current working copy and globally as well 
                if accuracy > best_accuracy_current: 
                    best_accuracy_current = accuracy
                    feature_to_add_at_this_level = j 
        current_set_of_features.append(feature_to_add_at_this_level)
        #print operatings to display progress and results
        print(f"On level {i} I added feature {feature_to_add_at_this_level} to current set")   
        print(f"Using feature(s) {current_set_of_features} the accuracy is {best_accuracy_current * 100}%")     
        if best_accuracy_current >= best_so_far_accuracy: 
            print(f"From {best_so_far_accuracy}")
            best_so_far_accuracy = best_accuracy_current
            print(f"To {best_so_far_accuracy}")
            print("-------------------------")
            best_features = list(current_set_of_features)
            
    print(f"Finished Search!! The complete best feature subset is {best_features} which has an accuracy of {best_so_far_accuracy*100}%")
                  

def backward_chaining(dataset):
    dataset = center(dataset)
    best_so_far_accuracy = 0 
    best_features = []
    current_set_of_features = [feature for feature in range(1, len(dataset[0]))]
            #loops same number of times for during each outer loop
    for i in range(1, len(dataset[0])): 
        worst_in_set = 0 #tracking the value to be removed
        best_accuracy_current  = 0 
        print(f"Current Best: {best_so_far_accuracy} {best_features} ") #intermediate results provider
        for j in range(1, len(dataset[0])): 
            if j in current_set_of_features: 
                print(f"Considering feature {j}")
                hold_out = list(current_set_of_features) ##creating a working copy of current list of features for testing
                hold_out.remove(j)  #adding newest entry and testing how that affects accuracy in the search
                hits = 0 #successful classifications
                shortest_distance = math.inf
                prediction_class = 0 
                #comparing each of the features across rows as vectors and passing it through the euclidean distance functoin 

                for m in dataset: 
                    shortest_distance = math.inf
                    for n in dataset: 
                        same = (m == n).all()
                        if not same: #make sure the numpy arrays aren't the sam
                            value = euclidean_distance(m , n , hold_out)
                            if math.sqrt(value) < shortest_distance: #square root is done here instead of in the ED function cos i was experiencing a size-1 error
                                shortest_distance = math.sqrt(value)
                                prediction_class = n[0]
                    if prediction_class == m[0]: 
                        hits += 1 #recording successful classifications
                accuracy = hits / (len(dataset) )  #measuring accuracy
                #checking if current stored accuracy should be updated to reflect on the current working copy and globally as well 
                if accuracy > best_accuracy_current: 
                    best_accuracy_current = accuracy
                    worst_in_set = j
        if worst_in_set in current_set_of_features: 
            current_set_of_features.remove(worst_in_set)
                    #print operatings to display progress and results
            print(f"On level {i} feature {worst_in_set} was removed from the current set of features")            
            print(f"With {current_set_of_features} the accuracy is {best_accuracy_current * 100 }% ")
        if best_accuracy_current  >= best_so_far_accuracy: 
            best_so_far_accuracy = best_accuracy_current
            best_features = list(current_set_of_features)
    print(f"Complete Best features were: {best_features} with an accuracy of {best_so_far_accuracy* 100}%" )            
                
      

if __name__ == '__main__':
    main()