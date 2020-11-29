from sklearn.model_selection import cross_val_score
from sklearn import preprocessing
import pandas as pd
import statistics
import operator
import tqdm
import pprint


#Grades features based on how well they do in the model.
#Starts out with zero features -> finds the best first feature.
#Then tries all other features together with that intial feature Etc.
#Until all features have been added. 
#The order is their grade of "usefullness"

#Inputs:
#Model: The model you want to run

def feature_finder(model):
    score_keeper = {} # Will keep track of how much better we get when we add features
    features_added = pd.DataFrame() # Empty
    features_pending = pd.read_csv("training_data.csv") # Has all features + labels
    y = features_pending.pop("label") # Separate label from features

    categorical_features = features_pending[["key", "mode", "time_signature"]].copy() #Do not normalize categorical data!
    features_pending = features_pending.drop(["key", "mode", "time_signature"], axis=1) #Do not normalize categorical data!
    features_pending=(features_pending-features_pending.mean())/features_pending.std() #Normalize data!
    features_pending = pd.concat([features_pending,categorical_features],axis = 1)

    for i in tqdm.tqdm(range(len(features_pending.columns))):
        iteration_counter = 1
        temp_score_keeper = {}
        for feature in features_pending:

            temp_dataset = features_added.copy()
            temp_dataset = pd.concat([temp_dataset,features_pending[feature]],axis = 1)
            
            average_score = statistics.mean(cross_val_score(model, temp_dataset, y, cv=5)) #Take average of the 5 runs.
            temp_score_keeper[feature] = average_score
        
        best_feature = max(temp_score_keeper.items(), key=operator.itemgetter(1))[0] #Gets label with best score
        features_added = pd.concat([features_added,features_pending.pop(best_feature)],axis = 1)
        score_keeper[str(i+1)+"_"+best_feature] = temp_score_keeper[best_feature]

    return pprint.pformat(score_keeper)

#Test / Demo
#from sklearn import svm
#clf = svm.SVC(kernel='linear', C=1)
#print(feature_finder(clf))
