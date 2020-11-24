from sklearn.model_selection import cross_val_score
from sklearn import preprocessing
import pandas as pd
import statistics
import operator
import tqdm
import pprint


def train(model):
    features_pending = pd.read_csv("training_data.csv") # Has all features + labels
    y = features_pending.pop("label") # Separate label from features

    categorical_features = features_pending[["key", "mode", "time_signature"]].copy() #Do not normalize categorical data!
    features_pending = features_pending.drop(["key", "mode", "time_signature"], axis=1) #Do not normalize categorical data!
    features_pending=(features_pending-features_pending.mean())/features_pending.std() #Normalize data!
    features_pending = pd.concat([features_pending,categorical_features],axis = 1)

    x = features_pending[["speechiness", "loudness", "acousticness", "instrumentalness", "tempo"]].copy() #Put the features you want to use here!

    model.fit(x,y)
    return model

def final_test(trained_model): #NOTE!! This assumes you have trained on normalized features!!! If you're unsure, hit me up!

    features_pending = pd.read_csv("songs_to_classify.csv") # Has all features + labels


    categorical_features = features_pending[["key", "mode", "time_signature"]].copy() #Do not normalize categorical data!
    features_pending = features_pending.drop(["key", "mode", "time_signature"], axis=1) #Do not normalize categorical data!
    features_pending=(features_pending-features_pending.mean())/features_pending.std() #Normalize data!
    features_pending = pd.concat([features_pending,categorical_features],axis = 1)

    x = features_pending[["speechiness", "loudness", "acousticness", "instrumentalness", "tempo"]].copy() #Put the features you want to use here!
    prediction = trained_model.predict(x)
    str_prediction = [str(i) for i in prediction]

    return "".join(str_prediction)




#Example!
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(random_state=0)
model = train(model)
result = final_test(model)
print(result)

