import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn.neighbors as skl_nb
from sklearn.model_selection import train_test_split
from IPython.core.pylabtools import figsize
figsize(10, 6) # Width and hight
from feature_finder import feature_finder

def knn():
    
    features_pending = pd.read_csv("training_data.csv") # Has all features + labels   
    data_to_classify = pd.read_csv('songs_to_classify.csv') #Testdata with all features, no labels

    y = features_pending.pop("label") # Separate label from features
    categorical_features = features_pending[["key", "mode", "time_signature"]].copy()#Do not normalize categorical data!
    features_pending = features_pending.drop(["key", "mode", "time_signature"], axis=1)#Do not normalize categorical data!
    features_pending = (features_pending-features_pending.mean())/features_pending.std()#Normalize data!
    features_pending = pd.concat([features_pending, categorical_features], axis=1)

    categorical_features2 = data_to_classify[["key", "mode", "time_signature"]].copy()#Do not normalize categorical data!
    data_to_classify = data_to_classify.drop(["key", "mode", "time_signature"], axis=1)#Do not normalize categorical data!
    data_to_classify = (data_to_classify-data_to_classify.mean())/data_to_classify.std()#Normalize data!
    data_to_classify = pd.concat([data_to_classify, categorical_features2], axis=1)

    data_to_classify = data_to_classify[['speechiness','acousticness','instrumentalness','time_signature']]
    X = features_pending[['speechiness','acousticness','instrumentalness','time_signature']]

    X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.1, random_state = 5) #Select testdata from training data

    classification = []
    for k in range(50):
        model = skl_nb.KNeighborsClassifier(k+1)
        model.fit(X_train, Y_train)
        prediction = model.predict(X_test)
        classification.append(np.mean(prediction == Y_test))

        
    print("Chosen Nr neighbors are", classification.index(max(classification))+1, "with maximum accuracy of", max(classification)) #print out which nr neighbors have the max accuracy

    

    model = skl_nb.KNeighborsClassifier(classification.index(max(classification))+1)
    model.fit(X_train, Y_train)
    prediction = model.predict(data_to_classify) #Actual prediction on the songs

    string = ""
    for i in prediction:
        string = string + str(i)
    return string


model = skl_nb.KNeighborsClassifier(n_neighbors=22)
print(feature_finder(model))
#{'10_danceability': 0.8253333333333334,
# '11_liveness': 0.82,
# '12_valence': 0.7933333333333333,
# '13_key': 0.768,
# '1_speechiness': 0.7426666666666667,
# '2_acousticness': 0.8186666666666667,
# '3_instrumentalness': 0.8280000000000001,
# '4_time_signature': 0.8306666666666667,
# '5_mode': 0.8240000000000001,
# '6_loudness': 0.8226666666666667,
# '7_tempo': 0.8280000000000001,
# '8_duration': 0.8253333333333334,
# '9_energy': 0.8200000000000001}

#Note, removing any of the items seemed to lower the accuracy
s = knn()
print(s)
