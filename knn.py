import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn.neighbors as skl_nb
from sklearn.model_selection import train_test_split
from IPython.core.pylabtools import figsize
figsize(10, 6) # Width and hight


def knn():
    
    features_pending = pd.read_csv("training_data.csv") # Has all features + labels   
    data_to_classify = pd.read_csv('songs_to_classify.csv') #Testdata with all features, no labels

    y = features_pending.pop("label") # Separate label from features
    categorical_features = features_pending[["key", "mode", "time_signature"]].copy()#Do not normalize categorical data!
    features_pending = features_pending.drop(["key", "mode", "time_signature"], axis=1)#Do not normalize categorical data!
    features_pending = (features_pending-features_pending.mean())/features_pending.std()#Normalize data!
    features_pending = pd.concat([features_pending, categorical_features], axis=1)

    X = features_pending[['acousticness','danceability','duration', 'energy', 'instrumentalness', 'key', 'liveness', 'loudness', 'mode', 'speechiness','tempo', 'time_signature', 'valence']]

    X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.1, random_state = 5) #Select testdata from training data

    classification = []
    for k in range(50): #Creates a plot that shows the accuracy of different amount of neighbors
        model = skl_nb.KNeighborsClassifier(n_neighbors=k+1)
        model.fit(X_train, Y_train)
        prediction = model.predict(X_test)
        classification.append(np.mean(prediction == Y_test))

    K = np.linspace(1, 50, 50)
    plt.plot(K, classification, '.')
    plt.ylabel('Classification')
    plt.xlabel('Number of neighbors')
    plt.show()

    print("Chosen Nr neighbors are", classification.index(max(classification))+1, "with maximum accuracy of", max(classification)) #print out which nr neighbors have the max accuracy
    prediction = model.predict(data_to_classify) #Actual prediction on the songs
    return


knn()
