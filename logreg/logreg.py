
# import warnings filter
from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)




# '1_speechiness': 0.7626666666666666,
# '2_loudness': 0.8093333333333333,
# '3_acousticness': 0.8173333333333334,
# '4_instrumentalness': 0.8173333333333334,
# '5_tempo': 0.82,
import pandas as pd
from sklearn.linear_model import LogisticRegression
import statistics
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import BaggingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

score_sum = []
for state in range(100):
    x = pd.read_csv("training_data.csv")
    y = x.pop("label")
    x = x[["speechiness", "loudness", "acousticness", "instrumentalness", "tempo"]].copy()
    #x = x[["speechiness","instrumentalness"]].copy()

    x=(x-x.mean())/x.std()

    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=state)

    clf = LogisticRegression(random_state=0).fit(X_train,y_train)
    guess = clf.predict(X_test)
    score_sum.append(accuracy_score(y_test, guess))
print("Log", statistics.mean(score_sum))