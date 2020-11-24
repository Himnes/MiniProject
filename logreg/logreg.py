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


x = pd.read_csv("training_data.csv")
y = x.pop("label")
x = x[["speechiness", "loudness", "acousticness", "instrumentalness", "tempo"]].copy()
#x = x[["speechiness","instrumentalness"]].copy()

x=(x-x.mean())/x.std()

clf = LogisticRegression(random_state=0)
score = cross_val_score(clf, x, y, cv=5)
print(score)
score = statistics.mean()
print(score)