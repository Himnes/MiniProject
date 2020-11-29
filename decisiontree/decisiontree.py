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
from sklearn.tree import DecisionTreeClassifier
from graphviz import Source
from sklearn import tree

x = pd.read_csv("training_data.csv")
song_test_data = pd.read_csv('songs_to_classify.csv')
print(song_test_data.columns)

y = x.pop("label")
features_to_be_droped =['tempo', 'valence', 'energy','danceability', 'mode', 'instrumentalness', 'time_signature']
x = x.drop(features_to_be_droped, axis=1)
x=(x-x.mean())/x.std()

song_test_data = song_test_data.drop(features_to_be_droped, axis=1)
song_test_data=(song_test_data-song_test_data.mean())/song_test_data.std()

max_depth = 0
max_score = 0
for i in range(1,7):
  clf = DecisionTreeClassifier(max_depth=i)
  score = cross_val_score(clf, x, y, cv=5)
  print(i, score)
  score = statistics.mean(score)
  if max_score < score:
     max_score = score
     max_depth = i
  print(i, score)

clf = DecisionTreeClassifier(max_depth=max_depth)

clf.fit(x.values, y.values)

song_test_data['label'] = clf.predict(song_test_data)

print(song_test_data.head(10))
result = [''.join([row for row in str(song_test_data['label'].values)])]
print(''.join(song_test_data['label'].values.astype(str)))


# test_data: 11100001001111011000111111111001110111101100110000010110100111100100110101010100101111110001010101110010001010010011101001100001011011011101011110001010000001111110010001111010101111110010101011111010
