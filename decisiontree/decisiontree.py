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
y = x.pop("label")
features_to_be_droped =['tempo', 'valence', 'energy','danceability', 'mode', 'instrumentalness', 'time_signature']
x = x.drop(features_to_be_droped, axis=1)
#x = x[["speechiness","instrumentalness"]].copy()
x=(x-x.mean())/x.std()
clf = tree_classifier = DecisionTreeClassifier(max_depth=5)

score = cross_val_score(clf, x, y, cv=5)
print(score)
score = statistics.mean(score)
print(score)

# test on test data

clf = DecisionTreeClassifier(max_depth=5)

clf.fit(x.values, y.values)

tree_plot = Source(tree.export_graphviz(clf, out_file='tree_best.dot', feature_names=x.columns, class_names=['LIKE', 'DISLIKE'], filled=True, rounded=True, special_characters=True))
#tree_plot

tree_view = Source.from_file('tree_best.dot')
print(x.columns)
tree_view.view()

###
song_test_data = pd.read_csv('songs_to_classify.csv')
print(song_test_data.columns)
song_test_data = song_test_data.drop(features_to_be_droped, axis=1)
song_test_data=(song_test_data-song_test_data.mean())/song_test_data.std()

song_test_data['label'] = clf.predict(song_test_data)

print(song_test_data.head(100))
result = [''.join([row for row in str(song_test_data['label'].values)])]
print(''.join(song_test_data['label'].values.astype(str)))


# test_data: 11100001001111011000111111111001110111101100110000010110100111100100110101010100101111110001010101110010001010010011101001100001011011011101011110001010000001111110010001111010101111110010101011111010
