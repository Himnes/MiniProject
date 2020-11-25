import pandas as pd
from sklearn.linear_model import LogisticRegression
import statistics
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import BaggingClassifier
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

for state in range(20):
    x = pd.read_csv("training_data.csv")
    y = x.pop("label").to_numpy()
    #x = x[["speechiness", "loudness", "acousticness", "instrumentalness", "tempo"]].copy()
    x = x[["speechiness", "loudness", "acousticness", "instrumentalness"]].copy()

    x=(x-x.mean())/x.std()
    x = x.to_numpy()
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=state)

    x_len = len(X_train)
    data_set = []
    for i in range(4):
        data_set.append([X_train[int(i*0.2*x_len) : int((i+2)*0.2*x_len)], 
                        y_train[int(i*0.2*x_len) : int((i+2)*0.2*x_len)]])

    models = []
    for [x,y] in data_set:
        clf = LogisticRegression(random_state=13).fit(x,y)
        models.append(clf)


    from sklearn.metrics import accuracy_score

    all_guess = []
    for model in models:
        guess = model.predict(X_test)
        print(guess.shape)
        all_guess.append(guess)

    score.append(accuracy_score(y_test, guess))

    print(state, statistics.mean(score))
