import collections
import pandas as pd
from sklearn import tree

class BagOfWords:
    def __init__(self, x: [str], y: [int], words=None, field_key=(lambda x:-x[1])):
        self._x = x
        self._y = y
        self._field_key = field_key
        self._init_words(words)
        self._init_word_count()
        self._learner = tree.DecisionTreeClassifier()
        self._fields = [field for field, _ in sorted(self._word_count.items(), key=self._field_key)]

    def _init_words(self, words):
        if type(words) is set:
            self._words = {word.lower() for word in words}
        else:
            self._words = {word for sent in self._x for word in sent.split(" ")}

    def _init_word_count(self):
        self._word_count = collections.defaultdict(int)
        for sent in self._x:
            for word in sent.split():
                if word.lower() in self._words:
                    self._word_count[word.lower()] += 1


    def _x_data(self):
        return [[int(field in sent) for field in self._fields] for sent in self._x]

    def vectorize(self, classify=False) -> pd.DataFrame:
        data = []
        for key, sent in enumerate(self._x):
            vector = dict()
            for field in self._fields:
                vector[field] = int(field in sent)
            if classify:
                vector['y_class'] = self._y[key]
            data.append(vector)

        return pd.DataFrame(data)

    def train(self):
        self._learner.fit(self._x_data(), self._y)

    def plot(self):
        tree.plot_tree(self._learner)

    def predict(self, sent: str) -> int:
        data = [int(field in sent) for field in self._fields]
        y_hat = self._learner.predict([data])
        return y_hat[0]

    def error_rate(self, x_test: [str], y_test: [int]):
        incorrect = 0
        for i, x in enumerate(x_test):
            y_hat = self.predict(x)
            y = y_test[i]
            if y != y_hat:
                incorrect += 1
        return incorrect/len(x_test)


if __name__ == '__main__':
    X_train = ["I'm so hungry I eat a horse", "I drink water", "She drinks milk",
               "Why can't I eat anything", "Are you thirsty", "Do you have anything to eat"]
    X_test = ["I am hungry", "He is thirsty", "I eat food", "I drink juice"]
    Y_train = [1, 0, 0, 1, 0, 1]
    Y_test = [1, 0, 1, 0]
    bag = BagOfWords(X_train, Y_train)
    print(bag.vectorize())
    bag.train()
    print(bag.error_rate(X_test, Y_test))