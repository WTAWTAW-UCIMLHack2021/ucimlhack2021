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
        """
        Initializes the bag's 'words' attribute to ensure that capitalization has
        no influence on the learner's decision.
        """
        if type(words) is set:
            self._words = {word.lower() for word in words}
        else:
            self._words = {word for sent in self._x for word in sent.split(" ")}

    def _init_word_count(self):
        """
        Counts the numbers of words that appear in the dataset and
        stores it in a dictionary with those counts. The fields attribute
        would be sorted from most common to least common word.
        """
        self._word_count = collections.defaultdict(int)
        for sent in self._x:
            for word in sent.split():
                if word.lower() in self._words:
                    self._word_count[word.lower()] += 1


    def _x_data(self):
        """
        Converts the training x set into a vector of values.
        :return: [[int]]
        """
        return [[int(field in sent) for field in self._fields] for sent in self._x]

    def vectorize(self, classify=False) -> pd.DataFrame:
        """
        Converts every sentence in self._x to a vector of 0/1 based on the property
        that a word from self._fields is contained in that sentence.
        :param classify: bool
        :return: pd.DataFrame [len(self._x) rows x len(self._fields)]
        """
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
        """
        Trains the self._learner using a decision tree from
        the library sklearn.
        :return: None
        """
        self._learner.fit(self._x_data(), self._y)

    def plot(self):
        """
        Plots the self._learner as a tree.
        :return:
        """
        tree.plot_tree(self._learner)

    def predict(self, sent: str) -> int:
        """
        Given a sentence, the predict function predicts which class
        that sentence belongs to and returns it's prediction.
        :param sent: str Any given sentence passed through it
        :return: int The class the learner the sentence belongs to
        """
        data = [int(field in sent) for field in self._fields]
        y_hat = self._learner.predict([data])
        return y_hat[0]

    def error_rate(self, x_test: [str], y_test: [int]) -> float:
        """
        Given an array of test sentences, it gives a proportion from
        0 to 1 which states the error rate of the built-in learner.
        :param x_test: [str] An array of test sentences
        :param y_test: [int] An array of the true class of the corresponding sentences in x_test
        :return: The error rate of the learner of the object
        """
        incorrect = 0
        for i, x in enumerate(x_test):
            y_hat = self.predict(x)
            y = y_test[i]
            if y != y_hat:
                incorrect += 1
        return incorrect/len(x_test)


if __name__ == '__main__':
    covidlies = pd.read_csv('../data/covid_lies_processed.csv')
    X_train = list(covidlies['tweet'])[:5000]
    X_test = list(covidlies['tweet'])[5000:]
    Y_train = list(covidlies['label'])[:5000]
    Y_test = list(covidlies['label'])[5000:]
    bag = BagOfWords(X_train, Y_train)
    print(bag.vectorize())
    bag.train()
    print(bag.error_rate(X_test, Y_test))