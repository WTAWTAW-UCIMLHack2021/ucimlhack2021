import collections
import pandas as pd
from sklearn import tree
from util import transform
from cross_validation import stratify_split

def data_split(data: pd.DataFrame, id: str, x: str, y: str) -> ([str], [bool], [str], [bool]):
    """
    This function takes any DataFrame along with 3 fields from that set {id, x, y}
    and calls the imported stratify_split function in order to split the data into
    4 lists (2 for training {X_train, Y_train} and 2 for cross-validation/testing {X_test, Y_test}).
    :param data(pd.DataFrame): The dataset the user wants to split into training and testing data.
    :param id(str): The field name of the given data set that uniquely identifies each row in the data set.
    :param x(str): The field name of the column where its values are the desired input values (X_train/X_test).
    :param y(str): The field names of the column where its values we want to train/test on (Y_train/Y_test).
    :return X_train, Y_train, X_test, Y_test: The resulting data subsets to use for training/testing.
    """
    assert(id in [field for field in data]), "Given id field is not a field found in the given data set"
    assert(x in [field for field in data]), "Given input field is not a field found in the given data set"
    assert(y in [field for field in data]), "Given prediction field is not a field found in the given data set"
    assert(len(data[id]) == len(data[x]) == len(data[y])), "Some x and y value pairs associated with a unique id"
    assert(len(data[id]) == len({id for id in data[id]})), "Id field given does not uniquely \
                                                            identify rows in the DataFrame"
    index_dict = {t[id]: (t[x], t[y]) for _, t in data.iterrows()}
    _, train, test = stratify_split(list(data[x]), list(data[y]))

    X_train, Y_train = zip(*[index_dict[data[id].iloc[tid]] for tid in train])
    X_test, Y_test = zip(*[index_dict[data[id].iloc[tid]] for tid in test])

    return X_train, Y_train, X_test, Y_test

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
    covidlies = transform(covidlies)

    """
    tweet_index_dict = {t['tweet_id']: (t['tweet'], t['misconception']) for _, t in covidlies.iterrows()}
    _, train, test = stratify_split(list(covidlies['tweet_id']), list(covidlies['misconception']))

    X_train, Y_train = zip(*[tweet_index_dict[covidlies['tweet_id'].iloc[tid]] for tid in train])
    X_test, Y_test = zip(*[tweet_index_dict[covidlies['tweet_id'].iloc[tid]] for tid in test])
    """
    X_train, Y_train, X_test, Y_test = data_split(covidlies, 'tweet_id', 'tweet', 'misconception')
    bag = BagOfWords(X_train, Y_train)
    print(bag.vectorize())
    bag.train()
    print(bag.error_rate(X_test, Y_test))
