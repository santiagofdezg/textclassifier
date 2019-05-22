from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import precision_score, recall_score, f1_score
from . import exceptions
import warnings

#Pending to test:
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

class ClassificationAlgorithm(object):
    """
    Super class that establish a interface for classification algorithms.
    These methods are prepared for an easy implementation of the
    classification algorithms from sklearn library. To implement algorithms
    from other libraries it may be necessary to overwrite the methods of
    this base class.

    Attributes
    ----------
    algorithm_class
        Instance of a classification algorithm.
    classifier
        Instance of a classification algorithm wrapped by the
        OneVsRestClassifier class.
    precision : float
        Precision of the classification algorithm.
    """

    algorithm_class = None
    classifier = None
    precision = None

    def fit(self, X_train, Y_train, X_test=None, Y_test=None):
        """
        Train the classification algorithm.

        Parameters
        ----------
        X_train : array-like
            Feature vector with texts to train the algorithm.
        Y_train : array-like
            Labels of the training texts.
        X_test : array-like, optional (default=None)
            Feature vector with texts to test the algorithm.
        Y_test : array-like, optional (default=None)
            Labels of the test texts.
        """

        # Each document can has several labels, so we need a
        # multilabel classifier
        self.mlb = MultiLabelBinarizer()
        train_labels = self.mlb.fit_transform(Y_train)

        # Train the classifier
        self.classifier = OneVsRestClassifier(self.algorithm_class)
        self.classifier.fit(X_train, train_labels)

        return self.get_accuracy(X_test, Y_test)

    def predict(self, texts):
        """
        Classify the input texts and return their topics.

        Parameters
        ----------
        texts : array-like
            Texts to classify.
        """

        if self.classifier is None:
            raise exceptions.NotFittedAlgorithmError('The classification '
                        'algorithm is not fitted. Fit it before using it.')
        else:
            predictions = self.classifier.predict(texts)
            return self.mlb.inverse_transform(predictions)

    def get_accuracy(self, X_test=None, Y_test=None):
        """
        Return the accuracy of the classification algorithm. Instead of get
        the current accuracy of the algorithm, it is possible to measure the
        accuracy again by using the inputs X_test and Y_test.

        Parameters
        ----------
        X_test : array-like, optional (default=None)
            Feature vector with texts to calculate the accuracy.
        Y_test : array-like, optional (default=None)
            Labels of the test texts.
        """

        if self.classifier is None:
            raise exceptions.NotFittedAlgorithmError('The classification '
                                        'algorithm is not fitted. Fit it '
                                        'before using it.')
        else:
            if (X_test is not None) and (Y_test is not None):
                test_labels = self.mlb.transform(Y_test)
                test_predictions = self.classifier.predict(X_test)
                self.precision = '%.4f' % (precision_score(test_labels,
                                                           test_predictions,
                                                            average='micro'))
            else:
                if self.precision is None:
                    # If the accuracy of the algorithm was not measured with
                    # a test set
                    warnings.warn('The accuracy is unknown. Try to calculate '
                                  'it by passing a test set to the '
                                  'get_accuracy method. You can also fit the '
                                  'algorithm again with training and test '
                                  'sets.')
            return self.precision


class LinearSupportVectorMachines(ClassificationAlgorithm):

    def __init__(self):
        self.algorithm_class = LinearSVC()


class NaiveBayes(ClassificationAlgorithm):

    def __init__(self):
        self.algorithm_class = MultinomialNB()


class RandomForest(ClassificationAlgorithm):

    def __init__(self):
        self.algorithm_class = RandomForestClassifier()


class DecisionTree(ClassificationAlgorithm):

    def __init__(self):
        self.algorithm_class = DecisionTreeClassifier()

