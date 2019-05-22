from nltk import word_tokenize
from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords
import re
import gensim
from gensim.models.doc2vec import TaggedDocument
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
from . import exceptions


class ParameterizationAlgorithm(object):
    """
    Super class that establish a interface for parameterization algorithms.
    """

    def get_feature_vectors_of_dataset(self, x_train, y_train, x_test=None,
                                       y_test=None):
        """
        Return a feature vector for the training with the classifier.

        Parameters
        ----------
        x_train : array-like
            List of texts to train the parameterization algorithm.
        y_train : array-like, optional (default=None)
            List of labels for the training texts
        x_test : array-like
            List of texts to test the classifier algorithm.
        y_test : array-like, optional (default=None)
            List of labels for the test texts
        """
        pass

    def get_feature_vector(self, texts):
        """
        Return a feature vector that can be introduced in the classifier
        to predict the topics of the texts.

        Parameters
        ----------
        texts : array-like
            List of texts for prediction.
        """
        pass


class TFIDF(ParameterizationAlgorithm):
    tfidf = None

    def get_feature_vectors_of_dataset(self, x_train, y_train, x_test=None,
                                       y_test=None):
        # The implementation of TF-IDF of the library sklearn allows to make
        # a lot of variations of the tf-idf algorithm. Another option is the
        # module gesim.models.TfidfModel

        self.tfidf = TfidfVectorizer(tokenizer=tokenize, use_idf=True,
                                     norm='l2', min_df=3)
        x_train_new = self.tfidf.fit_transform(x_train)

        if x_test is not None and y_test is not None:
            x_test_new = self.tfidf.transform(x_test)
            y_test_new = y_test
        else:
            x_test_new = None
            y_test_new = None
        return x_train_new, y_train, x_test_new, y_test_new

    def get_feature_vector(self, texts):
        # If the parameterization algorithm was not trained it throws
        # an exception
        if self.tfidf is None:
            raise exceptions.NotFittedAlgorithmError('The parameterization '
                        'algorithm is not fitted. Fit it before using it.')
        else:
            return self.tfidf.transform(texts)


class BagOfWords(ParameterizationAlgorithm):
    bow = None

    def get_feature_vectors_of_dataset(self, x_train, y_train, x_test=None,
                                       y_test=None):
        self.bow = CountVectorizer(tokenizer=tokenize,)
        x_train_new = self.bow.fit_transform(x_train)

        if x_test is not None and y_test is not None:
            x_test_new = self.bow.transform(x_test)
            y_test_new = y_test
        else:
            x_test_new = None
            y_test_new = None
        return x_train_new, y_train, x_test_new, y_test_new

    def get_feature_vector(self, texts):
        if self.bow is None:
            raise exceptions.NotFittedAlgorithmError('The parameterization '
                        'algorithm is not fitted. Fit it before using it.')
        else:
            return self.bow.transform(texts)


class Doc2Vec(ParameterizationAlgorithm):
    d2v = None

    def _infer_vector(self, tagged_docs):
        targets, regressors = \
            zip(*[(doc.tags, self.d2v.infer_vector(doc.words, steps=30))
                  for doc in tagged_docs])
        return regressors, targets

    def _tag_document(self, doc_list):
        tagged_documents = \
            [TaggedDocument(words=tokenize(doc[0]), tags=doc[1])
             for doc in doc_list]
        return tagged_documents

    def get_feature_vectors_of_dataset(self, x_train, y_train, x_test=None,
                                       y_test=None):
        self.d2v = gensim.models.doc2vec.Doc2Vec(dm=0, vector_size=600,
                                                 min_count=1, workers=3)
        # Try to add: negative=5, hs=0, sample=0

        # Build vocabulary
        train_list = list(zip(x_train, y_train))
        train_tagged = self._tag_document(train_list)
        self.d2v.build_vocab(train_tagged)

        # Train the algorithm
        self.d2v.train(train_tagged, total_examples=len(train_tagged),
                       epochs=20)

        # Obtain the feature vectors of the training set
        x_train_new, y_train_new = self._infer_vector(train_tagged)

        if x_test is not None and y_test is not None:
            # Obtain the feature vectors of the test set
            test_list = list(zip(x_test, y_test))
            test_tagged = self._tag_document(test_list)
            x_test_new, y_test_new = self._infer_vector(test_tagged)
        else:
            x_test_new = None
            y_test_new = None

        return x_train_new, y_train_new, x_test_new, y_test_new

    def get_feature_vector(self, texts):
        if self.d2v is None:
            raise exceptions.NotFittedAlgorithmError('The parameterization '
                        'algorithm is not fitted. Fit it before using it.')
        else:
            feature_v = [self.d2v.infer_vector(tokenize(text), steps=30)
                         for text in texts]
            return feature_v


#######################
# SUPPORT FUNCTIONS

stop_words = stopwords.words("english")


def tokenize(document):
    # Convert a document into a list of lowercase tokens, ignoring tokens
    # that are too short
    words = gensim.utils.simple_preprocess(document, min_len=3)
    # Remove the stop words and stem the final words
    stemmer = SnowballStemmer('english')
    words = [stemmer.stem(word) for word in words if word not in stop_words]
    # Remove no-alphabetic characters
    f = re.compile('[a-zA-Z]+')
    filtered_words = list(filter(lambda word : f.match(word) , words))
    return filtered_words

