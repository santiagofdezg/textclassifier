# import nltk
# nltk.download('reuters')
from nltk.corpus import reuters
from sklearn.model_selection import train_test_split

class Dataset(object):
    """
    Super class that establish a interface for the datasets.
    """

    @staticmethod
    def get_data():
        """
        Return the texts with their respective labels.

        Returns
        -------
        X_data: array
            Texts of the dataset.
        Y_data : arrays
            Labels of the texts.
        """
        pass

    @staticmethod
    def get_available_labels():
        """
        Return the available labels in the dataset.

        Returns
        -------
        labels: array
            Available labels.
        """

    @staticmethod
    def get_default_split():
        """
        Some datasets are already divided in training and test sets. This
        method returns the default split.

        Returns
        -------
        X_train, X_test, Y_train, Y_test : arrays
            Default training and test sets with their respective labels.
        """
        pass


    @staticmethod
    def get_random_split(train_size=.7):
        """
        Return a random split of the original dataset.

        Parameters
        ----------
        train_size : float in range [0.01,0.99], optional (default=.7)
            Percentage of the dataset used for training.

        Returns
        -------
        X_train, X_test, Y_train, Y_test : arrays
            Random training and test sets with their respective labels.
        """
        pass




class Reuters(Dataset):
    """
    One of the most widely used test collection for text categorization
    research. This collection contains 21578 documents with their respective
    category/categories. It is available to download in:
    http://kdd.ics.uci.edu/databases/reuters21578/reuters21578.html.

    Notes
    -----
    Possible actions using the dataset provided by the nltk library:

    - List of documents
        documents = reuters.fileids()
    - List of categories
        categories = reuters.categories()
    - Documents in a category
        category_docs = reuters.fileids("acq")
    - Categories in a document
        reuters.categories(doc_id)
    - Words in a document
        document_words = reuters.words(document_id)
    - Raw document
        print(reuters.raw(document_id))
    """

    @staticmethod
    def get_data():
        document_id = reuters.fileids()
        Y_data = []
        X_data = []
        for doc in document_id:
            Y_data.append(reuters.categories(doc))
            X_data.append(reuters.raw(doc))

        return X_data, Y_data

    @staticmethod
    def get_available_labels():
        return reuters.categories()

    @staticmethod
    def get_random_split(train_size=0.7):
        document_id = reuters.fileids()
        doc_categories = []
        doc_text = []
        for doc in document_id:
            doc_categories.append(reuters.categories(doc))
            doc_text.append(reuters.raw(doc))

        # Divide dataset in training and test
        X_train, X_test, Y_train, Y_test = \
                train_test_split(doc_text, doc_categories,
                                train_size=train_size)
        return X_train, Y_train, X_test, Y_test

    @staticmethod
    def get_default_split():
        documents = reuters.fileids()
        train_docs_id = list(filter(lambda doc: doc.startswith("train"), documents))
        test_docs_id = list(filter(lambda doc: doc.startswith("test"), documents))

        X_train = [reuters.raw(doc_id) for doc_id in train_docs_id]
        X_test = [reuters.raw(doc_id) for doc_id in test_docs_id]
        Y_train = [reuters.categories(doc_id) for doc_id in train_docs_id]
        Y_test = [reuters.categories(doc_id) for doc_id in test_docs_id]

        return X_train, Y_train, X_test, Y_test