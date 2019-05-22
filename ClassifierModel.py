
import os.path
import importlib
from joblib import dump, load
from os import listdir
from pathlib import Path

from . import parameterizationAlgorithms
from . import classificationAlgorithms
from . import datasets
from . import exceptions

# There is more information in the documentation about implemented
# algorithms and datasets
parameterization_algorithms_list = [
    'TFIDF',
    'BagOfWords',
    'Doc2Vec'
]
classification_algorithms_list = [
    'LinearSupportVectorMachines',
    'NaiveBayes',
    # 'RandomForest',
    # 'DecisionTree'
]
datasets_list = [
    'Reuters'
]


class ClassifierModel:
    """
    Model for document classification.

    Parameters
    ----------
    param_alg : string
        Parameterization algorithm for document representation (to create
        feature vectors).

    classif_alg : string
        Classification algorithm.

    dataset : string, optional (default='Reuters')
        Dataset used for training and testing.

    train_size : float in range [0.01,0.99], optional (default=.7)
        Percentage of the dataset used for training.

    saved_model : boolean, optional (default=False)
        Only in case you want to use a model already trained. More
        information about available models using the static method
        available_implementations.

    Attributes
    ----------
    models_dir: string
        Relative path where saving the trained models.
    """

    models_dir = './trained_models/'

    def __init__(self, param_alg, classif_alg, dataset='Reuters',
                 train_size=.7, saved_model=False):
        # Validate inputs
        if param_alg not in parameterization_algorithms_list:
            raise ValueError('The parameterization algorithm is not '
                                     'valid. The implemented parameterization '
                                     'algorithms are: '+
                                     ', '.join(parameterization_algorithms_list))
        if classif_alg not in classification_algorithms_list:
            raise ValueError('The classification algorithm is not '
                                     'valid. The implemented classification '
                                     'algorithms are: ' +
                                     ', '.join(classification_algorithms_list))
        if dataset not in datasets_list:
            raise ValueError('The dataset is not valid. The available '
                                     'datasets are: ' +', '.join(datasets_list))
        if not (0.01 <= train_size <= 0.99):
            raise ValueError('The trainig size is not valid. It should'
                                     ' be a float in the range [0.01, 0.99]')

        if saved_model:
        	# If saved_model=True then look for the model
            self.name = param_alg+'_'+classif_alg+'_'+dataset
            filename = Path(self.models_dir+self.name+'.joblib')
            print(filename)
            path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), filename)
            print(path)
            if os.path.isfile(path):
                # There is a trained model with these parameters
                with open(path, 'rb') as f:
                    (self.param_alg, self.classif_alg, self.accuracy) = \
                        load(f)
                # self.loaded_model : to indicate if the model can be
                # trained again
                self.loaded_model = True
            else:
                raise exceptions.NotAvailableModelError(
                    'There is no saved model with this parameters. Check the '
                    'list of available models running the static method '
                    'available_implementations()')
        else:
            self.loaded_model = False
            self.accuracy = None
            # Import dynamically the classes
            self.__DatasetClass = getattr(datasets, dataset)
            self.X_train, self.Y_train, self.X_test, self.Y_test = \
                self.__DatasetClass.get_random_split(train_size)
            self.__ParamClass = getattr(parameterizationAlgorithms,param_alg)
            self.param_alg = self.__ParamClass()
            self.__ClassifClass = getattr(classificationAlgorithms,classif_alg)
            self.classif_alg = self.__ClassifClass()
            param = self.__ParamClass.__name__
            classif = self.__ClassifClass.__name__
            dataset_name = self.__DatasetClass.__name__
            self.name = param+'_'+classif+'_'+dataset_name

    def train(self):
        """
        Train the model.

        Returns
        -------
        Accuracy of the model.
        """
        if self.loaded_model:
            return self.accuracy
        else:
            # Train the parameterization algorithm and obtain the
            # feature vectors
            self.X_train, self.Y_train, self.X_test, self.Y_test = \
                self.param_alg.get_feature_vectors_of_dataset(
                    self.X_train, self.Y_train, self.X_test, self.Y_test)

            self.classif_alg.fit(self.X_train, self.Y_train, self.X_test,
                                 self.Y_test)
            self.accuracy = self.classif_alg.get_accuracy()
            return self.accuracy

    def classify(self, texts):
        """
        Classify texts to obtain their topics.

        Parameters
        ----------
        texts : array-like
            List of texts to classify.

        Returns
        -------
        List with the topics of the texts.
        """
        feature_vector = self.param_alg.get_feature_vector(texts)
        return self.classif_alg.predict(feature_vector)

    @staticmethod
    def available_implementations():
        """
        Return information about the implemented algorithms and the saved models.

        Returns
        -------
        String with the information.
        """
        line1 = '### Information about available algorithms, models and ' \
                'datasets ###\n\n'
        line2 = '- Parameterization algorithms: ' + \
                ', '.join(parameterization_algorithms_list) + '\n'
        line3 = '- Classification algorithms: ' + \
                ', '.join(classification_algorithms_list) + '\n'
        line4 = '- Datasets: ' + ', '.join(datasets_list) + '\n'
        models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                  ClassifierModel.models_dir)
        saved_models = [model.replace('.joblib','')
                        for model in listdir(models_dir)
                        if model.endswith('.joblib')]
        line5 = '- Saved models: ' + ', '.join(saved_models) + '\n\n'
        return line1+line2+line3+line4+line5

    def save_model(self):
        """
        Save the model to use it later.
        """
        filename = Path(self.models_dir + self.name + '.joblib')
        path = os.path.join(
                    os.path.dirname(os.path.abspath(__file__)), filename)
        with open(path, 'wb') as f:
            dump((self.param_alg, self.classif_alg, self.accuracy), f)

    def get_accuracy(self):
        """
        Return the accuracy of the model.

        Returns
        -------
        Accuracy of the model.
        """
        return self.accuracy



if __name__ == '__main__':

    print('Running classification...')

    from textclassifier.ClassifierModel import ClassifierModel
    classifier = ClassifierModel('TFIDF', 'LinearSupportVectorMachines')
    classifier.train()
    print("Accuracy: {}".format(classifier.get_accuracy()))

    result = classifier.classify([text1,text2,text3])
    print(result)

