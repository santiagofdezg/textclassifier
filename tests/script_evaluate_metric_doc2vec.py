import gensim
import time
from joblib import dump, load
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, accuracy_score
import warnings
import sys


available_models = [
    '20epochs_600vsize_40fv',
    # '20epochs_600vsize_100fv',
    # '15epochs_600vsize_60fv',
    # '15epochs_600vsize_100fv',
]

classifier_list = [
    ('SVC',LinearSVC),
    ('RFC',RandomForestClassifier)
]

for selected_model in available_models:
    print('')
    print('### '+selected_model+' ###')
    print('')
    print('# Loading model and dataset')
    model = gensim.models.doc2vec.Doc2Vec.load('./doc2vec_DBOW_' + selected_model)
    with open('./Y_train-X_train--feature_vector--DBOW_' + selected_model, 'rb') as f_train:
        (Y_train_new, X_train_new) = load(f_train)
    with open('./Y_test-X_test--feature_vector--DBOW_' + selected_model, 'rb') as f_test:
        (Y_test_new, X_test_new) = load(f_test)

    for selected_classifier in classifier_list:
        print('')
        print('### ' + selected_classifier[0] + ' ###')
        print('')
        print('# Training Labels')
        mlb = MultiLabelBinarizer()
        train_labels = mlb.fit_transform(Y_train_new)
        test_labels = mlb.transform((Y_test_new))

        print('# Training classifier')
        classifier = OneVsRestClassifier(selected_classifier[1]())

        if selected_classifier[0] == 'RFC':
            warnings.filterwarnings("ignore", category=FutureWarning)

        start = time.time()
        classifier.fit(X_train_new, train_labels)
        end = time.time()
        execution_time = end - start
        print(execution_time)

        if selected_classifier[0] == 'RFC':
            warnings.filterwarnings("default", category=FutureWarning)

        print('# Predicting test labels')
        predictions = classifier.predict(X_test_new)

        orig_stdout = sys.stdout
        f = open('./accuracy_doc2vec.txt', 'a')
        sys.stdout = f

        precision = '%.4f' % (precision_score(test_labels, predictions, average='micro'))
        accuracy = '%.4f' % (accuracy_score(test_labels, predictions))
        print('Doc2Vec + '+selected_classifier[0]+' + '+selected_model)
        print('Precision: ' + precision)
        print('Accuracy: ' + accuracy)
        print('Time: '+str(execution_time))
        print('')

        sys.stdout = orig_stdout
        f.close()