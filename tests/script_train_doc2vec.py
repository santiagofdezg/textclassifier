import gensim
from gensim.models.doc2vec import TaggedDocument
from classifierlib.datasets import Reuters
# from tqdm import tqdm
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
import re
import time
from joblib import dump, load

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
    filtered_words = list(filter(lambda word: f.match(word), words))
    return filtered_words

# def feature_vector(model, tagged_docs):
#     targets, regressors = zip(*[(doc.tags, model.infer_vector(doc.words, steps=100)) for doc in tagged_docs])
#     return targets, regressors


# print('## VERSION: DBOW_100epochs_1000vsize')
#
# print('# Loading dataset')
# X_train, X_test, Y_train, Y_test = Reuters.get_random_split()
#
# train = list(zip(X_train,Y_train))
# test = list(zip(X_test,Y_test))
#
# print('# Transform to TaggedDocument')
# train_tagged = [TaggedDocument(words=tokenize(doc[0]), tags=doc[1]) for doc in train]
# test_tagged = [TaggedDocument(words=tokenize(doc[0]), tags=doc[1]) for doc in test]
#
# model = gensim.models.doc2vec.Doc2Vec(dm=0, vector_size=1000, min_count=1, workers=3)
#
# model.build_vocab(train_tagged)
#
# print('# Training')
# start = time.time()
# # Lanzar warning sobre que pode tardar moito e que non ten ningún tipo de verbose
# model.train(train_tagged, total_examples=len(train_tagged), epochs=100)
# end = time.time()
# print(end - start)
#
# print('# Saving model')
# model.save('./doc2vec_DBOW_100epochs_1000vsize')
#
# print('# Obtaining feature vectors of dataset')
# start = time.time()
# Y_train_new, X_train_new = feature_vector(model, train_tagged)
# Y_test_new, X_test_new = feature_vector(model, test_tagged)
# end = time.time()
# print(end - start)
#
# print('# Saving feature vectors')
# with open('./Y_train-X_train--feature_vector--DBOW_100epochs_1000vsize', 'wb') as f:
#     dump((Y_train_new, X_train_new), f)
#
# with open('./Y_test-X_test--feature_vector--DBOW_100epochs_1000vsize', 'wb') as f:
#     dump((Y_test_new, X_test_new), f)

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

# print('## VERSION: DBOW_100epochs_800vsize')
#
# print('# Loading dataset')
# X_train, X_test, Y_train, Y_test = Reuters.get_random_split()
#
# train = list(zip(X_train,Y_train))
# test = list(zip(X_test,Y_test))
#
# print('# Transform to TaggedDocument')
# train_tagged = [TaggedDocument(words=tokenize(doc[0]), tags=doc[1]) for doc in train]
# test_tagged = [TaggedDocument(words=tokenize(doc[0]), tags=doc[1]) for doc in test]
#
# model = gensim.models.doc2vec.Doc2Vec(dm=0, vector_size=800, min_count=1, workers=3)
#
# model.build_vocab(train_tagged)
#
# print('# Training')
# start = time.time()
# # Lanzar warning sobre que pode tardar moito e que non ten ningún tipo de verbose
# model.train(train_tagged, total_examples=len(train_tagged), epochs=100)
# end = time.time()
# print(end - start)
#
# print('# Saving model')
# model.save('./doc2vec_DBOW_100epochs_800vsize')
#
# print('# Obtaining feature vectors of dataset')
# start = time.time()
# Y_train_new, X_train_new = feature_vector(model, train_tagged)
# Y_test_new, X_test_new = feature_vector(model, test_tagged)
# end = time.time()
# print(end - start)
#
# print('# Saving feature vectors')
# with open('./Y_train-X_train--feature_vector--DBOW_100epochs_800vsize', 'wb') as f:
#     dump((Y_train_new, X_train_new), f)
#
# with open('./Y_test-X_test--feature_vector--DBOW_100epochs_800vsize', 'wb') as f:
#     dump((Y_test_new, X_test_new), f)

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

# print('## VERSION: DBOW_100epochs_600vsize')
#
# print('# Loading dataset')
# X_train, X_test, Y_train, Y_test = Reuters.get_random_split()
#
# train = list(zip(X_train,Y_train))
# test = list(zip(X_test,Y_test))
#
# print('# Transform to TaggedDocument')
# train_tagged = [TaggedDocument(words=tokenize(doc[0]), tags=doc[1]) for doc in train]
# test_tagged = [TaggedDocument(words=tokenize(doc[0]), tags=doc[1]) for doc in test]
#
# model = gensim.models.doc2vec.Doc2Vec(dm=0, vector_size=600, min_count=1, workers=3)
#
# model.build_vocab(train_tagged)
#
# print('# Training')
# start = time.time()
# # Lanzar warning sobre que pode tardar moito e que non ten ningún tipo de verbose
# model.train(train_tagged, total_examples=len(train_tagged), epochs=100)
# end = time.time()
# print(end - start)
#
# print('# Saving model')
# model.save('./doc2vec_DBOW_100epochs_600vsize')
#
# print('# Obtaining feature vectors of dataset')
# start = time.time()
# Y_train_new, X_train_new = feature_vector(model, train_tagged)
# Y_test_new, X_test_new = feature_vector(model, test_tagged)
# end = time.time()
# print(end - start)
#
# print('# Saving feature vectors')
# with open('./Y_train-X_train--feature_vector--DBOW_100epochs_600vsize', 'wb') as f:
#     dump((Y_train_new, X_train_new), f)
#
# with open('./Y_test-X_test--feature_vector--DBOW_100epochs_600vsize', 'wb') as f:
#     dump((Y_test_new, X_test_new), f)

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

# print('## VERSION: DBOW_70epochs_1000vsize')
#
# def feature_vector(model, tagged_docs):
#     targets, regressors = zip(*[(doc.tags, model.infer_vector(doc.words, steps=70)) for doc in tagged_docs])
#     return targets, regressors
#
# print('# Loading dataset')
# X_train, X_test, Y_train, Y_test = Reuters.get_random_split()
#
# train = list(zip(X_train,Y_train))
# test = list(zip(X_test,Y_test))
#
# print('# Transform to TaggedDocument')
# train_tagged = [TaggedDocument(words=tokenize(doc[0]), tags=doc[1]) for doc in train]
# test_tagged = [TaggedDocument(words=tokenize(doc[0]), tags=doc[1]) for doc in test]
#
# model = gensim.models.doc2vec.Doc2Vec(dm=0, vector_size=1000, min_count=1, workers=3)
#
# model.build_vocab(train_tagged)
#
# print('# Training')
# start = time.time()
# # Lanzar warning sobre que pode tardar moito e que non ten ningún tipo de verbose
# model.train(train_tagged, total_examples=len(train_tagged), epochs=70)
# end = time.time()
# print(end - start)
#
# print('# Saving model')
# model.save('./doc2vec_DBOW_70epochs_1000vsize')
#
# print('# Obtaining feature vectors of dataset')
# start = time.time()
# Y_train_new, X_train_new = feature_vector(model, train_tagged)
# Y_test_new, X_test_new = feature_vector(model, test_tagged)
# end = time.time()
# print(end - start)
#
# print('# Saving feature vectors')
# with open('./Y_train-X_train--feature_vector--DBOW_70epochs_1000vsize', 'wb') as f:
#     dump((Y_train_new, X_train_new), f)
#
# with open('./Y_test-X_test--feature_vector--DBOW_70epochs_1000vsize', 'wb') as f:
#     dump((Y_test_new, X_test_new), f)

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

# print('## VERSION: DBOW_50epochs_600vsize')
#
# def feature_vector(model, tagged_docs):
#     targets, regressors = zip(*[(doc.tags, model.infer_vector(doc.words, steps=50)) for doc in tagged_docs])
#     return targets, regressors
#
# print('# Loading dataset')
# X_train, X_test, Y_train, Y_test = Reuters.get_random_split()
#
# train = list(zip(X_train,Y_train))
# test = list(zip(X_test,Y_test))
#
# print('# Transform to TaggedDocument')
# train_tagged = [TaggedDocument(words=tokenize(doc[0]), tags=doc[1]) for doc in train]
# test_tagged = [TaggedDocument(words=tokenize(doc[0]), tags=doc[1]) for doc in test]
#
# model = gensim.models.doc2vec.Doc2Vec(dm=0, vector_size=600, min_count=1, workers=3)
#
# model.build_vocab(train_tagged)
#
# print('# Training')
# start = time.time()
# # Lanzar warning sobre que pode tardar moito e que non ten ningún tipo de verbose
# model.train(train_tagged, total_examples=len(train_tagged), epochs=50)
# end = time.time()
# print(end - start)
#
# print('# Saving model')
# model.save('./doc2vec_DBOW_50epochs_600vsize')
#
# print('# Obtaining feature vectors of dataset')
# start = time.time()
# Y_train_new, X_train_new = feature_vector(model, train_tagged)
# Y_test_new, X_test_new = feature_vector(model, test_tagged)
# end = time.time()
# print(end - start)
#
# print('# Saving feature vectors')
# with open('./Y_train-X_train--feature_vector--DBOW_50epochs_600vsize', 'wb') as f:
#     dump((Y_train_new, X_train_new), f)
#
# with open('./Y_test-X_test--feature_vector--DBOW_50epochs_600vsize', 'wb') as f:
#     dump((Y_test_new, X_test_new), f)

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

# print('## VERSION: DBOW_40epochs_600vsize')
#
# def feature_vector(model, tagged_docs):
#     targets, regressors = zip(*[(doc.tags, model.infer_vector(doc.words, steps=40)) for doc in tagged_docs])
#     return targets, regressors
#
# print('# Loading dataset')
# X_train, X_test, Y_train, Y_test = Reuters.get_random_split()
#
# train = list(zip(X_train,Y_train))
# test = list(zip(X_test,Y_test))
#
# print('# Transform to TaggedDocument')
# train_tagged = [TaggedDocument(words=tokenize(doc[0]), tags=doc[1]) for doc in train]
# test_tagged = [TaggedDocument(words=tokenize(doc[0]), tags=doc[1]) for doc in test]
#
# model = gensim.models.doc2vec.Doc2Vec(dm=0, vector_size=600, min_count=1, workers=3)
#
# model.build_vocab(train_tagged)
#
# print('# Training')
# start = time.time()
# # Lanzar warning sobre que pode tardar moito e que non ten ningún tipo de verbose
# model.train(train_tagged, total_examples=len(train_tagged), epochs=40)
# end = time.time()
# print(end - start)
#
# print('# Saving model')
# model.save('./doc2vec_DBOW_40epochs_600vsize')
#
# print('# Obtaining feature vectors of dataset')
# start = time.time()
# Y_train_new, X_train_new = feature_vector(model, train_tagged)
# Y_test_new, X_test_new = feature_vector(model, test_tagged)
# end = time.time()
# print(end - start)
#
# print('# Saving feature vectors')
# with open('./Y_train-X_train--feature_vector--DBOW_40epochs_600vsize', 'wb') as f:
#     dump((Y_train_new, X_train_new), f)
#
# with open('./Y_test-X_test--feature_vector--DBOW_40epochs_600vsize', 'wb') as f:
#     dump((Y_test_new, X_test_new), f)

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

# print('## VERSION: DBOW_30epochs_600vsize')
#
# def feature_vector(model, tagged_docs):
#     targets, regressors = zip(*[(doc.tags, model.infer_vector(doc.words, steps=30)) for doc in tagged_docs])
#     return targets, regressors
#
# print('# Loading dataset')
# X_train, X_test, Y_train, Y_test = Reuters.get_random_split()
#
# train = list(zip(X_train,Y_train))
# test = list(zip(X_test,Y_test))
#
# print('# Transform to TaggedDocument')
# train_tagged = [TaggedDocument(words=tokenize(doc[0]), tags=doc[1]) for doc in train]
# test_tagged = [TaggedDocument(words=tokenize(doc[0]), tags=doc[1]) for doc in test]
#
# model = gensim.models.doc2vec.Doc2Vec(dm=0, vector_size=600, min_count=1, workers=3)
#
# model.build_vocab(train_tagged)
#
# print('# Training')
# start = time.time()
# # Lanzar warning sobre que pode tardar moito e que non ten ningún tipo de verbose
# model.train(train_tagged, total_examples=len(train_tagged), epochs=30)
# end = time.time()
# print(end - start)
#
# print('# Saving model')
# model.save('./doc2vec_DBOW_30epochs_600vsize')
#
# print('# Obtaining feature vectors of dataset')
# start = time.time()
# Y_train_new, X_train_new = feature_vector(model, train_tagged)
# Y_test_new, X_test_new = feature_vector(model, test_tagged)
# end = time.time()
# print(end - start)
#
# print('# Saving feature vectors')
# with open('./Y_train-X_train--feature_vector--DBOW_30epochs_600vsize', 'wb') as f:
#     dump((Y_train_new, X_train_new), f)
#
# with open('./Y_test-X_test--feature_vector--DBOW_30epochs_600vsize', 'wb') as f:
#     dump((Y_test_new, X_test_new), f)

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


# print('## VERSION: DBOW_25epochs_600vsize')
#
# def feature_vector(model, tagged_docs):
#     targets, regressors = zip(*[(doc.tags, model.infer_vector(doc.words, steps=25)) for doc in tagged_docs])
#     return targets, regressors
#
# print('# Loading dataset')
# X_train, X_test, Y_train, Y_test = Reuters.get_random_split()
#
# train = list(zip(X_train,Y_train))
# test = list(zip(X_test,Y_test))
#
# print('# Transform to TaggedDocument')
# train_tagged = [TaggedDocument(words=tokenize(doc[0]), tags=doc[1]) for doc in train]
# test_tagged = [TaggedDocument(words=tokenize(doc[0]), tags=doc[1]) for doc in test]
#
# model = gensim.models.doc2vec.Doc2Vec(dm=0, vector_size=600, min_count=1, workers=3)
#
# model.build_vocab(train_tagged)
#
# print('# Training')
# start = time.time()
# # Lanzar warning sobre que pode tardar moito e que non ten ningún tipo de verbose
# model.train(train_tagged, total_examples=len(train_tagged), epochs=25)
# end = time.time()
# print(end - start)
#
# print('# Saving model')
# model.save('./doc2vec_DBOW_25epochs_600vsize')
#
# print('# Obtaining feature vectors of dataset')
# start = time.time()
# Y_train_new, X_train_new = feature_vector(model, train_tagged)
# Y_test_new, X_test_new = feature_vector(model, test_tagged)
# end = time.time()
# print(end - start)
#
# print('# Saving feature vectors')
# with open('./Y_train-X_train--feature_vector--DBOW_25epochs_600vsize', 'wb') as f:
#     dump((Y_train_new, X_train_new), f)
#
# with open('./Y_test-X_test--feature_vector--DBOW_25epochs_600vsize', 'wb') as f:
#     dump((Y_test_new, X_test_new), f)

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

# print('## VERSION: DBOW_30epochs_700vsize')
#
# def feature_vector(model, tagged_docs):
#     targets, regressors = zip(*[(doc.tags, model.infer_vector(doc.words, steps=30)) for doc in tagged_docs])
#     return targets, regressors
#
# print('# Loading dataset')
# X_train, X_test, Y_train, Y_test = Reuters.get_random_split()
#
# train = list(zip(X_train,Y_train))
# test = list(zip(X_test,Y_test))
#
# print('# Transform to TaggedDocument')
# train_tagged = [TaggedDocument(words=tokenize(doc[0]), tags=doc[1]) for doc in train]
# test_tagged = [TaggedDocument(words=tokenize(doc[0]), tags=doc[1]) for doc in test]
#
# model = gensim.models.doc2vec.Doc2Vec(dm=0, vector_size=700, min_count=1, workers=3)
#
# model.build_vocab(train_tagged)
#
# print('# Training')
# start = time.time()
# # Lanzar warning sobre que pode tardar moito e que non ten ningún tipo de verbose
# model.train(train_tagged, total_examples=len(train_tagged), epochs=30)
# end = time.time()
# print(end - start)
#
# print('# Saving model')
# model.save('./doc2vec_DBOW_30epochs_700vsize')
#
# print('# Obtaining feature vectors of dataset')
# start = time.time()
# Y_train_new, X_train_new = feature_vector(model, train_tagged)
# Y_test_new, X_test_new = feature_vector(model, test_tagged)
# end = time.time()
# print(end - start)
#
# print('# Saving feature vectors')
# with open('./Y_train-X_train--feature_vector--DBOW_30epochs_700vsize', 'wb') as f:
#     dump((Y_train_new, X_train_new), f)
#
# with open('./Y_test-X_test--feature_vector--DBOW_30epochs_700vsize', 'wb') as f:
#     dump((Y_test_new, X_test_new), f)

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

print('## VERSION: DBOW_20epochs_600vsize_40fv')

def feature_vector(model, tagged_docs):
    targets, regressors = zip(*[(doc.tags, model.infer_vector(doc.words, steps=40)) for doc in tagged_docs])
    return targets, regressors

print('# Loading dataset')
X_train, X_test, Y_train, Y_test = Reuters.get_random_split()

train = list(zip(X_train,Y_train))
test = list(zip(X_test,Y_test))

print('# Transform to TaggedDocument')
train_tagged = [TaggedDocument(words=tokenize(doc[0]), tags=doc[1]) for doc in train]
test_tagged = [TaggedDocument(words=tokenize(doc[0]), tags=doc[1]) for doc in test]

model = gensim.models.doc2vec.Doc2Vec(dm=0, vector_size=600, min_count=1, workers=3)

model.build_vocab(train_tagged)

print('# Training')
start = time.time()
# Lanzar warning sobre que pode tardar moito e que non ten ningún tipo de verbose
model.train(train_tagged, total_examples=len(train_tagged), epochs=20)
end = time.time()
print(end - start)

print('# Saving model')
model.save('./doc2vec_DBOW_20epochs_600vsize_40fv')

print('# Obtaining feature vectors of dataset')
start = time.time()
Y_train_new, X_train_new = feature_vector(model, train_tagged)
Y_test_new, X_test_new = feature_vector(model, test_tagged)
end = time.time()
print(end - start)

print('# Saving feature vectors')
with open('./Y_train-X_train--feature_vector--DBOW_20epochs_600vsize_40fv', 'wb') as f:
    dump((Y_train_new, X_train_new), f)

with open('./Y_test-X_test--feature_vector--DBOW_20epochs_600vsize_40fv', 'wb') as f:
    dump((Y_test_new, X_test_new), f)

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

# print('## VERSION: DBOW_20epochs_600vsize_100fv')
#
# def feature_vector(model, tagged_docs):
#     targets, regressors = zip(*[(doc.tags, model.infer_vector(doc.words, steps=100)) for doc in tagged_docs])
#     return targets, regressors
#
# print('# Loading dataset')
# X_train, X_test, Y_train, Y_test = Reuters.get_random_split()
#
# train = list(zip(X_train,Y_train))
# test = list(zip(X_test,Y_test))
#
# print('# Transform to TaggedDocument')
# train_tagged = [TaggedDocument(words=tokenize(doc[0]), tags=doc[1]) for doc in train]
# test_tagged = [TaggedDocument(words=tokenize(doc[0]), tags=doc[1]) for doc in test]
#
# model = gensim.models.doc2vec.Doc2Vec(dm=0, vector_size=600, min_count=1, workers=3)
#
# model.build_vocab(train_tagged)
#
# print('# Training')
# start = time.time()
# # Lanzar warning sobre que pode tardar moito e que non ten ningún tipo de verbose
# model.train(train_tagged, total_examples=len(train_tagged), epochs=20)
# end = time.time()
# print(end - start)
#
# print('# Saving model')
# model.save('./doc2vec_DBOW_20epochs_600vsize_100fv')
#
# print('# Obtaining feature vectors of dataset')
# start = time.time()
# Y_train_new, X_train_new = feature_vector(model, train_tagged)
# Y_test_new, X_test_new = feature_vector(model, test_tagged)
# end = time.time()
# print(end - start)
#
# print('# Saving feature vectors')
# with open('./Y_train-X_train--feature_vector--DBOW_20epochs_600vsize_100fv', 'wb') as f:
#     dump((Y_train_new, X_train_new), f)
#
# with open('./Y_test-X_test--feature_vector--DBOW_20epochs_600vsize_100fv', 'wb') as f:
#     dump((Y_test_new, X_test_new), f)

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

# print('## VERSION: DBOW_15epochs_600vsize_60fv')
#
# def feature_vector(model, tagged_docs):
#     targets, regressors = zip(*[(doc.tags, model.infer_vector(doc.words, steps=60)) for doc in tagged_docs])
#     return targets, regressors
#
# print('# Loading dataset')
# X_train, X_test, Y_train, Y_test = Reuters.get_random_split()
#
# train = list(zip(X_train,Y_train))
# test = list(zip(X_test,Y_test))
#
# print('# Transform to TaggedDocument')
# train_tagged = [TaggedDocument(words=tokenize(doc[0]), tags=doc[1]) for doc in train]
# test_tagged = [TaggedDocument(words=tokenize(doc[0]), tags=doc[1]) for doc in test]
#
# model = gensim.models.doc2vec.Doc2Vec(dm=0, vector_size=600, min_count=1, workers=3)
#
# model.build_vocab(train_tagged)
#
# print('# Training')
# start = time.time()
# # Lanzar warning sobre que pode tardar moito e que non ten ningún tipo de verbose
# model.train(train_tagged, total_examples=len(train_tagged), epochs=15)
# end = time.time()
# print(end - start)
#
# print('# Saving model')
# model.save('./doc2vec_DBOW_15epochs_600vsize_60fv')
#
# print('# Obtaining feature vectors of dataset')
# start = time.time()
# Y_train_new, X_train_new = feature_vector(model, train_tagged)
# Y_test_new, X_test_new = feature_vector(model, test_tagged)
# end = time.time()
# print(end - start)
#
# print('# Saving feature vectors')
# with open('./Y_train-X_train--feature_vector--DBOW_15epochs_600vsize_60fv', 'wb') as f:
#     dump((Y_train_new, X_train_new), f)
#
# with open('./Y_test-X_test--feature_vector--DBOW_15epochs_600vsize_60fv', 'wb') as f:
#     dump((Y_test_new, X_test_new), f)

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

# print('## VERSION: DBOW_15epochs_600vsize_100fv')
#
# def feature_vector(model, tagged_docs):
#     targets, regressors = zip(*[(doc.tags, model.infer_vector(doc.words, steps=100)) for doc in tagged_docs])
#     return targets, regressors
#
# print('# Loading dataset')
# X_train, X_test, Y_train, Y_test = Reuters.get_random_split()
#
# train = list(zip(X_train,Y_train))
# test = list(zip(X_test,Y_test))
#
# print('# Transform to TaggedDocument')
# train_tagged = [TaggedDocument(words=tokenize(doc[0]), tags=doc[1]) for doc in train]
# test_tagged = [TaggedDocument(words=tokenize(doc[0]), tags=doc[1]) for doc in test]
#
# model = gensim.models.doc2vec.Doc2Vec(dm=0, vector_size=600, min_count=1, workers=3)
#
# model.build_vocab(train_tagged)
#
# print('# Training')
# start = time.time()
# # Lanzar warning sobre que pode tardar moito e que non ten ningún tipo de verbose
# model.train(train_tagged, total_examples=len(train_tagged), epochs=15)
# end = time.time()
# print(end - start)
#
# print('# Saving model')
# model.save('./doc2vec_DBOW_15epochs_600vsize_100fv')
#
# print('# Obtaining feature vectors of dataset')
# start = time.time()
# Y_train_new, X_train_new = feature_vector(model, train_tagged)
# Y_test_new, X_test_new = feature_vector(model, test_tagged)
# end = time.time()
# print(end - start)
#
# print('# Saving feature vectors')
# with open('./Y_train-X_train--feature_vector--DBOW_15epochs_600vsize_100fv', 'wb') as f:
#     dump((Y_train_new, X_train_new), f)
#
# with open('./Y_test-X_test--feature_vector--DBOW_15epochs_600vsize_100fv', 'wb') as f:
#     dump((Y_test_new, X_test_new), f)

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++