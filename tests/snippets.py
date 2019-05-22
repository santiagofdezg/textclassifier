#-###################
#-## TEXT MINING

# nltk - Natural Language Toolkit  (ya contiene algoritmos de clasificación implementados, incluido WEKA)

import nltk
#Solo necesario la primera vez
#nltk.download('stopwords')
#nltk.download('punkt')
#nltk.download('averaged_perceptron_tagger')
#nltk.download('wordnet')
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.probability import FreqDist
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer #Transforma una palabra en su raíz
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import wordnet

frases = sent_tokenize(texto)
palabras = word_tokenize(texto.lower())

# Frecuencia de palabras
FreqDist(palabras).most_common(20)

#stopwords = conjunto de palabras más básico y menos informativo
stop_word = stopwords.words('spanish')
#Ahora vamos a sacar estas palabras y los signos de puntuación de nuestra lista
palabras = [x for x in palabras if x not in stop_word + list(string.punctuation)] #lista de palabras con significado

#Obtener la raíz de las palabras
stemmer = SnowballStemmer('spanish')
raices = []
for palabra in palabras:
    raices.append(stemmer.stem(palabra))
#De esta forma acabamos obteniendo una "bag of words"

#Reducir una palabra considerando su posición en una frase: dependiendo
# del tipo que es (verbo,nombre...) lo reduce de una forma u otra
lem = WordNetLemmatizer()
lem.lemmatize('running','n')

#Obtener una lista con las palabras y su posición: tipo de palabra, si es plural o singular, etc.
nltk.pos_tag(palabras)

#Simplifica la variabilidad que devuelve pos_tag() a algo que entienda lemmatize()
def get_wordnet_pos(word):
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {'J': wordnet.ADJ,
               'N': wordnet.NOUN,
               'V': wordnet.VERB,
               'R': wordnet.ADV}
    return tag_dict.get(tag, wordnet.NOUN)

for palabra in nltk.word_tokenize(frase):
        print(lem.lemmatize(palabra, get_wordnet_pos(palabra)))


#-#######################
#-## SENTIMENT ANALYSIS
from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import RegexpTokenizer
import pandas as pd

lista1 = ['Bueno','Malo','Malo','Malo','Bueno','Bueno','Malo','Bueno','Bueno','Malo']
lista2 = ['Lo recomendaría a todos mis amigos',
          'Es el peor producto que he comprado nunca',
          'Ni loco compraría este producto',
          'No se lo recomendaría ni a mi enemigo',
          'Es un buen producto, sí que lo recomendaría',
          'Me ha encantado',
          'Es una basura absoluta, ni me molesté en lavarlo, lo tiré directamente',
          'El enemigo público número uno de la suciedad',
          'Es un producto genial, se lo recomendaría a todos los compradores',
          'Hay que estar loco para comprar esta basura']

# Convertir las listas en un dataframe
df = pd.DataFrame({'Sentimiento':lista1, 'Valoracion':lista2})

# tokenizer que elimina todo lo que no sean letras o números
token = RegexpTokenizer(r'[a-zA-Z0-9]+')
cv = CountVectorizer(lowercase=True, ngram_range=(1,2), tokenizer=token.tokenize)
                #ngram_range: que rango de palabras queremos, es decir, indica que unidades vamos
                            # a considerar como unidades de significado. En este caso (1,2) indica que
                            # va a analizar todas las palabras por separado y también en parejas
                #tokenizer: creará las unidades basándose en la expresión regular del tokenizer
# Entrena y transforma la columna de valoraciones
text_counts = cv.fit_transform(df['Valoracion'])
    # devuelve una lista con las palabras y parejas de palabras y el número de veces que aparecen.


#-#######################
#-## ENTRENAMIENTO DE UN MODELO PARA CLASIFICACIÓN AUTOMÁTICA DE TEXTOS

# Vamos a ajustar un modelo para el ejemplo anterior de sentiment analysis
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics

X_train,X_test,y_train,y_test = train_test_split(text_counts, df['Sentimiento'], test_size=.5, random_state=1)
        # variable regresora -> text_counts
        # respuesta -> el sentimiento
        # random_state = semilla

clf = MultinomialNB().fit(X_train, y_train)

predicted = clf.predict(X_test)

print('MultinomialNB Accuracy: ', metrics.accuracy_score(y_test, predicted))


#-#######################
#-## TOPIC MODELING

from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk import sent_tokenize, word_tokenize
from nltk.stem import SnowballStemmer
import pandas as pd
import gensim
from gensim import corpora, models

texto = open("D:\\Development_Projects\\curso-python-data-scientist-avanzado\\nelson_mandela.txt","r",encoding='utf8').read()
frases = sent_tokenize(texto)
frases = pd.DataFrame({'Frases':frases})

stemmer = SnowballStemmer('english') #sirve para obtener la raíz de las palabras

#Función que elimina las stop words y palabras con longitud menor o igual que 3
def preproceso(texto):
    resultado = []
    for token in gensim.utils.simple_preprocess(texto):
        if token not in gensim.parsing.preprocessing.STOPWORDS and len(token)>3:
            resultado.append(stemmer.stem(token))
    return resultado

frasesProcesadas = frases['Frases'].map(preproceso)


#Creamos un diccionario con la lista de palabras que hay en el texto
dictionary = gensim.corpora.Dictionary(frasesProcesadas)
dictionary.filter_extremes(no_below=3, no_above=.5, keep_n=100000) #Solo palabras que aparezcan un mínimo de 3 veces y que no ocupen más del 50% de las palabras que aparecen en una frase

corpus = [dictionary.doc2bow(doc) for doc in frasesProcesadas]
    #El corpus nos dice qué palabras del diccionario aparecen en qué documento

#Corpus alternativo: dá más peso a aquellas palabras más representativas y compensa el efecto de que haya muchos documentos muy largos con muchas palabras
tfidf = models.TfidfModel(corpus) #tfidf = term frequency–inverse document frequency
corpus_tfidf = tfidf[corpus]


#lda = Latent Dirichlet Allocation -> detecta los temas que subyacen a determinados textos
#lda_model = gensim.models.LdaMulticore(corpus, num_topics=10, id2word=dictionary, passes=5, workers=3)
                                        #num_topics = número de temas a encontrar
                                        #id2word = diccionario para encontrarlos
                                        #passes = cuantas veces tiene que iterar sobre los datos
                                        #workers = para paralelización
#alternativa:
lda_model = gensim.models.LdaMulticore(corpus_tfidf, num_topics=10, id2word=dictionary, passes=5, workers=3)
for idx, topic in lda_model.print_topics(-1):
    print('Tema: {} \nPalabras: {}\n'.format(idx, topic))


