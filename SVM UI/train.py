import pandas as pd
import os
from werkzeug.utils import secure_filename
import re
import string
import nltk
import numpy as np
nltk.download('punkt')
nltk.download('stopwords')
from nltk.tokenize import sent_tokenize, word_tokenize

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer

from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
import sklearn.metrics as metrics

from sklearn.model_selection import train_test_split
import pickle

class file_input():
  df_ulasan = pd.read_excel('static/review_shopee.xlsx')
  print(df_ulasan.head())
  Label = []
  for index, row in df_ulasan.iterrows():
      if row["Kelas"] == "Negatif":
          Label.append(0)
      else:
          Label.append(1)
  df_ulasan["Label"] = Label
  print(df_ulasan.tail())
  # Sudah dalam bentuk string semua
  for i in range(len(df_ulasan)):
    if type(df_ulasan["Ulasan"][i]) != str:
        print(str(i), "Bukan string")

inputan = file_input()

# Case Folding & Noise Removal
class cleaning:

  def clean_ulasan(self, input):
      text = input.copy(deep=True)
      for i in range(len(text)):
        text[i] = text[i].lower() # menjadikan lowercase
        text[i] = re.sub("[^a-z]", " ", str(text[i])) # hapus semua karakter kecuali a-z
        text[i] = re.sub("\t", " ", str(text[i])) # mengganti tab dengan spasi
        text[i] = re.sub("\n", " ", str(text[i])) # mengganti new line dengan spasi
        text[i] = re.sub("\s+", " ", str(text[i])) # mengganti spasi > 1 dengan 1 spasi
        text[i] = text[i].strip() # menghapus spasi di awal dan akhir

      return text

perbaikan = cleaning()

inputan.df_ulasan["ulasan_clean"] = perbaikan.clean_ulasan(inputan.df_ulasan["Ulasan"])

# Sastrawi
class Sastrawi_Stopwords:
  from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

  factory = StopWordRemoverFactory()
  stopword_sastrawi = factory.get_stop_words()
  print(len(stopword_sastrawi))

  from nltk.corpus import stopwords
  stopword_nltk = set(stopwords.words("indonesian"))
  print(len(stopword_nltk)) # Lihat berapa banyak kata yang termasuk dalam stopword NLTK

  df1 = pd.DataFrame(stopword_nltk)
  df2 = pd.DataFrame(stopword_sastrawi)

  common = df1.merge(df2, on = [0], how = "left")
  print(common)

  ga_ada = df2[(~df2[0].isin(common[0]))] # ~ -> negasi (True jadi False)
  print(ga_ada)
  # Ada 17 kata stopword di Sastrawi yang tidak ada di nltk

class normal:
    def normalisasi(self, text):
        import re

        dict_koreksi = {}
        file = open("static/list_spell_check.txt")
        for x in file:
            f = x.split(":")
            dict_koreksi.update({f[0].strip(): f[1].strip()})

        for awal, pengganti in dict_koreksi.items():
            #text = str(text).replace(awal, pengganti)
            text = re.sub(r"\b" + awal + r"\b", pengganti, text)

        return text

norm = normal()

inputan.df_ulasan["ulasan_clean"] = inputan.df_ulasan["ulasan_clean"].apply(norm.normalisasi)

# Filtering (Stopword Removal)
class filtering:
  def clean_stopword(self, text):
    # Stopword Sastrawi
    from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

    factory = StopWordRemoverFactory()
    stopword_sastrawi = factory.get_stop_words()

    text = text.split() # split jadi kata per kata
    text = [w for w in text if w not in stopword_sastrawi] # hapus stopwords
    text = " ".join(w for w in text) # join semua kata yang bukan stopwords

    # Stopword NLTK
    import nltk
    #nltk.download()
    from nltk.corpus import stopwords

    stopword_nltk = set(stopwords.words("indonesian"))
    stopword_nltk = stopword_nltk

    text = text.split() # split jadi kata per kata
    text = [w for w in text if w not in stopword_nltk] # hapus stopwords
    text = " ".join(w for w in text) # join semua kata yang bukan stopwords

    return text

    # Stopword Tambahan

  def clean_stopword_tambahan(self, text):
    with open("static/list_stopword_tambahan.txt", "r") as f:
        stopwords_tambahan = f.read().splitlines()

    text = text.split() # split jadi kata per kata
    text = [w for w in text if w not in stopwords_tambahan] # hapus stopwords
    text = " ".join(w for w in text) # join semua kata yang bukan stopwords

    return text

bersih = filtering()
inputan.df_ulasan["ulasan_clean"] = inputan.df_ulasan["ulasan_clean"].apply(bersih.clean_stopword)
inputan.df_ulasan["ulasan_clean"] = inputan.df_ulasan["ulasan_clean"].apply(bersih.clean_stopword_tambahan)

# Stemming
class stemming:
  def clean_stem(self, text):
    # Stemming Sastrawi
    from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

    factory = StemmerFactory()
    stemmer_sastrawi = factory.create_stemmer()

    text = stemmer_sastrawi.stem(text)

    return text

stemmer = stemming()
inputan.df_ulasan["ulasan_clean"] = inputan.df_ulasan["ulasan_clean"].apply(stemmer.clean_stem)

#TF IDF
class bobotTFIDF:

  def tf_idf(self, text):
    cv = CountVectorizer()
    word_count_vector = cv.fit_transform(text)
    tf = pd.DataFrame(word_count_vector.todense().T,
                        index=cv.get_feature_names_out(),
                        columns=[f'D{i+1}' for i in range(len(text))])
    print(tf)

    tfidf_transformer = TfidfTransformer(norm=None)
    X = tfidf_transformer.fit_transform(word_count_vector)
    idf = pd.DataFrame({'feature_name':cv.get_feature_names_out(), 'idf_weights':tfidf_transformer.idf_})
    print(idf)

    tf_idf = pd.DataFrame(X.todense().T,
                        index=cv.get_feature_names_out(),
                        columns=[f'D{i+1}' for i in range(len(text))])

    print(tf_idf)

    return X

TFIDF = bobotTFIDF()

x = TFIDF.tf_idf(inputan.df_ulasan['ulasan_clean'])
y = inputan.df_ulasan['Label']

# Modelling SVM
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.3, stratify=y, random_state=42)
class modelSVM:

  def klasifikasi_linear(self):
    # menentukan range parameter
    param_grid = {'C': [0.1, 1, 10, 100, 1000],
                  'kernel': ['linear']}
    grid = GridSearchCV(SVC(), param_grid, refit = True, verbose = 3)
    # sesuaikan model dengan data
    grid.fit(x_train, y_train)
    # print parameter terbaik
    print(grid.best_params_)

    # print bagaimana model kita terlihat setelah penyetelan hyper-parameter
    print(grid.best_estimator_)

    grid_predictions = grid.predict(x_test)

    # print hasil klasifikasi
    akurasi = metrics.accuracy_score(y_test, grid_predictions)
    print(f"Akurasi dari : ", akurasi)
    cm = classification_report(y_test, grid_predictions)
    print(cm)
    print(type(cm))

    with open("modelsvmlinear.pkl", "wb") as f: pickle.dump(grid, f)

  def klasifikasi_rbf(self):
    # menentukan range parameter
    param_grid = {'C': [0.1, 1, 10, 100, 1000],
                  'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
                  'kernel': ['rbf']}
    grid = GridSearchCV(SVC(), param_grid, refit = True, verbose = 3)
    # sesuaikan model dengan data
    grid.fit(x_train, y_train)
    # print parameter terbaik
    print(grid.best_params_)

    # print bagaimana model kita terlihat setelah penyetelan hyper-parameter
    print(grid.best_estimator_)

    grid_predictions = grid.predict(x_test)

    # print hasil klasifikasi
    print(classification_report(y_test, grid_predictions))

    with open("modelsvmrbf.pkl", "wb") as f: pickle.dump(grid, f)

  def klasifikasi_polinomial(self):
    # menentukan range parameter
    param_grid = {'C': [0.1, 1, 10, 100, 1000],
                  'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
                  'degree': [2,3,4],
                  'kernel': ['poly']}
    grid = GridSearchCV(SVC(), param_grid, refit = True, verbose = 3)
    # sesuaikan model dengan data
    grid.fit(x_train, y_train)
    # print parameter terbaik
    print(grid.best_params_)

    # print bagaimana model kita terlihat setelah penyetelan hyper-parameter
    print(grid.best_estimator_)

    grid_predictions = grid.predict(x_test)

    # print hasil klasifikasi

    print(classification_report(y_test, grid_predictions))

    with open("modelsvmpol.pkl", "wb") as f: pickle.dump(grid, f)

klasifikasi = modelSVM()
klasifikasi.klasifikasi_linear()
klasifikasi.klasifikasi_rbf()
klasifikasi.klasifikasi_polinomial()