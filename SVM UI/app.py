from flask import Flask, render_template, request, jsonify
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
	def __init__(self, masukan):
		self.df_ulasan = pd.read_excel(masukan)
		Label = []
		for index, row in self.df_ulasan.iterrows():
			if row["Kelas"] == "Negatif":
				Label.append(0)
			else:
				Label.append(1)
		self.df_ulasan["Label"] = Label
  
		for i in range(len(self.df_ulasan)):
			if type(self.df_ulasan["Ulasan"][i]) != str:
				print(str(i), "Bukan string")

# Case Folding & Noise Removal
class cleaning():

  def clean_ulasan(self, text):
        text = text.lower() # menjadikan lowercase
        text = re.sub("[^a-z]", " ", str(text)) # hapus semua karakter kecuali a-z
        text = re.sub("\t", " ", str(text)) # mengganti tab dengan spasi
        text = re.sub("\n", " ", str(text)) # mengganti new line dengan spasi
        text = re.sub("\s+", " ", str(text)) # mengganti spasi > 1 dengan 1 spasi
        text = text.strip() # menghapus spasi di awal dan akhir

        return text
		
class cleaning_train():

  def clean_ulasan_train(self, input):
      text = input.copy(deep=True)
      for i in range(len(text)):
        text[i] = text[i].lower() # menjadikan lowercase
        text[i] = re.sub("[^a-z]", " ", str(text[i])) # hapus semua karakter kecuali a-z
        text[i] = re.sub("\t", " ", str(text[i])) # mengganti tab dengan spasi
        text[i] = re.sub("\n", " ", str(text[i])) # mengganti new line dengan spasi
        text[i] = re.sub("\s+", " ", str(text[i])) # mengganti spasi > 1 dengan 1 spasi
        text[i] = text[i].strip() # menghapus spasi di awal dan akhir

      return text

class normal:
  def normalisasi(self, text):
    import re

    dict_koreksi = {}
    file = open("./static/list_spell_check.txt")
    for x in file:
        f = x.split(":")
        dict_koreksi.update({f[0].strip(): f[1].strip()})

    for awal, pengganti in dict_koreksi.items():
        #text = str(text).replace(awal, pengganti)
        text = re.sub(r"\b" + awal + r"\b", pengganti, text)

    return text
	
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
    with open("./static/list_stopword_tambahan.txt", "r") as f:
        stopwords_tambahan = f.read().splitlines()

    text = text.split() # split jadi kata per kata
    text = [w for w in text if w not in stopwords_tambahan] # hapus stopwords
    text = " ".join(w for w in text) # join semua kata yang bukan stopwords

    return text

# Stemming
class Stemming:
  def clean_stem(self, text):
    # Stemming Sastrawi
    from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

    factory = StemmerFactory()
    stemmer_sastrawi = factory.create_stemmer()

    text = stemmer_sastrawi.stem(text)

    return text

#TF IDF
class bobotTFIDF:

  def tf_idf(self, input):
    text_raw = pd.read_excel("./static/data_stem.xlsx")
    x = pd.DataFrame({'Ulasan' : [' '], 'Kelas' : ['Negatif'], 'Label' : [0], 'ulasan_clean' : [input]})
    text_raw = pd.concat([text_raw, x], ignore_index=True)
    text = text_raw['ulasan_clean']
    
    cv = CountVectorizer()
    word_count_vector = cv.fit_transform(text)
    tf = pd.DataFrame(word_count_vector.todense().T,
                        index=cv.get_feature_names_out(),
                        columns=[f'D{i+1}' for i in range(len(text))])
    #print(tf)

    tfidf_transformer = TfidfTransformer(norm=None)
    X = tfidf_transformer.fit_transform(word_count_vector)
    idf = pd.DataFrame({'feature_name':cv.get_feature_names_out(), 'idf_weights':tfidf_transformer.idf_})
    #print(idf)

    tf_idf = pd.DataFrame(X.todense().T,
                        index=cv.get_feature_names_out(),
                        columns=[f'D{i+1}' for i in range(len(text))])

    #print(tf_idf)
    
    return X
	
class bobotTFIDF_train:

  def tf_idf_train(self, text):
    cv = CountVectorizer()
    word_count_vector = cv.fit_transform(text)
    tf = pd.DataFrame(word_count_vector.todense().T,
                        index=cv.get_feature_names_out(),
                        columns=[f'D{i+1}' for i in range(len(text))])
    #print(tf)

    tfidf_transformer = TfidfTransformer(norm=None)
    X = tfidf_transformer.fit_transform(word_count_vector)
    idf = pd.DataFrame({'feature_name':cv.get_feature_names_out(), 'idf_weights':tfidf_transformer.idf_})
    #print(idf)

    tf_idf = pd.DataFrame(X.todense().T,
                        index=cv.get_feature_names_out(),
                        columns=[f'D{i+1}' for i in range(len(text))])

    #print(tf_idf)

    return X
	
class modelSVM:

  def klasifikasi_linear(self):
    # defining parameter range
    param_grid = {'C': [0.1, 1, 10, 100, 1000],
                  'kernel': ['linear']}
    grid = GridSearchCV(SVC(), param_grid, scoring = 'accuracy', refit = True, verbose = 3, cv = 5, return_train_score = False)
    # fitting the model for grid search
    grid.fit(x_train, y_train)
    # print(grid.cv_results_)
    df_result = pd.DataFrame(grid.cv_results_)
    df_result.to_excel("static/result_linear.xlsx", index = False)
    # print best parameter after tuning
    print(grid.best_params_)

    # print how our model looks after hyper-parameter tuning
    print(grid.best_estimator_)

    grid_predictions = grid.predict(x_test)

    cm = confusion_matrix(y_test, grid_predictions)
    print(cm)

    cr = classification_report(y_test, grid_predictions, output_dict=True)
    # print classification report
    # return(classification_report(y_test, grid_predictions))
    with open("model/modelsvmlinear.pkl", "wb") as f: pickle.dump(grid, f)
    return cr

  def klasifikasi_rbf(self):
    # defining parameter range
    param_grid = {'C': [0.1, 1, 10, 100, 1000],
                  'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
                  'kernel': ['rbf']}
    grid = GridSearchCV(SVC(), param_grid, scoring = 'accuracy', refit = True, verbose = 3, cv = 5, return_train_score = True)
    # fitting the model for grid search
    grid.fit(x_train, y_train)
    # print(grid.cv_results_)
    df_result = pd.DataFrame(grid.cv_results_)
    df_result.to_excel("static/result_rbf.xlsx", index = False)
    # print best parameter after tuning
    print(grid.best_params_)

    # print how our model looks after hyper-parameter tuning
    print(grid.best_estimator_)

    grid_predictions = grid.predict(x_test)

    cm = confusion_matrix(y_test, grid_predictions)
    print(cm)

    cr = classification_report(y_test, grid_predictions, output_dict=True)
    # print classification report
    # return(classification_report(y_test, grid_predictions))
    with open("model/modelsvmrbf.pkl", "wb") as f: pickle.dump(grid, f)
    return cr

  def klasifikasi_polinomial(self):
    # defining parameter range
    param_grid = {'C': [0.1, 1, 10, 100, 1000],
                  'degree': [2,3,4],
                  'kernel': ['poly']}
    grid = GridSearchCV(SVC(), param_grid, scoring = 'accuracy', refit = True, verbose = 3, cv = 5, return_train_score = False)
    # fitting the model for grid search
    grid.fit(x_train, y_train)
    # print(grid.cv_results_)
    df_result = pd.DataFrame(grid.cv_results_)
    df_result.to_excel("static/result_poly.xlsx", index = False)
    # print best parameter after tuning
    print(grid.best_params_)

    # print how our model looks after hyper-parameter tuning
    print(grid.best_estimator_)

    grid_predictions = grid.predict(x_test)

    cm = confusion_matrix(y_test, grid_predictions)
    print(cm)
    cr = classification_report(y_test, grid_predictions, output_dict=True)
    # print classification report
    # return(classification_report(y_test, grid_predictions))
    with open("model/modelsvmpol.pkl", "wb") as f: pickle.dump(grid, f)
    return cr


app = Flask(__name__)

@app.route('/')
def index():
	try:
		pass
	except:
		pass
	return(render_template('index.html'))
	
@app.route('/training', methods=['GET', 'POST'])
def training():
	if request.method == 'GET':
		pass
	elif request.method == 'POST':
		uploaded_file = request.files['file']
		filename = secure_filename(uploaded_file.filename)
		uploaded_file.save(os.path.join("static", filename))
		#result = kenali_suara("./static/"+filename)
		#os.remove("./static/"+filename)
		#global nama_file
		#nama_file = "static/"+filename
		inputan = file_input("static/"+filename)
			
		perbaikan = cleaning_train()
		inputan.df_ulasan["ulasan_clean"] = perbaikan.clean_ulasan_train(inputan.df_ulasan["Ulasan"])
		
		norm = normal()
		inputan.df_ulasan["ulasan_clean"] = inputan.df_ulasan["ulasan_clean"].apply(norm.normalisasi)
		
		bersih = filtering()
		# Stopword removal dan ditambah "stopword tambahan"
		inputan.df_ulasan["ulasan_clean"] = inputan.df_ulasan["ulasan_clean"].apply(bersih.clean_stopword)
		inputan.df_ulasan["ulasan_clean"] = inputan.df_ulasan["ulasan_clean"].apply(bersih.clean_stopword_tambahan)
		
		stemmer = Stemming()
		inputan.df_ulasan["ulasan_clean"] = inputan.df_ulasan["ulasan_clean"].apply(stemmer.clean_stem)
		inputan.df_ulasan.to_excel("static/data_stem.xlsx", index = False)
		
		TFIDF = bobotTFIDF_train()
		x = TFIDF.tf_idf_train(inputan.df_ulasan['ulasan_clean'])
		y = inputan.df_ulasan['Label']
		
		# Modelling SVM
		global x_train, x_test, y_train, y_test
		x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.3, stratify=y, random_state=42)
		#hasil = inputan.df_ulasan["ulasan_clean"].head()
		klasifikasi = modelSVM()
		hasil1 = klasifikasi.klasifikasi_linear()
		hasil2 = klasifikasi.klasifikasi_rbf()
		hasil3 = klasifikasi.klasifikasi_polinomial()
			
		return render_template('training.html', name =filename, result1=hasil1, result2=hasil2, result3=hasil3)
	return(render_template('training.html'))

@app.route("/", methods=['GET', 'POST'])
def main():
		if request.method == 'GET':
			return
		elif request.method == 'POST':
			#uploaded_file = request.files['file']
			#filename = secure_filename(uploaded_file.filename)
			#uploaded_file.save(os.path.join("static", filename))
			#result = kenali_suara("./static/"+filename)
			#os.remove("./static/"+filename)
			#global nama_file
			#nama_file = "static/"+filename
			#, result=result, name=filename, file="static/"+filename
			
			inputan = request.form['input_baru']
			
			perbaikan = cleaning()
			hasil_perbaikan = perbaikan.clean_ulasan(inputan)
			
			norm = normal()
			hasil_normalisasi = norm.normalisasi(hasil_perbaikan)
			
			bersih = filtering()
			# Stopword removal dan ditambah "stopword tambahan"
			hasil_filtering = bersih.clean_stopword(hasil_normalisasi)
			hasil_filtering = bersih.clean_stopword_tambahan(hasil_filtering)
			
			stemmer = Stemming()
			hasil_stem = stemmer.clean_stem(hasil_filtering)
			
			
			TFIDF = bobotTFIDF()
			x = TFIDF.tf_idf(hasil_stem)
			
			# Modelling SVM
			#global x_train, x_test, y_train, y_test
			#x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.3, stratify=y, random_state=30)
			#hasil = inputan.df_ulasan["ulasan_clean"].head()
			#klasifikasi = modelSVM()
			#hasil1 = klasifikasi.klasifikasi_linear()
			#hasil2 = klasifikasi.klasifikasi_rbf()
			#hasil3 = klasifikasi.klasifikasi_polinomial()
			
			model_file = open('./model/modelsvmlinear.pkl', 'rb')
			svmlinear = pickle.load(model_file)
			model_file = open('./model/modelsvmrbf.pkl', 'rb')
			svmrbf = pickle.load(model_file)
			model_file = open('./model/modelsvmpol.pkl', 'rb')
			svmpol = pickle.load(model_file)
      
			"""
			hasil1=x[x.shape[0]-1].shape
			xy = x[0:x.shape[0]-1,0:570]
			hasil2=xy.shape
			hasil3=hasil2
			"""
			hasil = svmlinear.predict(x[x.shape[0]-1,0:700]) == 1
			if hasil == False:
				hasil1 = "Negatif"
			else:
				hasil1 = "Positif"
			hasil = svmrbf.predict(x[x.shape[0]-1,0:700]) == 1
			if hasil == False:
				hasil2 = "Negatif"
			else:
				hasil2 = "Positif"
			hasil = svmpol.predict(x[x.shape[0]-1,0:700]) == 1
			if hasil == False:
				hasil3 = "Negatif"
			else:
				hasil3 = "Positif"
      
			return render_template('index.html', ulasan=inputan, result1=hasil1, result2=hasil2, result3=hasil3)
		else:
			return "Unsupported Request Method"

if __name__ == '__main__':
    app.run(port=5000, debug=True)
	
