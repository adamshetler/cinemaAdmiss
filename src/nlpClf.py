import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import train_test_split

# data = pd.read_csv("data/out/cinemaFeatures.csv")
data = pd.read_csv("data/out/cinemaFeaturesTextScrape.csv")
print(data.head)
print(data.shape)
data = data.drop(data[data.release_date.str.len()!=10].index)
print(data.shape)
data = data.reset_index(drop=True)
data = data.drop(data[data.NumberShows < 10].index)
data = data.reset_index(drop=True)
data["LogGross$"] = np.log(data["Gross$"])
data["release_date"] = pd.to_datetime(data["release_date"])
data = data.sort_values(by="release_date", ascending=True)
data = data.reset_index()
data.dropna(subset=['scrapedInfo'], inplace=True)
data = data.reset_index()
#data.dropna(subset=['overview'], inplace=True)
#data = data.reset_index()
print(data.shape)
data["LogAdmissions"] = np.log(data.Admissions)
data["LogAdmissions"].hist(bins=50)
plt.title("LogAdmissions Historgram")
plt.show()

# IQR Log Admissions
Q1 = np.percentile(data['LogAdmissions'], 25,
                   method="normal_unbiased")
Q3 = np.percentile(data['LogAdmissions'], 75,
                   method="normal_unbiased")
IQR = Q3 - Q1
print("IQR:", IQR)
print("Old Shape: ", data.shape)
# Upper bound
upper = Q3 + 1.5 * IQR
# Lower bound
lower = Q1 - 1.5 * IQR

#create Admissions labels for classifiction model
# conditions = [(data["LogAdmissions"] < upper),
#               (data["LogAdmissions"] >= lower) & (data["LogAdmissions"] <= upper),
#               (data["LogAdmissions"] >= upper)]

conditions = [(data["LogAdmissions"] < 7),
              (data["LogAdmissions"] >= 7)]
label = ["Low", "High"]

data["label"] = np.select(conditions, label)
nlp_data = data[["scrapedInfo", "label"]]
#nlp_data = data[["overview", "label"]]

#data split non-time dependent
# X_train, X_test, y_train, y_test = train_test_split(data["overview"], data["label"], test_size=0.33, random_state=42)

# time dependent data split (80-20)
X_train = nlp_data.scrapedInfo[0:378]
y_train = nlp_data.label[0:378]
X_test = nlp_data.scrapedInfo[378:len(data)]
y_test = nlp_data.label[378:len(data)]

# X_train = nlp_data.overview[0:378]
# y_train = nlp_data.label[0:378]
# X_test = nlp_data.overview[378:len(data)]
# y_test = nlp_data.label[378:len(data)]

count_vec = CountVectorizer(stop_words = "english", lowercase=True, ngram_range=(1,1))
X_train_vectors = count_vec.fit_transform(X_train).toarray()
print(count_vec.get_feature_names_out())

#train classifier
clf_svm = svm.SVC(kernel="linear")
clf_svm.fit(X_train_vectors, y_train)

#predictions
X_test_vectors = count_vec.transform(X_test).toarray()
y_pred = clf_svm.predict(X_test_vectors)

#model evaluation
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))

# tf–idf: “Term Frequency times Inverse Document Frequency”.
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_vectors)

#train classifier

clf_svm = SGDClassifier(loss='hinge', penalty='l2',
                           alpha=1e-3, random_state=42,
                           max_iter=5, tol=None)
clf_svm.fit(X_train_tfidf, y_train)

#predictions
X_test_vectors = count_vec.transform(X_test).toarray()
X_test_tfidf = tfidf_transformer.transform(X_test_vectors)
y_pred = clf_svm.predict(X_test_tfidf)

#model evaluation
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))

#model with pipeline
text_clf = Pipeline([
     ('vect', CountVectorizer()),
     ('tfidf', TfidfTransformer()),
     ('clf', SGDClassifier(loss='hinge', penalty='l2',
                           alpha=1e-3, random_state=42,
                           max_iter=5, tol=None))])

parameters = {
    'vect__ngram_range': [(1, 1), (1, 2), (1,3), (1,4), (1,5)],
     'tfidf__use_idf': (True, False),
     'clf__alpha': (1e-2, 1e-3),
}
gs_clf = GridSearchCV(text_clf, parameters, cv=5, n_jobs=-1)
gs_clf = gs_clf.fit(X_train, y_train)
text_clf.fit(X_train, y_train)
predicted = text_clf.predict(y_test)
print("mean accuracy: ", np.mean(predicted == y_test))
print("best score: ", gs_clf.best_score_)
for param_name in sorted(parameters.keys()):
    print("%s: %r" % (param_name, gs_clf.best_params_[param_name]))

