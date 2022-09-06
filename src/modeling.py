import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
from sklearn import preprocessing, svm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score

data = pd.read_csv("data/out/cinemaFeatures.csv")
print(data.head)
print(data.shape)
data = data.drop(data[data.release_date.str.len()!=10].index)
print(data.shape)
data = data.reset_index(drop=True)
data = data.drop(data[data.NumberShows < 10].index)
data = data.reset_index(drop=True)
# data = data.drop(data[data.popularity == 0.6].index)
# data = data.reset_index(drop=True)
# print(data.shape)
data["LogGross$"] = np.log(data["Gross$"])
#plotting the Scatter plot to check relationship
# plt.figure()
# sns.lmplot(x ="popularity", y ="LogGross$", data = data, order = 1, ci = 95)
# sns.lmplot(x ="vote_average", y ="LogGross$", data = data, order = 1, ci = 95)
# sns.lmplot(x ="runtime", y ="LogGross$", data = data, order = 1, ci = 95)
# sns.lmplot(x ="budget", y ="LogGross$", data = data, order = 1, ci = 95)
# sns.lmplot(x ="revenue", y ="LogGross$", data = data, order = 1, ci = 95)
# plt.show()
from datetime import datetime
data["release_date"] = pd.to_datetime(data["release_date"])
#print(release_date)
data = data.sort_values(by="release_date", ascending=True)
data = data.reset_index()

data.dropna(subset=['overview'], inplace=True)
data = data.reset_index()

print(data.shape)

data["LogAdmissions"] = np.log(data.Admissions)
data["LogAdmissions"].hist(bins=50)
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
nlp_data = data[["overview", "label"]]

X_train = nlp_data.overview[0:378]
y_train = nlp_data.label[0:378]

X_test = nlp_data.overview[378:len(data)]
y_test = nlp_data.label[378:len(data)]

from sklearn.feature_extraction.text import CountVectorizer
count_vec = CountVectorizer(stop_words = "english", lowercase=True, ngram_range=(1,3))
X_train_vectors = count_vec.fit_transform(X_train).toarray()
print(count_vec.get_feature_names_out())

from sklearn import svm
#train classifier
clf_svm = svm.SVC(kernel="linear")
clf_svm.fit(X_train_vectors, y_train)

#predictions
X_test_vectors = count_vec.transform(X_test).toarray()
y_pred = clf_svm.predict(X_test_vectors)

#model evaluation
from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))

# tf–idf: “Term Frequency times Inverse Document Frequency”.
from sklearn.feature_extraction.text import TfidfTransformer
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
from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))

#model with pipeline
from sklearn.pipeline import Pipeline

text_clf = Pipeline([
     ('vect', CountVectorizer()),
     ('tfidf', TfidfTransformer()),
     ('clf', SGDClassifier(loss='hinge', penalty='l2',
                           alpha=1e-3, random_state=42,
                           max_iter=5, tol=None))])
from sklearn.model_selection import GridSearchCV
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


####### Regression Models
#IQR Revenue
#
# Q1 = np.percentile(data['revenue'], 25,
#                    method="normal_unbiased")
#
# Q3 = np.percentile(data['revenue'], 75,
#                    method="normal_unbiased")
# IQR = Q3 - Q1
# print("IQR:", IQR)
# print("Old Shape: ", data.shape)
# # Upper bound
# # upper = np.where(data['revenue'] >= (Q3 + 1.5 * IQR))
# # # Lower bound
# # lower = np.where(data['revenue'] <= (Q1 - 1.5 * IQR))
# #
# # data.drop(upper[0], inplace=True)
# # # data = data.reset_index()
# # data.drop(lower[0], inplace=True)
# # data = data.reset_index()
# # print("New Shape: ", data.shape)
# # color = {
# #     "boxes": "DarkGreen",
# #     "whiskers": "DarkOrange",
# #     "medians": "DarkBlue",
# #     "caps": "Gray",
# # }
# # data["revenue"].plot.box(color=color, sym="r+")
# # plt.show()
# # print("blanks: ", len(data[data["popularity"] == min(data["popularity"])]))
# # print(data["popularity"].min())
# # data["popularity"] = np.where(data["popularity"] < 1,
# #                               data["popularity"] * 100,
# #                               data["popularity"])
# #
# # data["popularity"] = np.where(data["popularity"] < 10,
# #                               data["popularity"] * 10,
# #                               data["popularity"])
#
# # transform data
# from sklearn.preprocessing import OneHotEncoder
# # one hot encode original_language feature for test set
# # ohe = OneHotEncoder()
# # ohe_array = ohe.fit_transform(data[["original_language"]]).toarray()
# # feature_labels = np.array(ohe.categories_).ravel()
# # original_language_ohe = pd.DataFrame(ohe_array, columns=feature_labels)
# # data = data.drop(["original_language"], axis=1)
# # data = pd.concat([data, original_language_ohe], axis=1)
# # print(data.columns)
# numerical_data = data[["revenue"]]
# scaler = preprocessing.StandardScaler().fit(numerical_data)
# data_scaled = scaler.transform(numerical_data)
# data_scaled = pd.DataFrame(data_scaled, columns=["revenue"])
# data = pd.concat([data_scaled, data[["LogGross$"]]], axis=1)
# # data = pd.concat([data_scaled, original_language_ohe, data[["LogGross$"]]], axis=1)
# print(data.columns)
# #before training linear regression model - split data by release date (80-20 split)
# # therefore train model on first 386 rows, use remainder for validation
#
# dataTrain = data[0:111]
# dataTest = data[111:len(data)]
#
# print(dataTest.shape)
# print(dataTrain.shape)
# # "LogGross$", "Gross$", "Net$", "Admissions", "AvgPerShow$", "NumberShows"
# y_train = dataTrain["LogGross$"]
# y_test = dataTest["LogGross$"]
#
# # X_train = dataTrain[["vote_average", "popularity", "revenue", "runtime", "budget"]]
# # X_test = dataTest[["vote_average", "popularity", "revenue", "runtime", "budget"]]
# X_train = dataTrain[["revenue"]]
# X_test = dataTest[["revenue"]]
#
# # X_train = dataTrain[["vote_average", "popularity", 'ar', 'bn', 'cn', 'cs', 'da', 'de', 'el', 'en', 'es', 'fr', 'he', 'hi',
# #        'hu', 'it', 'ja', 'ko', 'ms', 'nl', 'no', 'pl', 'pt', 'ru', 'sh', 'st',
# #        'sv', 'tl', 'tr', 'uk', 'ur', 'xx', 'zh', "revenue", "runtime", "budget"]]
# # X_test = dataTest[["vote_average", "popularity", 'ar', 'bn', 'cn', 'cs', 'da', 'de', 'el', 'en', 'es', 'fr', 'he', 'hi',
# #        'hu', 'it', 'ja', 'ko', 'ms', 'nl', 'no', 'pl', 'pt', 'ru', 'sh', 'st',
# #        'sv', 'tl', 'tr', 'uk', 'ur', 'xx', 'zh', "revenue", "runtime", "budget"]]
#
#
# #### Model 1: Simple Linear Regression (Revenue)
# linRegModel = linear_model.LinearRegression()
# linRegModel.fit(X_train, y_train)
# # Make predictions using the testing set
# y_pred = linRegModel.predict(X_test)
# print(linRegModel.coef_)
# # The coefficients
# print("Coefficients: \n", linRegModel.coef_)
# # The mean squared error
# print("Mean squared error: %.2f" % mean_squared_error(y_test, y_pred))
# # The coefficient of determination: 1 is perfect prediction
# print("Coefficient of determination: %.2f" % r2_score(y_test, y_pred))
#
# # Plot outputs
# plt.scatter(X_test, y_test, color="black")
# plt.plot(X_test, y_pred, color="blue", linewidth=3)
# plt.title("Simple Linear Regression Model")
# plt.xticks(())
# plt.yticks(())
# plt.show()
#
#
# from sklearn.model_selection import GridSearchCV
# from sklearn.svm import SVR
# from sklearn.kernel_ridge import KernelRidge
# import time
# #print(X_train.columns, X_test.columns, y_train.columns)
# kr = GridSearchCV(
#     KernelRidge(kernel="rbf", gamma=0.1),
#     param_grid={"alpha": [1e0, 0.1, 1e-2, 1e-3], "gamma": np.logspace(-2, 2, 5)},
# )
# X_plot = np.linspace(-1, 5, 100000)[:, None]
# t0 = time.time()
# kr.fit(X_train, y_train)
# kr_fit = time.time() - t0
# print(f"Best KRR with params: {kr.best_params_} and R2 score: {kr.best_score_:.3f}")
# print("KRR complexity and bandwidth selected and model fitted in %.3f s" % kr_fit)
#
# t0 = time.time()
# y_kr = kr.predict(X_plot)
# kr_predict = time.time() - t0
# print("KRR prediction for %d inputs in %.3f s" % (X_plot.shape[0], kr_predict))
#
# plt.plot(
#     X_plot, y_kr, c="g", label="KRR (fit: %.3fs, predict: %.3fs)" % (kr_fit, kr_predict)
# )
# plt.xlabel("data")
# plt.ylabel("target")
# plt.title("Kernel Ridge Regression")
# _ = plt.legend()
# plt.scatter(X_test, y_test, color="black")
# plt.show()
#
# #### Gaussian Process Model
# from sklearn import gaussian_process
# from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel
#
# kernel = ConstantKernel() + Matern(length_scale=2, nu=3/2) + WhiteKernel(noise_level=1)
# gp = gaussian_process.GaussianProcessRegressor(kernel=kernel)
# gp.fit(X_train, y_train)
#
# x_pred = X_plot.reshape(-1,1)
# y_pred, sigma = gp.predict(x_pred, return_std=True)
# plt.plot(
#     X_plot, y_pred, c="g", label="GPR"
# )
# plt.xlabel("data")
# plt.ylabel("target")
# plt.title("Gaussian Process Regression")
# _ = plt.legend()
# plt.scatter(X_test, y_test, color="black")
# plt.show()
#
#
# plt.plot(
#     X_plot, y_pred, c="g", label="GPR"
# )
# plt.xlabel("data")
# plt.ylabel("target")
# plt.title("Gaussian Process Regression")
# _ = plt.legend()
# plt.scatter(X_train, y_train, color="black")
# plt.show()
#
#
#
