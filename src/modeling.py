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
print(data.shape)
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
#release_date = datetime.strptime(data["release_date"],'%d/%m/%y')
data["release_date"] = pd.to_datetime(data["release_date"])
#print(release_date)
data = data.sort_values(by="release_date", ascending=True)
data = data.reset_index()
# transform data
from sklearn.preprocessing import OneHotEncoder
#one hot encode original_language feature for test set
# ohe = OneHotEncoder()
# ohe_array = ohe.fit_transform(data[["original_language"]]).toarray()
# feature_labels = np.array(ohe.categories_).ravel()
# original_language_ohe = pd.DataFrame(ohe_array, columns=feature_labels)
# data = data.drop(["original_language"], axis=1)
# data = pd.concat([data, original_language_ohe], axis=1)
print(data.columns)
numerical_data = data[["vote_average", "popularity","revenue", "runtime", "budget"]]
scaler = preprocessing.StandardScaler().fit(numerical_data)
data_scaled = scaler.transform(numerical_data)
data_scaled = pd.DataFrame(data_scaled, columns=["vote_average", "popularity","revenue", "runtime", "budget"])
data = pd.concat([data_scaled, data[["LogGross$"]]], axis=1)
#data = pd.concat([data_scaled, original_language_ohe, data[["LogGross$"]]], axis=1)
print(data.columns)
#before training linear regression model - split data by release date (80-20 split)
# therefore train model on first 386 rows, use remainder for validation
dataTrain = data[0:387]
dataTest = data[387:len(data)]

# "LogGross$", "Gross$", "Net$", "Admissions", "AvgPerShow$", "NumberShows"
y_train = dataTrain["LogGross$"]
y_test = dataTest["LogGross$"]


X_train = dataTrain[["vote_average", "popularity", "revenue", "runtime", "budget"]]
X_test = dataTest[["vote_average", "popularity", "revenue", "runtime", "budget"]]

# X_train = dataTrain[["vote_average", "popularity", 'ar', 'bn', 'cn', 'cs', 'da', 'de', 'el', 'en', 'es', 'fr', 'he', 'hi',
#        'hu', 'it', 'ja', 'ko', 'ms', 'nl', 'no', 'pl', 'pt', 'ru', 'sh', 'st',
#        'sv', 'tl', 'tr', 'uk', 'ur', 'xx', 'zh', "revenue", "runtime", "budget"]]
# X_test = dataTest[["vote_average", "popularity", 'ar', 'bn', 'cn', 'cs', 'da', 'de', 'el', 'en', 'es', 'fr', 'he', 'hi',
#        'hu', 'it', 'ja', 'ko', 'ms', 'nl', 'no', 'pl', 'pt', 'ru', 'sh', 'st',
#        'sv', 'tl', 'tr', 'uk', 'ur', 'xx', 'zh', "revenue", "runtime", "budget"]]


print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)

linRegModel = linear_model.LinearRegression()
linRegModel.fit(X_train, y_train)
# Make predictions using the testing set
y_pred = linRegModel.predict(X_test)
print(linRegModel.coef_)
# The coefficients
print("Coefficients: \n", linRegModel.coef_)
# The mean squared error
print("Mean squared error: %.2f" % mean_squared_error(y_test, y_pred))
# The coefficient of determination: 1 is perfect prediction
print("Coefficient of determination: %.2f" % r2_score(y_test, y_pred))






