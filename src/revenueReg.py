import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn import linear_model
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.kernel_ridge import KernelRidge
from sklearn import gaussian_process
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel
import time
from sklearn import ensemble
from sklearn.inspection import permutation_importance

data = pd.read_csv("data/out/cinemaFeatures.csv")
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
data = data.reset_index(drop=True)
data.dropna(subset=['revenue'], inplace=True)
data = data.reset_index(drop=True)
data.dropna(subset=['LogGross$'], inplace=True)
print(data.shape)
data = data.drop(data[data["revenue"] == 0 ].index)
data = data.reset_index(drop=True)
###### Regression Models
#IQR Revenue

Q1 = np.percentile(data['revenue'], 25,
                   method="normal_unbiased")

Q3 = np.percentile(data['revenue'], 75,
                   method="normal_unbiased")
IQR = Q3 - Q1
print("IQR:", IQR)
print("Old Shape: ", data.shape)
# Upper bound
upper = np.where(data['revenue'] >= (Q3 + 1.5 * IQR))
# Lower bound
lower = np.where(data['revenue'] <= (Q1 - 1.5 * IQR))
data.drop(upper[0], inplace=True)
data = data.reset_index(drop=True)
data.drop(lower[0], inplace=True)
data = data.reset_index(drop=True)
print("New Shape: ", data.shape)
# color = {
#     "boxes": "DarkGreen",
#     "whiskers": "DarkOrange",
#     "medians": "DarkBlue",
#     "caps": "Gray",
# }
# data["revenue"].plot.box(color=color, sym="r+")
# plt.show()
# print("blanks: ", len(data[data["popularity"] == min(data["popularity"])]))
# print(data["popularity"].min())
# data["popularity"] = np.where(data["popularity"] < 1,
#                               data["popularity"] * 100,
#                               data["popularity"])
#
# data["popularity"] = np.where(data["popularity"] < 10,
#                               data["popularity"] * 10,
#                               data["popularity"])

# transform data

# one hot encode original_language feature for test set
# ohe = OneHotEncoder()
# ohe_array = ohe.fit_transform(data[["original_language"]]).toarray()
# feature_labels = np.array(ohe.categories_).ravel()
# original_language_ohe = pd.DataFrame(ohe_array, columns=feature_labels)
# data = data.drop(["original_language"], axis=1)
# data = pd.concat([data, original_language_ohe], axis=1)
# print(data.columns)
numerical_data = data[["revenue"]]
scaler = sklearn.preprocessing.StandardScaler().fit(numerical_data)
data_scaled = scaler.transform(numerical_data)
data_scaled = pd.DataFrame(data_scaled, columns=["revenue"])
data1 = pd.concat([data_scaled, data[["LogGross$"]]], axis=1)
# data = pd.concat([data_scaled, original_language_ohe, data[["LogGross$"]]], axis=1)
print(data.columns)
#before training linear regression model - split data by release date (80-20 split)
# therefore train model on first 386 rows, use remainder for validation

dataTrain = data1[0:113]
dataTest = data1[113:len(data)]

print(dataTest.shape)
print(dataTrain.shape)
# "LogGross$", "Gross$", "Net$", "Admissions", "AvgPerShow$", "NumberShows"
y_train = dataTrain["LogGross$"]
y_test = dataTest["LogGross$"]

# X_train = dataTrain[["vote_average", "popularity", "revenue", "runtime", "budget"]]
# X_test = dataTest[["vote_average", "popularity", "revenue", "runtime", "budget"]]
X_train = dataTrain[["revenue"]]
X_test = dataTest[["revenue"]]

# X_train = dataTrain[["vote_average", "popularity", 'ar', 'bn', 'cn', 'cs', 'da', 'de', 'el', 'en', 'es', 'fr', 'he', 'hi',
#        'hu', 'it', 'ja', 'ko', 'ms', 'nl', 'no', 'pl', 'pt', 'ru', 'sh', 'st',
#        'sv', 'tl', 'tr', 'uk', 'ur', 'xx', 'zh', "revenue", "runtime", "budget"]]
# X_test = dataTest[["vote_average", "popularity", 'ar', 'bn', 'cn', 'cs', 'da', 'de', 'el', 'en', 'es', 'fr', 'he', 'hi',
#        'hu', 'it', 'ja', 'ko', 'ms', 'nl', 'no', 'pl', 'pt', 'ru', 'sh', 'st',
#        'sv', 'tl', 'tr', 'uk', 'ur', 'xx', 'zh', "revenue", "runtime", "budget"]]


#### Model 1: Simple Linear Regression (Revenue)
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

# Plot outputs
plt.scatter(X_test, y_test, color="black")
plt.plot(X_test, y_pred, color="blue", linewidth=3)
plt.title("Simple Linear Regression Model")
plt.xticks(())
plt.yticks(())
plt.show()


#print(X_train.columns, X_test.columns, y_train.columns)
kr = GridSearchCV(
    KernelRidge(kernel="rbf", gamma=0.1),
    param_grid={"alpha": [1e0, 0.1, 1e-2, 1e-3], "gamma": np.logspace(-2, 2, 5)},
)
X_plot = np.linspace(-1, 5, 100000)[:, None]
t0 = time.time()
kr.fit(X_train, y_train)
kr_fit = time.time() - t0
print(f"Best KRR with params: {kr.best_params_} and R2 score: {kr.best_score_:.3f}")
print("KRR complexity and bandwidth selected and model fitted in %.3f s" % kr_fit)

t0 = time.time()
y_kr = kr.predict(X_plot)
kr_predict = time.time() - t0
print("KRR prediction for %d inputs in %.3f s" % (X_plot.shape[0], kr_predict))

plt.plot(
    X_plot, y_kr, c="g", label="KRR (fit: %.3fs, predict: %.3fs)" % (kr_fit, kr_predict)
)
plt.xlabel("data")
plt.ylabel("target")
plt.title("Kernel Ridge Regression")
_ = plt.legend()
plt.scatter(X_test, y_test, color="black")
plt.show()

#### Gaussian Process Model
kernel = ConstantKernel() + Matern(length_scale=2, nu=3/2) + WhiteKernel(noise_level=1)
gp = gaussian_process.GaussianProcessRegressor(kernel=kernel)
gp.fit(X_train, y_train)

x_pred = X_plot.reshape(-1,1)
y_pred, sigma = gp.predict(x_pred, return_std=True)
plt.plot(
    X_plot, y_pred, c="g", label="GPR"
)
plt.xlabel("data")
plt.ylabel("target")
plt.title("Gaussian Process Regression")
_ = plt.legend()
plt.scatter(X_test, y_test, color="black")
plt.show()


plt.plot(
    X_plot, y_pred, c="g", label="GPR"
)
plt.xlabel("data")
plt.ylabel("target")
plt.title("Gaussian Process Regression")
_ = plt.legend()
plt.scatter(X_train, y_train, color="black")
plt.show()


# Gradient Boosted Regression - univariate
params = {
    "n_estimators": 500,
    "max_depth": 4,
    "min_samples_split": 5,
    "learning_rate": 0.01,
    "loss": "squared_error",
}

reg = ensemble.GradientBoostingRegressor(**params)
reg.fit(X_train, y_train)
mse = mean_squared_error(y_test, reg.predict(X_test))
print("(GBM: The mean squared error (MSE) on test set: {:.4f}".format(mse))
y_pred = reg.predict(X_plot)
plt.plot(
    X_plot, y_pred, c="g", label="GBM"
)
plt.xlabel("data")
plt.ylabel("target")
plt.title("GBM")
_ = plt.legend()
plt.scatter(X_test, y_test)
plt.show()

test_score = np.zeros((params["n_estimators"],), dtype=np.float64)
for i, y_pred in enumerate(reg.staged_predict(X_test)):
    test_score[i] = reg.loss_(y_test, y_pred)

fig = plt.figure(figsize=(6, 6))
plt.subplot(1, 1, 1)
plt.title("Deviance")
plt.plot(
    np.arange(params["n_estimators"]) + 1,
    reg.train_score_,
    "b-",
    label="Training Set Deviance",
)
plt.plot(
    np.arange(params["n_estimators"]) + 1, test_score, "r-", label="Test Set Deviance"
)
plt.legend(loc="upper right")
plt.xlabel("Boosting Iterations")
plt.ylabel("Deviance")
fig.tight_layout()
plt.show()

# Gradient Boosted Regression - multivariate
print(data.columns)
numerical_data = data[["vote_average", "popularity", "revenue", "runtime", "budget"]]
scaler = sklearn.preprocessing.StandardScaler().fit(numerical_data)
data_scaled = scaler.transform(numerical_data)
data_scaled = pd.DataFrame(data_scaled, columns=["vote_average", "popularity", "revenue", "runtime", "budget"])
data = pd.concat([data_scaled, data[["LogGross$"]]], axis=1)

dataTrain = data[0:113]
dataTest = data[113:len(data)]

print(dataTest.shape)
print(dataTrain.shape)
# "LogGross$", "Gross$", "Net$", "Admissions", "AvgPerShow$", "NumberShows"
y_train = dataTrain["LogGross$"]
y_test = dataTest["LogGross$"]

X_train = dataTrain[["vote_average", "popularity", "revenue", "runtime", "budget"]]
X_test = dataTest[["vote_average", "popularity", "revenue", "runtime", "budget"]]


print("Train", X_train.shape, y_train.shape)
print("Test", X_test.shape, y_test.shape)

# X_train = dataTrain[["revenue"]]
# X_test = dataTest[["revenue"]]

params = {
    "n_estimators": 500,
    "max_depth": 4,
    "min_samples_split": 5,
    "learning_rate": 0.01,
    "loss": "squared_error",
}

reg = ensemble.GradientBoostingRegressor(**params)
reg.fit(X_train, y_train)

mse = mean_squared_error(y_test, reg.predict(X_test))
print("(GBM: The mean squared error (MSE) on test set: {:.4f}".format(mse))


feature_importance = reg.feature_importances_
sorted_idx = np.argsort(feature_importance)
pos = np.arange(sorted_idx.shape[0]) + 0.5
fig = plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.barh(pos, feature_importance[sorted_idx], align="center")
plt.yticks(pos, np.array(dataTrain.columns)[sorted_idx])
plt.title("Feature Importance (MDI)")

result = permutation_importance(
    reg, X_test, y_test, n_repeats=10, random_state=42, n_jobs=2
)
sorted_idx = result.importances_mean.argsort()
plt.subplot(1, 2, 2)
plt.boxplot(
    result.importances[sorted_idx].T,
    vert=False,
    labels=np.array(dataTrain.columns)[sorted_idx],
)
plt.title("Permutation Importance (test set)")
fig.tight_layout()
plt.show()