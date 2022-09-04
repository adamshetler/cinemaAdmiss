import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing, svm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


data = pd.read_csv("data/out/cinemaFeatures.csv")
print(data.head)
data["LogGross$"] = np.log(data["Gross$"])
data2 = data[data["popularity"] >10]
#plotting the Scatter plot to check relationship
plt.figure()
sns.lmplot(x ="popularity", y ="LogGross$", data = data2, order = 2, ci = 95)
plt.show()