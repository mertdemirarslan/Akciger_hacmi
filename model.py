
import numpy as np 
import pandas as pd 
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, RandomizedSearchCV, RepeatedStratifiedKFold
from sklearn.pipeline import make_pipeline
import pickle
from sklearn.linear_model import LinearRegression
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error
import statsmodels.api as sm
from sklearn.model_selection import train_test_split



df=pd.read_spss("akcigerhacmi.sav")

X=df.drop(["sagAK","solAK"],axis=1)
y = df[["sagAK"]]

lm  = LinearRegression()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30, random_state = 1)

model = lm.fit(X_train, y_train)
pipe=make_pipeline(StandardScaler(),lm)
pipe.fit(X_train, y_train)
pickle.dump(pipe, open("model.pkl","wb"))

model = pickle.load(open("model.pkl","rb"))






