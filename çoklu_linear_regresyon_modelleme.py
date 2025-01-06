# Veriyi içe aktar
import pandas as pd
df = pd.read_csv(r"C:\Users\ruvey\Desktop\pyth\Advertising.csv")
df = df.iloc[:, 1:len(df)]
print(df.head())

# Bağımsız ve bağımlı değişkenleri ayır
X = df.drop('sales', axis=1)
y = df["sales"]  # y bir pandas Series olarak tanımlandı

# statsmodels ile model oluşturma
import statsmodels.api as sm
lm = sm.OLS(y, X)
model = lm.fit()
print(model.summary())

# scikit-learn ile model oluşturma
from sklearn.linear_model import LinearRegression
lm = LinearRegression()
model = lm.fit(X, y)

# Model çıktıları
print("Modelin sabiti (intercept):", model.intercept_)
print("Modelin katsayıları (coefficients):", model.coef_)

# TAHMİN FORECAST
sales = 2.94 + 30 * 0.04 + 10 * 0.19 - 40 * 0.001
print(sales)

yeni_veri = [[30],[10],[40]]
import pandas as pd
yeni_veri = pd.DataFrame(yeni_veri).T
print(model.predict(yeni_veri))

# MODEL DOĞRULUK DEĞERLENDİRME
from sklearn.metrics import mean_squared_error
print(y.head())
model.predict(X)[0:10]

MSE = mean_squared_error(y,model.predict(X))
print("MSE =",MSE)

import numpy as np
RMSE = np.sqrt(MSE)
print("RMSE =",RMSE)

# MODEL TUNING MODEL DOĞRULAMA
print(X.head())
print(y.head())

# sınama seti
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.20,random_state=99)
print(X_train.head())
print(y_train.head())

lm = LinearRegression()
model = lm.fit(X_train,y_train)
print(model)
#EĞİTİM HATASI
print("Train MSE =",np.sqrt(mean_squared_error(y_train,model.predict(X_train))))

#k-katlı cross-validation 
from sklearn.model_selection import cross_val_score
#cv mse
cross_val_score(model,X_train,y_train,cv=10,scoring="neg_mean_squared_error")
print("CV_MSE =",np.mean(-cross_val_score(model,X_train,y_train,cv=10,scoring="neg_mean_squared_error")))
#cv rmse
print("CV_RMSE =",np.sqrt(np.mean(-cross_val_score(model,X_train,y_train,cv=10,scoring="neg_mean_squared_error"))))