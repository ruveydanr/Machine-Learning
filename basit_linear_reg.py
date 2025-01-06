# CREATING MODEL
import pandas as pd
df = pd.read_csv(r"C:\Users\ruvey\Desktop\pyth\Advertising.csv")
df = df.iloc[:,1:len(df)]
df.head()
print(df.head())
df.info()

import seaborn as sns

import matplotlib.pyplot as plt
sns.jointplot(x="TV", y="sales", data=df, kind="reg")
plt.show() ;

import sklearn
from sklearn.linear_model import LinearRegression
x = df[["TV"]]
x.head()
print(x.head())
y= df[["sales"]]
reg = LinearRegression()
model =reg.fit(x,y)
print(dir(model))

print(model.intercept_)
print(model.coef_)

# r kare
print(model.score(x,y))


# TAHMIN FORECAST
g = sns.regplot(x=df["TV"],y=df["sales"],ci=None,scatter_kws={'color':'r','s':9})
g.set_title("Model Denklemi: Sales = 7.03 + TV * 0.05")
g.set_ylabel("Satış Sayısı")
g.set_xlabel("TV Harcamaları")
plt.xlim(-10,310)
plt.ylim(bottom = 0)
plt.show();

# Verilen denklem
TV = 165
Sales = 7.03 + TV * 0.05
print(Sales);

print(model.predict([[165]]))
yeni_veri =[[5],[15],[30]]
print(model.predict(yeni_veri))

# mse rmse optimizasyon 
print(y.head())

print(model.predict(x)[0:7])

gercek_y= y[0:10]
tahmin_edilen_y = pd.DataFrame(model.predict(x)[0:10])
hatalar=pd.concat([gercek_y,tahmin_edilen_y],axis=1)
hatalar.columns =["gercek_y","tahmin_edilen_y"]
hatalar["hata"]=hatalar["gercek_y"] - hatalar["tahmin_edilen_y"]
hatalar["hata_kareler"] = hatalar["hata"]**2
print(hatalar);
# ortalama hatamızı bulalım
import numpy as np
print(np.mean(hatalar["hata_kareler"]))






