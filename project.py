import matplotlib.pyplot as plt 
import pandas as pd 
import seaborn as sns 
df = pd.read_csv("E:\csvdhf5xlsxurlallfiles/house.csv")
print(df.head())
print(df.info())
# selecting features from df 
features = df[['bathrooms', 'bedrooms', 'listPrice', 'livingArea', 'lotSize', 'numParkingSpaces']]
print(features.info())
X=features.fillna(0)
print(X.info())
#selecting target
y=pd.get_dummies(df[['grade']])
print(y.info())
#using classifications
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=6)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state= 2, test_size=0.3)
knn.fit(X_train, y_train)
print(knn.predict(X_test))
print(knn.score(X_test, y_test))
from sklearn.linear_model import LinearRegression
linreg = LinearRegression()
linreg.fit(X_train, y_train)
print(linreg.predict(X_test))
print(linreg.score(X_test, y_test))
from sklearn.linear_model import Ridge
ridge=Ridge()
ridge.fit(X_train, y_train)
print(ridge.predict(X_test))
print(ridge.score(X_test, y_test))