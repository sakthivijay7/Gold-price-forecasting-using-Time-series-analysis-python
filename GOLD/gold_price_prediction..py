 ###             Import Libraries
import pandas as pd

##                Read the file
path="D:/ALL/Gold price/historical_gold_price.csv"
ds=pd.read_csv(path).copy()
# print(ds.columns)
df=ds[["Date","Closing price   XAU/USD (Troy Ounce=31.1035 grams)"]].copy()
print(df.head(5))

####        Null values check
null=df.isnull()
print(df.info())
print(null)
print(df.describe())

####       Datetime features Extraction
df["Date"]=pd.to_datetime(df["Date"])
df["year"]=df["Date"].dt.year
df["month"]=df["Date"].dt.year
df["day"]=df["Date"].dt.day
df["weekday"]=df["Date"].dt.dayofweek
df["is_weekend"]=df["Date"].dt.dayofweek >=5
df["week"]=df["Date"].dt.isocalendar().week
df["is_leap_year"]=df["Date"].dt.is_leap_year
extract_date=df[["Date","year","month","day","weekday","is_weekend","week","is_leap_year"]].copy()
print(extract_date.head())

#####       Convert it Integer
extract_date["week"]=extract_date["week"].astype(int)
extract_date["is_weekend"]=extract_date["is_weekend"].astype(int)
extract_date["is_leap_year"]=extract_date["is_leap_year"].astype(int)
print(extract_date.dtypes)

 ###         Features and Target Segregate
x=extract_date.drop(columns=["Date"])
print(x.head())
y=df["Closing price   XAU/USD (Troy Ounce=31.1035 grams)"]
print(y.head())

###   Train(80%) and Test (20%) split
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.20,random_state=0)
print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)

###     Scalling Features convert into specific range (0-1)
from sklearn.preprocessing import StandardScaler
scale=StandardScaler()
x_train_scale=scale.fit_transform(x_train)
x_test_scale=scale.transform(x_test)
print(x_test_scale)
print(x_test_scale.shape)


### REGRESSION ALGORITHMS

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from xgboost import XGBRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor

#### Regression Models
poly=PolynomialFeatures(degree=2)
x_train_poly=poly.fit_transform(x_train)
x_test_poly=poly.transform(x_test)
print(x_train_poly.shape)
print(x_test_poly.shape)

linear_model=LinearRegression()
poly_model=LinearRegression()
xgb_model=XGBRegressor(n_estimators=100)
tree_model=DecisionTreeRegressor()
forest_model=RandomForestRegressor(n_estimators=100)
gradient_model=GradientBoostingRegressor(n_estimators=100)
vector_model=SVR()
knn_model=KNeighborsRegressor(n_neighbors=5)

#### ALL models in one file

Models={"LinearRegression":linear_model,
         "PolynomialFeatures":poly_model,
         "XGBRegressor":xgb_model,
         "DecisionTreeRegressor":tree_model,
         "RandomForestRegressor":forest_model,
         "GradientBoostingRegressor":gradient_model,
         "SVR":vector_model,
         "KNeighborsRegressor":knn_model}
print(x_train.dtypes)
print(x_test.dtypes)
print(y_train.dtypes)
print(y_test.dtypes)

##     Remove Currency's commas
# print(y_train.head())
# y_train=y_train.str.replace(",","")
# y_train=y_train.astype(float)
# y_test=y_test.str.replace(",","")
# y_test=y_test.astype(float)
print(y_train.dtypes)
print(y_test.dtypes)

###    Train the models

linear_model.fit(x_train,y_train)
poly_model.fit(x_train_poly,y_train)
xgb_model.fit(x_train,y_train)
tree_model.fit(x_train,y_train)
forest_model.fit(x_train,y_train)
gradient_model.fit(x_train,y_train)
vector_model.fit(x_train_scale,y_train)
knn_model.fit(x_train_scale,y_train)

### Trained Models Load in Pickle file
import joblib
with open("Trained_models.pkl","wb")  as file:
    joblib.dump(Models,file)

###  MODEL PREDICTION
linear_ypred=linear_model.predict(x_test)
poly_ypred=poly_model.predict(x_test_poly)
xgb_ypred=xgb_model.predict(x_test)
tree_ypred=tree_model.predict(x_test)
forest_ypred=forest_model.predict(x_test)
gradient_ypred=gradient_model.predict(x_test)
vector_ypred=vector_model.predict(x_test_scale)
knn_ypred=knn_model.predict(x_test_scale)

###        R square score(High) and Mean squared error(Low)

from sklearn.metrics import r2_score,mean_squared_error
print(" LinearRegressor:" ,'\n',"R2_score",r2_score(y_test,linear_ypred),'\n',"MSE:",mean_squared_error(y_test,linear_ypred),'\n')
print("PolynomialFeatures:",'\n',"R2_score",r2_score(y_test,poly_ypred),'\n',"MSE",mean_squared_error(y_test,poly_ypred),'\n')
print("XGBRegressor:",'\n',"R2_score",r2_score(y_test,xgb_ypred),'\n',"MSE",mean_squared_error(y_test,xgb_ypred),'\n')
print("DecisionTreeRegressor:",'\n',"R2_score",r2_score(y_test,tree_ypred),'\n',"MSE",mean_squared_error(y_test,tree_ypred),'\n')
print("RandomForestRegressor:",'\n',"R2_score",r2_score(y_test,forest_ypred),'\n',"MSE",mean_squared_error(y_test,forest_ypred),'\n')
print("GradientBoostRegrssor:",'\n',"R2_score",r2_score(y_test,gradient_ypred),'\n',"MSE",mean_squared_error(y_test,gradient_ypred),'\n')
print("SupportVectorRegressor:",'\n',"R2_score",r2_score(y_test,vector_ypred),'\n',"MSE",mean_squared_error(y_test,vector_ypred),'\n')
print("KNeighborsRegressor:",'\n',"R2_score",r2_score(y_test,knn_ypred),'\n',"MSE",mean_squared_error(y_test,knn_ypred))



