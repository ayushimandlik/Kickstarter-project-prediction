import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')
import numpy as np
from sklearn import preprocessing
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.externals import joblib

df = pd.read_csv('ks-projects-201801.csv', encoding = "ISO-8859-1")
# Drop the columns that have values giving away the answer or the values that do not give any necessary addage to the model.
df.drop('ID', axis=1,inplace=True)
df.drop('pledged', axis=1,inplace=True)
df.drop('usd pledged', axis=1,inplace=True)
df.drop('goal', axis=1,inplace=True)
df.drop('usd_pledged_real', axis=1,inplace=True)
df.drop('name', axis=1,inplace=True)

# Converting non-numeric values to numeric ones:
le=preprocessing.LabelEncoder()
le.fit(df.category)
df.category=le.transform(df.category) 
# to see the corresponding names
#list(le.inverse_transform(le.transform(df.category.unique()) ))

le.fit(df.main_category)
df.main_category=le.transform(df.main_category) 

le.fit(df.currency)
df.currency=le.transform(df.currency)

le.fit(df.state)
df.state=le.transform(df.state)

le.fit(df.country)
df.country=le.transform(df.country)

# Make sure that all the date times are in the proper format in pandas
df['deadline'] = pd.to_datetime(df['deadline'])
df['launched'] = pd.to_datetime(df['launched'])

# to unix timestamp
df['deadline']=df['deadline'].astype(np.int64)
df['launched']=df['launched'].astype(np.int64)

y = df.state
X = df.drop('state', axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.2, 
                                                    random_state=123, 
                                                    stratify=y)
 
# 5. Declare data preprocessing steps
pipeline = make_pipeline(preprocessing.StandardScaler(), 
                         RandomForestRegressor(n_estimators=100))
 
# 6. Declare hyperparameters to tune
hyperparameters = { 'randomforestregressor__max_features' : ['auto', 'sqrt', 'log2'],
                  'randomforestregressor__max_depth': [None, 5, 3, 1]}
 
# 7. Tune model using cross-validation pipeline
clf = GridSearchCV(pipeline, hyperparameters, cv=10)
 
clf.fit(X_train, y_train)
 
# 9. Evaluate model pipeline on test data
pred = clf.predict(X_test)

acc = rf.score(X_test, y_test)

print(acc)