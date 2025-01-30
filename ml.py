import numpy as np
import pandas as pd
import warnings
import pickle

warnings.filterwarnings("ignore")

df = pd.read_csv("weatherHistory.csv")
df.rename(columns={"Temperature (C)":"Temperature(C)","Wind Speed (km/h)":"Wind_Speed(km/h)"}, inplace=True)

a = 'Dangerously Windy and Partly Cloudy'
li_of_list = [w.split() for w in a.split(',')]

from itertools import chain
def li(x):
  li_of_list = [w.split() for w in x.split(',')]
  flat_li = list(chain(*li_of_list))
  return flat_li

def get_weather(li1):
  if 'Breezy' in li1 and 'Cloudy' in li1:
    return 'Cloudy'
  elif 'Dry' in li1 and 'Cloudy' in li1:
    return 'Cloudy'
  elif 'Windy' in li1 and 'Cloudy' in li1:
    return 'Cloudy'
  elif 'Windy' in li1 and 'Overcast' in li1:
    return 'Overcast'
  elif 'Breezy' in li1 and 'Overcast' in li1:
    return 'Overcast'
  elif 'Humid' in li1 and 'Cloudy' in li1:
    return 'Cloudy'
  elif 'Breezy' in li1 and 'Foggy' in li1:
    return 'Cloudy'
  elif 'Windy' in li1 and 'Overcast' in li1:
    return 'Overcast'
  elif 'Humid' in li1 and 'Foggy' in li1:
    return 'Foggy'
  elif 'Cloudy' in li1:
    return 'Cloudy'
  elif 'Overcast' in li1:
    return 'Overcast'
  elif 'Clear' in li1:
    return 'Clear'
  elif 'Foggy' in li1:
    return 'Foggy'
  elif 'Drizzle' in li1:
    return 'Rain'
  elif 'Dry' in li1:
    return 'Clear'
  else:
    return 'Rain'

df['std_weather'] = df['Summary'].apply(lambda x : get_weather(li(x)))

from sklearn.preprocessing import LabelEncoder
label_Encode = LabelEncoder()
df['std_weather'] = label_Encode.fit_transform(df['std_weather'])
print(label_Encode.classes_)
df = df.drop(columns = ['Summary'],axis = 1) 
print(df)
df = np.array(df) 

X = df[1:,0:-1]

y = df[1:,-1]

y = y.astype('int')

X= X.astype('int')

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x_std = scaler.fit_transform(X)
x_std.mean(axis=0)
x_std.std(axis=0)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x_std, y, test_size=0.2, random_state=42)
X_train.shape,X_test.shape

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
dt_model = DecisionTreeClassifier()
dt_model.fit(X_train,y_train)
y_pred_dt =  dt_model.predict(X_test)

from sklearn.ensemble import RandomForestClassifier
rf_m = RandomForestClassifier()
rf_m.fit(X_train,y_train)
y_pred_rf = rf_m.predict(X_test)

from sklearn.model_selection import GridSearchCV,RandomizedSearchCV
para = {'n_estimators':[50,100],
        'max_features':['sqrt','log2',None]
        }

g_search = GridSearchCV(estimator = rf_m,param_grid = para)
g_search.fit(X_train,y_train)
g_search.best_params_

rf_model = RandomForestClassifier(max_features=None,n_estimators=100)
rf_model.fit(X_train,y_train)

pickle.dump(rf_model, open('ml.pkl','wb'))
