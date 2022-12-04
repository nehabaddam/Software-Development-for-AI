#importing pandas and numpy libraries
from flask import Flask, render_template, send_file, make_response, url_for, Response, redirect, request 
 
#initialise app
app = Flask(__name__)

import pandas as pd
import numpy as np

#importing utility libraries
import math
import warnings
import string

# Matplotlib libraries
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerLine2D


# sklearn libraries
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, plot_confusion_matrix, ConfusionMatrixDisplay
from sklearn import metrics
from sklearn.model_selection import GridSearchCV


#importing classifier
from sklearn.ensemble import RandomForestClassifier

#importing Seaborn
import seaborn as sns

#ignoring all warinings
warnings.filterwarnings("ignore")  

df = pd.read_csv("C:/Users/badda/OneDrive/Desktop/flask_app/cardio_train.csv",delimiter=";",skiprows= [49961])
df.head()

length = df.shape[0]*df.shape[1]
# print('Length of the data frame : ', length)

df=df.drop_duplicates()
df.drop(['id'], axis=1,inplace=True)
df['age'] = df['age'].apply(lambda x: x/365)  
outliers = len(df[(df["sys_bp"]>=280) | (df["dia_bp"]>=220) | (df["dia_bp"] < 0) | (df["sys_bp"] < 0) | (df["sys_bp"]<df["dia_bp"])])

# print(f'total {outliers} outliers')
# print(f'percent missing: {round(outliers/len(df)*100,1)}%')
df = df[ (df['dia_bp'] >= 0) & (df['sys_bp'] >= 0) ]  
df = df[ (df['dia_bp'] <= 220) & (df['sys_bp'] <= 280) ]
df = df[ (df['dia_bp'] < df['sys_bp']) ]  

#getting the first 5th percentile of the data
Quartile1_hi = df['sys_bp'].quantile(0.05) 
#getting the 95th percentile of the data
Quartile3_hi = df['sys_bp'].quantile(0.95)
InterQuartileRange_hi = Quartile3_hi - Quartile1_hi
lower, upper = Quartile1_hi - 1.5 * InterQuartileRange_hi, Quartile3_hi + 1.5 * InterQuartileRange_hi
df = df[(df['sys_bp'] >= lower) & (df['sys_bp'] <= upper)]  
#getting the first 5th percentile of the data
Quartile1_lo = df['dia_bp'].quantile(0.05) 
#getting the 95th percentile of the data
Quartile3_lo = df['dia_bp'].quantile(0.95)  # 95th percentile of the data of the given feature
InterQuartileRange_lo = Quartile3_lo - Quartile1_lo
lower, upper = Quartile1_lo - 1.5 * InterQuartileRange_lo, Quartile3_lo + 1.5 * InterQuartileRange_lo
df = df[(df['dia_bp'] >= lower) & (df['dia_bp'] <= upper)]  

def detect_outliers(df,q1,q3):
  for col in df.columns:
    df_feature = df[col]
    Quartile1 = df_feature.quantile(q1) # 5th percentile of the data of the given feature
    Quartile3 = df_feature.quantile(q3)  # 95th percentile of the data of the given feature
    IQR = Quartile3 - Quartile1    #IQR is interquartile range. 
    #print(f'Feature: {col}')
    #print(f'Percentiles: {int(q1*100)}th={Quartile1}, {int(q3*100)}th={Quartile3}, IQR={IQR}')
    # calculate the outlier lower and upper bound
    lower, upper = Quartile1 - 1.5 * IQR, Quartile3 + 1.5 * IQR
    # identify outliers
    outliers = [x for x in df_feature if x < lower or x > upper]
    # print('Identified outliers: %d \n' % len(outliers))
  
detect_outliers(df[['height', 'weight']],0.05,0.95)

df_cleaned = df 
for col in ['height','weight']:
  Quartile1 = df[col].quantile(0.05) # 5th percentile of the data of the given feature
  Quartile3 = df[col].quantile(0.95)  # 95th percentile of the data of the given feature
  InterQuartileRange = Quartile3 - Quartile1
  lower, upper = Quartile1 - 1.5 * InterQuartileRange, Quartile3 + 1.5 * InterQuartileRange
  df_cleaned = df_cleaned[(df_cleaned[col] >= lower) & (df_cleaned[col] <= upper)]  


df_cleaned['BMI'] = round(df_cleaned['weight']/((df_cleaned['height']*0.0328084)),1)
df_cleaned.head()
df_cleaned = df_cleaned[ (df_cleaned['BMI'] < 60) & (df_cleaned['BMI'] > 10)]
df_scaled=df_cleaned.copy()

columns_to_scale = ['age', 'weight', 'sys_bp', 'dia_bp','cholesterol','gender','BMI','height']

scaler = StandardScaler()
df_scaled[columns_to_scale] = scaler.fit_transform(df_cleaned[columns_to_scale])

df_scaled.head()

#we perform some Standardization using minmaxscaler
df_scaled_mm=df_cleaned.copy()

columns_to_scale_mm = ['age', 'weight', 'sys_bp', 'dia_bp','cholesterol','gender','BMI','height']

mmscaler = MinMaxScaler()
df_scaled_mm[columns_to_scale_mm] = mmscaler.fit_transform(df_cleaned[columns_to_scale_mm])

df_scaled_mm.head()


X = df_cleaned.drop(['target'], axis=1) #features 
y = df_cleaned['target']  #target feature

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle = True)


#Train-test-split for scaled data
X_scaled = df_scaled.drop(['target'], axis=1) #features 
y_scaled = df_scaled['target']  #target feature

X_scaled_mm = df_scaled_mm.drop(['target'], axis=1) #features 
y_scaled_mm = df_scaled_mm['target']  #target feature

X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42, shuffle = True)
X_train_scaled_mm, X_test_scaled_mm, y_train_scaled_mm, y_test_scaled_mm = train_test_split(X_scaled_mm, y_scaled_mm, test_size=0.2, random_state=42, shuffle = True)

rndForestClassifier = GridSearchCV(estimator=RandomForestClassifier(), param_grid={'n_estimators': [500], 'max_features':['sqrt'], 'max_depth':[20], 'max_leaf_nodes':[2,5,10,50,100,200,300,400,500,750,1000]}, cv=5, scoring=['accuracy','recall'], refit='accuracy').fit(X_train, y_train)

rf = RandomForestClassifier(n_estimators=500, max_depth=20, max_features='sqrt', max_leaf_nodes=750)
rf.fit(X_train,y_train)

y_predict = rf.predict(X_test)
y_predicted = np.array(y_predict > 0.5, dtype=float)

rndForest_acc = accuracy_score(y_test, y_predicted)
cm = confusion_matrix(y_test, y_predicted)
rndForest_tpr = cm[1][1] /(cm[1][0] + cm[1][1])
rndForest_report = classification_report(y_test, y_predicted)

print(X_test.columns)
print(X_test.head(3))

@app.route('/' )
@app.route('/home')
def home():
    return render_template('home.html')

@app.route('/result',methods=['POST', 'GET'])
def result():
    
    d = {'age':[], 'gender':[], 'height':[], 'weight':[], 'sys_bp':[], 'dia_bp':[], 'cholesterol':[],
       'glucose':[], 'smoke':[], 'alco':[], 'active':[], 'BMI':[]}
    output = request.form.to_dict()

    d["age"].append(output["age"])
    d["gender"].append(int(output["gender"]))
    d["height"].append(int(output["height"]))
    d["weight"].append(int(output["weight"]))
    d["sys_bp"].append(int(output["sys_bp"]))
    d["dia_bp"].append(int(output["dia_bp"]))
    d["cholesterol"].append(int(output["cholesterol"]))
    d["glucose"].append(int(output["glucose"]))
    d["smoke"].append( int(output["smoke"]))
    d["alco"].append(int(output["alco"]))
    d["active"].append(int(output["active"]))
    bmi = round(int(output["weight"])/((int(output["height"])*0.0328084)),1)
    d["BMI"].append(bmi)

    df1 = pd.DataFrame(d)
    pred = rf.predict(df1)
    if pred[0]==1:
      note = "You're vulnerable"
    elif pred[0]==0:
      note = "Woohoo! you look perfect"
    else:
      note = "No result"
    return render_template('result.html', note= note)


if __name__ == '__main__':
    app.run(debug = True)