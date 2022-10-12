import pandas as pd
import numpy as np
import datetime
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import colors
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, OrdinalEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt, numpy as np
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import AgglomerativeClustering
from matplotlib.colors import ListedColormap
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error,mean_absolute_error,mean_absolute_percentage_error,r2_score
from sklearn.metrics import f1_score, accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
import pickle

data=pd.read_csv("data/Fraud Claims.csv")
req_col=['Month', 'WeekOfMonth', 'DayOfWeek',
       'DayOfWeekClaimed', 'MonthClaimed', 'WeekOfMonthClaimed', 'Fault',
       'FraudFound_P', 'RepNumber', 'Days_Policy_Accident',
       'Days_Policy_Claim', 'PoliceReportFiled', 'WitnessPresent', 'AgentType',
       'NumberOfSuppliments', 'AddressChange_Claim', 'BasePolicy',
       'Vehicle Class', 'AccidentArea', 'Age', 'Product Name', 'VehiclePrice',
       'DriverRating', 'AgeOfVehicle']
reduced_data=data[req_col]
def var_sep(x):
    categorical_columns=[]
    numerical_columns=[]
    data_type=pd.DataFrame(x.dtypes).reset_index()
    data_type.columns=["variable_name","data_type"]
    for i in range(0,len(data_type)):
        if data_type.data_type[i]=="object":
            categorical_columns.append(data_type.variable_name[i])
        else:
            numerical_columns.append(data_type.variable_name[i])
    return categorical_columns,numerical_columns

cat_col,num_col=var_sep(reduced_data)
outlier_cols=["Age"]
def outlier_treatment():
    for i in range(0, len(outlier_cols)):
        pecentile10 = np.percentile(data[outlier_cols[i]], 10)
        percentile90 = np.percentile(data[outlier_cols[i]], 90)
        for j in range(0, len(data[outlier_cols[i]])):
            data[outlier_cols[i]][j] = np.where(data[outlier_cols[i]][j] > percentile90, percentile90,
                                                data[outlier_cols[i]][j])
            data[outlier_cols[i]][j] = np.where(data[outlier_cols[i]][j] < pecentile10, pecentile10,
                                                data[outlier_cols[i]][j])
outlier_treatment()

def label_encoding():
    global enc_data
    le = LabelEncoder()
    enc_data = reduced_data.copy()
    for i in cat_col:
        label_encoding = le.fit_transform(enc_data[i])
        enc_data[i] = label_encoding
label_encoding()

def x_y():
    global X
    global Y
    X = enc_data.drop(columns=["FraudFound_P"])
    Y = enc_data["FraudFound_P"]
x_y()

gaussian_pipeline=Pipeline([('Gaussian Naive Bayes',GaussianNB())])
SVC_pipeline=Pipeline([('Support Vector Classifier',SVC(kernel = 'rbf', probability = True))])
DTC_pipeline=Pipeline([('Decision Tree Classifier',DecisionTreeClassifier(random_state = 0))])
RFC_pipeline=Pipeline([('Random Forest Classifier',RandomForestClassifier(n_estimators = 100, random_state = 0))])
GBC_pipeline=Pipeline([('Gradient Boosting Classifier',GradientBoostingClassifier(random_state = 0))])
KNN_pipeline=Pipeline([('K-Nearest Neighbours',KNeighborsClassifier(n_neighbors=5))])

classifier_list=[gaussian_pipeline,SVC_pipeline,DTC_pipeline,RFC_pipeline,GBC_pipeline,KNN_pipeline]
classifiers = [GaussianNB(),
               SVC(kernel = 'rbf', probability = True),
               DecisionTreeClassifier(random_state = 0),
               RandomForestClassifier(n_estimators = 100, random_state = 0),
               GradientBoostingClassifier(random_state = 0),
               KNeighborsClassifier(n_neighbors=5)
              ]
classifier_names = ["Gaussian Naive Bayes",
                    "Support Vector Classifier",
                    "Decision Tree Classifier",
                    "Random Forest Classifier",
                    "Gradient Boosting Classifier",
                   "K-Nearest Neighbours"]
accuracies = []
x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.2)
pred_df2=pd.DataFrame()
def fit_predict():
    for i in range(len(classifier_list)):
        classifier_list[i].fit(x_train, y_train)
        y_pred = classifier_list[i].predict(x_test)
        pred_df2[i] = y_pred

fit_predict()

def acc():
    for i in range(len(classifiers)):
        accuracy = accuracy_score(y_test, pred_df2[i]) * 100
        accuracies.append(accuracy)

acc()

def model_selection():
    for i in range(0, len(accuracies)):
        if accuracies[i] == pd.Series(accuracies).max():
            pickle.dump(classifier_list[i], open('best_fraud_classification_model.pkl', 'wb'))

model_selection()

