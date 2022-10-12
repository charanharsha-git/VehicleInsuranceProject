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
#from yellowbrick.cluster import KElbowVisualizer
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
import pickle

data=pd.read_csv("data/Customer Premium.csv")
print(data.head())
outlier_cols=["Customer Lifetime Value","Monthly Premium Auto","Total Claim Amount","Age"]

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




def label_encoding():
    global enc_data
    le = LabelEncoder()
    enc_data = reduced_data.copy()
    for i in cat_col:
        label_encoding = le.fit_transform(enc_data[i])
        enc_data[i] = label_encoding

def scaling_data():
    global scaled_data
    scaler = StandardScaler()
    scaler.fit(enc_data)
    scaled_data = pd.DataFrame(scaler.transform(enc_data), columns=enc_data.columns)

def clustering():
    AC = AgglomerativeClustering(n_clusters=4)
    # fit model and predict clusters
    yhat_AC = AC.fit_predict(enc_data)
    enc_data["Clusters"] = yhat_AC
    # Adding the Clusters feature to the orignal dataframe.
    data["Clusters"] = yhat_AC

def x_y():
    global X
    global Y
    X = enc_data.drop(columns=["Monthly Premium Auto"])
    Y = enc_data["Monthly Premium Auto"]




def pipelines():
    global lr_pipeline
    global DT_pipeline
    global RF_pipeline
    global SVR_pipeline
    lr_pipeline = Pipeline([('Linear_regression', LinearRegression())])
    DT_pipeline = Pipeline([('DT', DecisionTreeRegressor(random_state=0))])
    RF_pipeline = Pipeline([('RF', RandomForestRegressor(n_estimators=10, random_state=0))])
    SVR_pipeline = Pipeline([('SVR', SVR(kernel='linear'))])

def model_fit():
    global pipeline_list
    pipeline_list = [lr_pipeline, DT_pipeline, RF_pipeline, SVR_pipeline]
    for i in pipeline_list:
        i.fit(x_train, y_train)

def predict1():
    global lr_pipeline_pred
    global DT_pipeline_pred
    global RF_pipeline_pred
    global SVR_pipeline_pred
    global pred_list
    global pipelines_list
    lr_pipeline_pred = pd.Series()
    DT_pipeline_pred = pd.Series()
    RF_pipeline_pred = pd.Series()
    SVR_pipeline_pred = pd.Series()

    pipelines_list = [lr_pipeline, DT_pipeline, RF_pipeline, SVR_pipeline]
    pred_list = [lr_pipeline_pred, DT_pipeline_pred, RF_pipeline_pred, SVR_pipeline_pred]
    for i in range(0, len(pipelines_list)):
        pred_list[i] = pipelines_list[i].predict(x_test)

def predicted_df():
    global pred_df
    lr_pipeline_pred = pd.Series(pred_list[0])
    DT_pipeline_pred = pd.Series(pred_list[1])
    RF_pipeline_pred = pd.Series(pred_list[2])
    SVR_pipeline_pred = pd.Series(pred_list[3])
    pred_df = pd.DataFrame()
    pred_df["lr_pipeline_pred"] = lr_pipeline_pred
    pred_df["DT_pipeline_pred"] = DT_pipeline_pred
    pred_df["RF_pipeline_pred"] = RF_pipeline_pred
    pred_df["SVR_pipeline_pred"] = SVR_pipeline_pred
    pred_df["y_test"] = y_test.values
    print(pred_df)


variables=['State', 'Response', 'Coverage',
                       'Education','EmploymentStatus', 'Gender',
                       'Income', 'Marital Status','Sales Channel',
                       'Vehicle Class', 'Vehicle Size', 'Age',
                       'VehiclePrice','Monthly Premium Auto','Product Name','AgeOfVehicle']
reduced_data=data[variables]
cat_col,num_col=var_sep(reduced_data)
label_encoding()
scaling_data()
clustering()
x_y()
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
pipelines()
model_fit()
predict1()
predicted_df()

def mape():
    global MAPE_df
    MAPE_df = pd.DataFrame()
    for i in pred_df.columns:
        MAPE_df[i + " MAPE"] = pd.Series(mean_absolute_percentage_error(pred_df[i], pred_df["y_test"]))
    MAPE_df.T
    MAPE_df.drop(columns=["y_test MAPE"], inplace=True)

mape()

for i in range(0,len(MAPE_df.T)):
    if MAPE_df.T.iloc[i,0]==MAPE_df.T[0].min():
        pickle.dump(pipelines_list[i], open('best_premium_prediction_model.pkl', 'wb'))