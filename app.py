import pandas as pd
import numpy as np
import pickle
from flask import Flask, render_template, request
import time

from datetime import datetime
from datetime import date
#import damage_detection as dd
#from modelpy import models
import secrets
import shutil
import sys
import random
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.models import load_model
from matplotlib import image as mpimg
from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator
import os

from werkzeug.utils import secure_filename

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
car_photo_upload_path='static/upload/car_photo'
pdf_doc_upload_path='static/upload/pdf_doc'


import io
import random
from flask import Response
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure



app = Flask(__name__)



global usr_pwd

usr_pwd=pd.read_csv("data/Username Password.csv")

@app.route('/CapstoneProject',methods=["POST","GET"])
def login():
    if request.method == "POST":

        return render_template("landingpage.html")
    return render_template("landingpage.html")

@app.route("/Login Page" , methods=['GET', 'POST'])
def test():
    global select
    select = request.form.get('comp_select')
    return render_template(str(select)+".html",select=select)

@app.route('/Employee Login',methods=["POST","GET"])
def employeelogin():
    if request.method == "POST":
        username = request.form.get('username')
        password= request.form.get('password')
        if username=="EMP001" and password=="EMP001":

            return render_template("Employee Dashboard.html",user=username)
        if username=="EMP002" and password=="EMP002":

            return render_template("Employee Dashboard2.html",user=username)
        elif username=="ADJ" and password=="ADJ":
            return render_template("test.html")
        else:
            error="Enter Correct Login Credentials"

        return render_template(str(select)+".html",error=error)
    return render_template(str(select)+".html")

@app.route('/Customer Login',methods=["POST","GET"])
def cxlogin():
    if request.method == "POST":
        username = request.form.get('username')
        password= request.form.get('password')

        if (username in usr_pwd['Username'].values) and (password in usr_pwd['Password'].values) :
            cx_df=pd.read_csv("data/New CX.csv")
            cx_details=list(cx_df.loc[np.where(cx_df["Customer ID"] == username)[0][0]])

            return render_template("claim_submission.html",user=username,cx_details=cx_details)


        else:
            error="Enter Correct Login Credentials"

        return render_template("Already Insured.html",error=error)
    return render_template("Already Insured.html")
new_enq=pd.read_csv("data/New ENQ.csv")
@app.route('/Enquiry Signup',methods=["POST","GET"])
def enquirysignup():
    global new_cx
    global input2
    global b
    global enquiry

    if request.method == "POST":
        Name = request.form.get('customer_name')
        Phone= request.form.get('phone_number')
        Email = request.form.get('email_address')
        State = request.form.get('comp_select1')
        Response = request.form.get('comp_select2')
        Coverage = request.form.get('comp_select3')
        Education = request.form.get('comp_select4')
        EmploymentStatus = request.form.get('comp_select5')
        Income=request.form.get('Income')
        Age = request.form.get('Age')
        Gender = request.form.get('comp_select6')
        MaritalStatus = request.form.get('comp_select7')
        VehicleClass = request.form.get('comp_select8')
        VehicleSize = request.form.get('comp_select9')
        VehiclePrice= request.form.get('comp_select10')
        SalesChannel= request.form.get('comp_select11')
        Clusters=3
        ProductName=request.form.get('comp_select12')
        AgeofVehicle=request.form.get('comp_select13')

        enquiry=list((str(Name),str(Phone),str(Email),str(State), str(Response), str(Coverage),
              str(Education), str(EmploymentStatus),
              str(Gender), int(Income), str(MaritalStatus),
              str(SalesChannel), str(VehicleClass),
              str(VehicleSize), int(Age), str(VehiclePrice), int(Clusters),str(ProductName),str(AgeofVehicle)))


        new_enq_len = len(new_enq)
        new_enq.loc[new_enq_len] = enquiry
        new_enq.to_csv(path_or_buf="data/New ENQ.csv",index=False)
        print(new_enq)

        state_dict = {'Washington': 4,
                      'Arizona': 0,
                      'Nevada': 2,
                      'California': 1,
                      'Oregon': 3}
        response_dict = {'No': 0,
                         'Yes': 1}
        coverage_dict = {'Basic': 0,
                         'Extended': 1,
                         'Premium': 2}
        education_dict = {'Bachelor': 0,
                          'College': 1,
                          'Master': 4,
                          'High School or Below': 3,
                          'Doctor': 2}
        employment_dict = {'Employed': 1,
                           'Unemployed': 4,
                           'Medical Leave': 2,
                           'Disabled': 0,
                           'Retired': 3}
        gender_dict = {'Female': 0,
                       'Male': 1}
        marital_dict = {'Married': 1,
                        'Single': 2,
                        'Divorced': 0}
        saleschannel_dict = {'Agent': 0,
                             'Call Center': 2,
                             'Web': 3,
                             'Branch': 1}
        vehicletype_dict = {'Two-Door Car': 5,
                            'Four-Door Car': 0,
                            'SUV': 3,
                            'Luxury SUV': 2,
                            'Sports Car': 4,
                            'Luxury Car': 1}
        vehiclesize_dict = {'Medium Size': 1,
                            'Small': 2,
                            'Large': 0}

        vehicleprice_dict = {'more than 69000': 5,
                             '20000 to 29000': 0,
                             '30000 to 39000': 1,
                             'less than 20000': 4,
                             '40000 to 59000': 2,
                             '60000 to 69000': 3}
        ageofvehicle_dict={"New":1,
                          "2 Years":2,
                          "3 Years":3,
                          "4 Years":4,
                          "5 Years": 5,
                          "6 Years": 6,
                          "7 Years": 7,
                          "more than 7":0}
        productname_dict={"Sedan - Liability":2,
                          "Sedan - Collision":1,
                          "Sedan - All Perils":0,
                          "Sport - Liability":5,
                          "Sport - Collision":4,
                          "Sport - All Perils":3,
                          "Utility - Liability":8,
                          "Utility - Collision":7,
                          "Utility - All Perils":6}
        global product_list
        product_list=["Sedan - Liability",
                          "Sedan - Collision",
                          "Sedan - All Perils",
                          "Sport - Liability",
                          "Sport - Collision",
                          "Sport - All Perils",
                          "Utility - Liability",
                          "Utility - Collision",
                          "Utility - All Perils"]

        #product_name=0



        pickled_model = pickle.load(open('best_premium_prediction_model.pkl', 'rb'))
        global premium_list
        premium_list=[]
        addl_price=[33,35,36,23,25,26,13,15,16]
        for i in range(len(product_list)):
            product_name=product_list[i]
            input1 = list((state_dict[str(State)], response_dict[str(Response)], coverage_dict[str(Coverage)],
                           education_dict[str(Education)], employment_dict[str(EmploymentStatus)],
                           gender_dict[str(Gender)], int(Income), marital_dict[str(MaritalStatus)],
                           saleschannel_dict[str(SalesChannel)], vehicletype_dict[str(VehicleClass)],
                           vehiclesize_dict[str(VehicleSize)], int(Age), vehicleprice_dict[str(VehiclePrice)],
                           int(Clusters), productname_dict[str(product_name)], ageofvehicle_dict[str(AgeofVehicle)]))
            a = input1
            b = pickled_model.predict([a[0:14]])
            premium_list.append(b[0]+addl_price[i])

            input_clasification = list((state_dict[str(State)], response_dict[str(Response)], coverage_dict[str(Coverage)],
                           education_dict[str(Education)], employment_dict[str(EmploymentStatus)],
                           gender_dict[str(Gender)], int(Income), marital_dict[str(MaritalStatus)],
                           saleschannel_dict[str(SalesChannel)], vehicletype_dict[str(VehicleClass)],
                           vehiclesize_dict[str(VehicleSize)], int(Age), vehicleprice_dict[str(VehiclePrice)],
                           ageofvehicle_dict[str(AgeofVehicle)]))

            pickled_model1=pickle.load(open('best_product_classification_model.pkl', 'rb'))
            c = pickled_model1.predict([input_clasification[0:14]])
            product_suggestion=c[0]

            dict_product = {2:"Sedan - Liability",
                               1: "Sedan - Collision",
                                0:"Sedan - All Perils",
                                5:"Sport - Liability",
                                4:"Sport - Collision",
                                3:"Sport - All Perils",
                                8:"Utility - Liability",
                                7:"Utility - Collision",
                                6:"Utility - All Perils"}


        global input2
        global premium
        global product
        premium=0
        product=0



        input2 = list((state_dict[str(State)], response_dict[str(Response)], coverage_dict[str(Coverage)],
                       education_dict[str(Education)], employment_dict[str(EmploymentStatus)],
                       gender_dict[str(Gender)], int(Income), marital_dict[str(MaritalStatus)], premium,
                       saleschannel_dict[str(SalesChannel)], vehicletype_dict[str(VehicleClass)],
                       vehiclesize_dict[str(VehicleSize)], int(Age), vehicleprice_dict[str(VehiclePrice)],
                       int(Clusters), product, ageofvehicle_dict[str(AgeofVehicle)]))
        enquiry = list((str(Name), str(Phone), str(Email), str(State), str(Response), str(Coverage),
                        str(Education), str(EmploymentStatus),
                        str(Gender), int(Income), str(MaritalStatus),
                        str(SalesChannel), str(VehicleClass),
                        str(VehicleSize), int(Age), str(VehiclePrice), int(Clusters), str(product),
                        str(AgeofVehicle)))

        pickled_model_accuracy = pickle.load(open('best_product_classification_model_accuracy.pkl', 'rb'))



        return render_template("new_cx_product_recommendation.html",b=premium,enquiry=enquiry,inp=input2,product_list=product_list,premium_list=premium_list,product_suggestion=dict_product[product_suggestion],pickled_model_accuracy=int(pickled_model_accuracy))

    return render_template("new_cx_product_recommendation.html")

new_cx=pd.read_csv("data/New CX.csv")

@app.route('/Congratulations',methods=["POST","GET"])
def saledone():
    global cx
    if request.method == "POST":
        premium = premium_list[int(request.form.get("select_product"))]
        product = product_list[int(request.form.get("select_product"))]
        cx=enquiry.copy()
        cx[17]=product
        cx.append(premium)
        c = input2
        c[8]=premium

        clv_model = pickle.load(open('clv_prediction_model.pkl', 'rb'))
        clv = clv_model.predict([c[0:17]])[0]
        cx.append(clv)
        new_cx_len = len(new_cx)
        cust_id="CX000"+str(new_cx_len)
        cx.append(cust_id)
        new_cx.loc[new_cx_len] = cx
        new_cx.to_csv(path_or_buf="data/New CX.csv", index=False)
        print(new_cx)
        login_cred=[]
        login_cred_len=len(usr_pwd)
        login_cred.append(cust_id)
        login_cred.append(cust_id)
        usr_pwd.loc[login_cred_len]=login_cred
        usr_pwd.to_csv(path_or_buf="data/Username Password.csv", index=False)




        return render_template("sale_done.html",cust_id=cust_id,password=cust_id)

    return render_template("sale_done.html")

@app.route('/Claim Submission',methods=["POST","GET"])
def claimsubmission():
    if request.method == "POST":
        car_photo = request.files['car_photo']
        pdf_doc = request.files['pdf_doc']


        def week_number_of_month(date_value):
            return (date_value.isocalendar()[1] - date_value.replace(day=1).isocalendar()[1] + 1)
        weekday_dict={1:"Monday",2:"Tuesday",3:"Wednesday",4:"Thursday",5:"Friday",6:"Saturday",7:"Sunday"}
        month_dict = {1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun", 7: "Jul", 8: "Aug", 9: "Sep",
                      10: "Oct", 11: "Nov", 12: "Dec"}
        claimed_on_date = datetime.now()
        claim_year=int(str(claimed_on_date)[0:4])
        claim_month = int(str(claimed_on_date)[5:7])
        claim_month_name=month_dict[claim_month]
        claim_date = int(str(claimed_on_date)[8:10])
        claim_date_weekday=weekday_dict[claimed_on_date.isoweekday()]
        claim_date_weekno=week_number_of_month(datetime(year=claim_year, month=claim_month, day=claim_date).date())
        DateofIncident = request.form.get('incident_date')

        DateofIncident_year= int(DateofIncident[0:4])
        DateofIncident_month = int(DateofIncident[5:7])
        DateofIncident_month_name = month_dict[DateofIncident_month]
        DateofIncident_date = int(DateofIncident[8:10])
        date_of_incident=date(DateofIncident_year,DateofIncident_month,DateofIncident_date)
        accident_weekday=weekday_dict[date_of_incident.isoweekday()]
        accident_weekno=week_number_of_month(datetime(year=DateofIncident_year, month=DateofIncident_month, day=DateofIncident_date).date())
        Fault = request.form.get('comp_select1')
        PoliceReportFiled = request.form.get('comp_select2')
        WitnessPresent = request.form.get('comp_select3')
        NoOfSuppliments = request.form.get('comp_select5')
        AddressChangeClaim = request.form.get('comp_select6')
        BasePolicy = request.form.get('comp_select7')
        AccidentArea = request.form.get('comp_select8')
        AgentType = request.form.get('comp_select10')
        Age = request.form.get('Age')

        Days_Policy_Claim = request.form.get('comp_select11')
        VehicleClass = request.form.get('comp_select9')

        Days_Policy_Accident = request.form.get('comp_select12')
        RepNumber = request.form.get('rep_number')
        Driver_rating = request.form.get('driver_rating')
        AgeofVehicle = request.form.get('comp_select4')

        Month_dict={"Jan":4, "Feb":3, "Mar":7, "Apr":0, "May":8, "Jun":6, "Jul":5, "Aug":1, "Sep":11,
                      "Oct":10, "Nov":9, "Dec":2}

        DayOfWeek_dict={"Monday":1,
                     "Tuesday":5,
                     "Wednesday":6,
                     "Thursday":4,
                     "Friday":0,
                     "Saturday":2,
                     "Sunday":3}


        Fault_dict={"Policy Holder":0,
                "Third Party":1
                }


        Days_Policy_dict={"more than 30":0,
                            "1 to 7":2,
                            "15 to 30":4,
                            "8 to 15":3,
                            "none":1}

        PoliceReportFiled_dict={"Yes":1,"No":0}
        WitnessPresent_dict={"Yes":1,"No":0}
        AgentType_dict={"External":0,"Internal":1}

        AddressChange_Claim_dict={"1 year":2,
                            "2 to 3 years":3,
                            "4 to 8 years":4,
                            "no change":0,
                            "under 6 months":1}
        BasePolicy_dict={"Liability":2,
                    "Collision":1,
                    "All Perils":0}
        Vehicle_class_dict={'Two-Door Car': 5,
                            'Four-Door Car': 0,
                            'SUV': 3,
                            'Luxury SUV': 2,
                            'Sports Car': 4,
                            'Luxury Car': 1}

        AccidentArea_dict={"Rural":0,"Urban":1}

        ProductName_dict={"Sedan - Liability":2,
                          "Sedan - Collision":1,
                          "Sedan - All Perils":0,
                          "Sport - Liability":5,
                          "Sport - Collision":4,
                          "Sport - All Perils":3,
                          "Utility - Liability":8,
                          "Utility - Collision":7,
                          "Utility - All Perils":6}

        AgeOfVehicle_dict={"New":1,
                          "2 Years":2,
                          "3 Years":3,
                          "4 Years":4,
                          "5 Years": 5,
                          "6 Years": 6,
                          "7 Years": 7,
                          "more than 7":0}
        vehicleprice_dict = {'more than 69000': 5,
                             '20000 to 29000': 0,
                             '30000 to 39000': 1,
                             'less than 20000': 4,
                             '40000 to 59000': 2,
                             '60000 to 69000': 3}
        ProductName=1
        VehiclePrice=2
        AgeOfVehicle=1
        NoOfSuppliments=3
        model_input=list((Month_dict[str(DateofIncident_month_name)],
                          int(accident_weekno),
                          DayOfWeek_dict[str(accident_weekday)],
                          DayOfWeek_dict[str(claim_date_weekday)],
                          Month_dict[str(claim_month_name)],
                          int(claim_date_weekno),
                          Fault_dict[str(Fault)],
                          int(RepNumber),
                          Days_Policy_dict[str(Days_Policy_Accident)],
                          Days_Policy_dict[str(Days_Policy_Claim)],
                          PoliceReportFiled_dict[str(PoliceReportFiled)],
                          WitnessPresent_dict[str(WitnessPresent)],
                          AgentType_dict[str(AgentType)],
                          int(NoOfSuppliments),
                          AddressChange_Claim_dict[str(AddressChangeClaim)],
                          BasePolicy_dict[str(BasePolicy)],
                          Vehicle_class_dict[str(VehicleClass)],
                          AccidentArea_dict[str(AccidentArea)],
                          int(Age),
                          int(ProductName),
                          int(VehiclePrice),
                          int(Driver_rating),
                          int(AgeOfVehicle)))
        new_claim = list((str(DateofIncident_month_name),
                            int(accident_weekno),
                            str(accident_weekday),
                            str(claim_date_weekday),
                            str(claim_month_name),
                            int(claim_date_weekno),
                            str(Fault),
                            int(RepNumber),
                            str(Days_Policy_Accident),
                            str(Days_Policy_Claim),
                            str(PoliceReportFiled),
                            str(WitnessPresent),
                            str(AgentType),
                            int(NoOfSuppliments),
                            str(AddressChangeClaim),
                            str(BasePolicy),
                            str(VehicleClass),
                            str(AccidentArea),
                            int(Age),
                            int(ProductName),
                            int(VehiclePrice),
                            int(Driver_rating),
                            int(AgeOfVehicle)))


        fraud_detection_model = pickle.load(open('best_fraud_classification_model.pkl', 'rb'))
        res = fraud_detection_model.predict([model_input[0:24]])

        new_claims = pd.read_csv("data/New Claims.csv")
        fraud_pred = res[0]
        new_claim.append(fraud_pred)
        new_claims_len = len(new_claims)
        claim_id = "NCLM000" + str(new_claims_len)
        new_claim.append(claim_id)
        new_claims.loc[new_claims_len] = new_claim
        new_claims.to_csv(path_or_buf="data/New Claims.csv", index=False)

        claim_adj = pd.read_csv("data/Claim Status.csv")

        new_claim.append("Under Process")
        claim_adj_len = len(claim_adj)

        claim_adj.loc[claim_adj_len] = new_claim
        claim_adj.to_csv(path_or_buf="data/Claim Status.csv", index=False)

        car_photo.filename = str(claim_id)+ str("photo.jpeg")
        car_photo_filename = secure_filename(car_photo.filename)
        car_photo.save(os.path.join(car_photo_upload_path, car_photo_filename))
        pdf_doc.filename = str(claim_id) + str("pdf.pdf")
        pdf_doc_filename = secure_filename(pdf_doc.filename)
        pdf_doc.save(os.path.join(pdf_doc_upload_path, pdf_doc_filename))

        AUTOTUNE = tf.data.experimental.AUTOTUNE
        IMG_SIZE = 150
        BATCH_SIZE = 32
        EPOCHS = 50

        def predictImage(filename):
            model = load_model('best_model.hdf5')
            img = cv2.imread(filename)
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            plt.imshow(img)

            Y = np.array(img)
            X = np.expand_dims(Y, axis=0)
            val = model.predict(X)
            print(val)
            if val < 50:
                return plt.xlabel("Car Severly Damaged", fontsize=30)
            elif val >= 50:
                return plt.xlabel("Car has Moderate or No damage", fontsize=30)

        fn = "static/upload/car_photo/"+str(claim_id)+ str("photo.jpeg")

        predictImage(fn)
        plt.savefig('static/output/damage_detect/'+str(claim_id)+ str("damage_detect.jpeg"))







        return render_template("claim_submitted.html",claim_id=claim_id,fraud_pred=fraud_pred)

    return render_template("claim_submitted.html")







@app.route('/Claim Tracking',methods=["POST","GET"])
def claim_tracking():


    if request.method == 'POST':
        claim_status_df = pd.read_csv("data/Claim Status.csv")
        claim_id=request.form.get("claim_id")
        track_status=claim_status_df[claim_status_df["Claim ID"]==claim_id].iloc[0,25]






        return render_template("claim_status.html",claim_id=claim_id,track_status=track_status)
    return render_template("claim_status.html")



@app.route('/Adjudication',methods=["POST","GET"])
def adjudication():
    global claim_status_df
    claim_status_df = pd.read_csv("data/Claim Status.csv")
    if request.method == 'POST':

        claims_open=pd.DataFrame(claim_status_df[claim_status_df.ClaimStatus=="Under Process"]["Claim ID"])








        return render_template("adjudication.html", tables = [claims_open.to_html(classes = 'data')], titles = claims_open.columns.values)
    return render_template("adjudication.html")

@app.route('/Adjudication Decision',methods=["POST","GET"])
def adjudication_decision():
    global claim_status_df
    global claim_adj_id
    claim_status_df = pd.read_csv("data/Claim Status.csv")
    if request.method == 'POST':
        claim_adj_id = request.form.get("claim_adj_id")

        claim_status_df_=pd.DataFrame(claim_status_df[claim_status_df["Claim ID"]==claim_adj_id])

        filename=claim_adj_id + 'damage_detect.jpeg'
        img = mpimg.imread('static/output/damage_detect/'+filename)
        img = plt.imshow(img)
        plt.savefig('static/img.jpeg')

        for i in range(0, len(claim_status_df.columns)):
            for j in range(0, len(claim_status_df)):
                if claim_status_df.iloc[j, i] == claim_adj_id:
                    claim_status_df.iloc[j, i + 1] = request.form.get("claim_decision")






        return render_template("adjudication_decision.html", tables = [claim_status_df_.to_html(classes = 'data')], titles = claim_status_df_.columns.values, claim_adj_id=claim_adj_id, filename=str(filename))
    return render_template("adjudication_decision.html")

@app.route('/Adjudication Decision Done',methods=["POST","GET"])
def adjudication_decision_done():
    global claim_status_df
    claim_status_df = pd.read_csv("data/Claim Status.csv")

    if request.method == 'POST':
        for i in range(0, len(claim_status_df.columns)):
            for j in range(0, len(claim_status_df)):
                if claim_status_df.iloc[j, i] == claim_adj_id:
                    claim_status_df.iloc[j, i + 1] = request.form.get("claim_decision")

        claim_status_df.to_csv(path_or_buf="data/Claim Status.csv", index=False)






        return render_template("test.html")
    return render_template("test.html")




@app.route('/Claim Status',methods=["POST","GET"])
def claim_status():

    if request.method == 'POST':





        return render_template("/")
    return render_template("/")




if __name__=='__main__':
    app.run(debug=True)