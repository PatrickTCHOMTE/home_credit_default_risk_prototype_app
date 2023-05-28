import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
import streamlit as st
import pickle

# Chargement du modèle KNeighborsClassifier sauvegardé
model = pickle.load(open('model.pkl', 'rb'))

# Le cache pour un chargement rapide du modèle
@st.cache


# Définition de la fonction de prédiction
def predict(EXT_SOURCE_3, EXT_SOURCE_2, PAYMENT_RATE, DAYS_ID_PUBLISH, DAYS_BIRTH, DAYS_EMPLOYED, DAYS_EMPLOYED_PERC, REGION_POPULATION_RELATIVE, DAYS_REGISTRATION, ANNUITY_INCOME_PERC, DAYS_LAST_PHONE_CHANGE, AMT_GOODS_PRICE, AMT_ANNUITY, INCOME_CREDIT_PERC, AMT_CREDIT, INCOME_PER_PERSON, AMT_INCOME_TOTAL, HOUR_APPR_PROCESS_START, CODE_GENDER, DEF_30_CNT_SOCIAL_CIRCLE, 
AMT_REQ_CREDIT_BUREAU_YEAR, NAME_INCOME_TYPE_Working, OBS_60_CNT_SOCIAL_CIRCLE, OBS_30_CNT_SOCIAL_CIRCLE, WEEKDAY_APPR_PROCESS_START_TUESDAY, WEEKDAY_APPR_PROCESS_START_FRIDAY, REGION_RATING_CLIENT, 
REGION_RATING_CLIENT_W_CITY, CNT_CHILDREN, NAME_TYPE_SUITE_Unaccompanied):
    
    columns_df = [EXT_SOURCE_3, EXT_SOURCE_2, PAYMENT_RATE, DAYS_ID_PUBLISH, DAYS_BIRTH, DAYS_EMPLOYED, DAYS_EMPLOYED_PERC, REGION_POPULATION_RELATIVE, DAYS_REGISTRATION, ANNUITY_INCOME_PERC, DAYS_LAST_PHONE_CHANGE, AMT_GOODS_PRICE, AMT_ANNUITY, INCOME_CREDIT_PERC, AMT_CREDIT, INCOME_PER_PERSON, AMT_INCOME_TOTAL, HOUR_APPR_PROCESS_START, CODE_GENDER, DEF_30_CNT_SOCIAL_CIRCLE, 
AMT_REQ_CREDIT_BUREAU_YEAR, NAME_INCOME_TYPE_Working, OBS_60_CNT_SOCIAL_CIRCLE, OBS_30_CNT_SOCIAL_CIRCLE, WEEKDAY_APPR_PROCESS_START_TUESDAY, WEEKDAY_APPR_PROCESS_START_FRIDAY, REGION_RATING_CLIENT, 
REGION_RATING_CLIENT_W_CITY, CNT_CHILDREN, NAME_TYPE_SUITE_Unaccompanied]
    
    data = pd.DataFrame([columns_df], columns=columns_df)
    
    prediction = model.predict(data)
    score = model.predict_proba(data)
    
    return prediction[0], score[0] 
    


st.title("Loan's Attribution Predictor")
st.image("Bank Loan.jpg")
st.header("Enter client's data: ")
EXT_SOURCE_3 = st.number_input('Normalized score from external data source 3:', min_value=0.000527, max_value=0.885488, value=0.499630)
EXT_SOURCE_2 = st.number_input('Normalized score from external data source 2:', min_value=0.000476, max_value=0.855000, value=0.521936)
PAYMENT_RATE = st.number_input('Payment rate:', min_value=0.025278, max_value=0.124429, value=0.052869)
DAYS_ID_PUBLISH = st.number_input('How many days before the application did client change the identity document with which he applied for the loan:', min_value=-6228.000000, max_value=0.000000, value=-2864.390808)
DAYS_BIRTH = st.number_input("Client's age in days at the time of application:", min_value=-25075.000000, max_value=-7742.000000, value=-14888.709785)
DAYS_EMPLOYED = st.number_input('How many days before the application the person started current employment:', min_value=-15632.000000, max_value=-17.000000	, value=-2479.178664)
DAYS_EMPLOYED_PERC = st.number_input('Percentage of days before the application the person started current employment:', min_value=0.001032, max_value=0.695770, value=0.162370)
REGION_POPULATION_RELATIVE = st.number_input('Normalized population of region where client lives:', min_value=0.000938, max_value=0.072508, value=0.020819)
DAYS_REGISTRATION = st.number_input('How many days before the application did client change his registration:', min_value=-20981.000000, max_value=0.000000, value=-4649.857556)
ANNUITY_INCOME_PERC = st.number_input('Annuity income percentage:', min_value=0.008333, max_value=0.818433, value=0.176294)
DAYS_LAST_PHONE_CHANGE = st.number_input('How many days before application did client change phone:', min_value=-3856.000000, max_value=0.000000, value=-1002.629432)
AMT_GOODS_PRICE = st.number_input('For consumer loans it is the price of the goods for which the loan is given:', min_value=4.500000e+04, max_value=2.925000e+06, value=5.605670e+05)
AMT_ANNUITY = st.number_input('Loan annuity:', min_value=2844.000000, max_value=135936.000000, value=27987.123725)
INCOME_CREDIT_PERC = st.number_input('Income credit percentage:', min_value=0.0400000, max_value=6.000000, value=0.401858)
AMT_CREDIT = st.number_input('Credit amount of the loan:', min_value=4.500000e+04, max_value=2.925000e+06, value=6.241625e+05)
INCOME_PER_PERSON = st.number_input('Income per person:', min_value=9.000000e+03, max_value=1.035000e+06, value=9.361403e+04)
AMT_INCOME_TOTAL = st.number_input('Income of the client:', min_value=2.700000e+04, max_value=1.350000e+06, value=1.773478e+05)
HOUR_APPR_PROCESS_START = st.number_input('Approximately at what hour did the client apply for the loan:', min_value=1.000000, max_value=23.000000, value=12.252169)
CODE_GENDER = st.number_input('Gender of the client:', min_value=0.000000, max_value=1.000000, value=0.629280)
DEF_30_CNT_SOCIAL_CIRCLE = st.number_input("How many observation of client's social surroundings defaulted on 30 DPD (days past due):", min_value=0.000000, max_value=5.000000, value=0.145640)
AMT_REQ_CREDIT_BUREAU_YEAR = st.number_input('Number of enquiries to Credit Bureau about the client one day year (excluding last 3 months before application):', min_value=0.000000, max_value=11.000000, value=1.846294)
NAME_INCOME_TYPE_Working = st.number_input('Name income type working:', min_value=0.000000, max_value=1.000000, value=0.642368)
OBS_60_CNT_SOCIAL_CIRCLE = st.number_input("How many observation of client's social surroundings with observable 60 DPD (days past due) default:", min_value=0.000000, max_value=25.000000, value=1.416223)
OBS_30_CNT_SOCIAL_CIRCLE = st.number_input("How many observation of client's social surroundings with observable 30 DPD (days past due) default:", min_value=0.000000, max_value=25.000000, value=1.432354)
WEEKDAY_APPR_PROCESS_START_TUESDAY = st.number_input('Weekday appr process start tuesday:', min_value=0.000000, max_value=1.000000, value=0.166793)
WEEKDAY_APPR_PROCESS_START_FRIDAY = st.number_input('Weekday appr process start friday:', min_value=0.000000, max_value=1.000000, value=0.166641)
REGION_RATING_CLIENT = st.number_input('Our rating of the region where client lives (1,2,3):', min_value=1.000000, max_value=3.000000, value=2.041394)
REGION_RATING_CLIENT_W_CITY = st.number_input('Our rating of the region where client lives with taking city into account (1,2,3):', min_value=1.000000, max_value=3.000000, value=2.019480)
CNT_CHILDREN = st.number_input('Number of children the client has:', min_value=0.000000, max_value=7.000000, value=1)
NAME_TYPE_SUITE_Unaccompanied = st.number_input('Name type suite unaccompanied:', min_value=0.000000, max_value=1.000000, value=0.818140)

if st.button('Predict Loan'):
    response_predicted, customer_score = predict(EXT_SOURCE_3, EXT_SOURCE_2, PAYMENT_RATE, DAYS_ID_PUBLISH, DAYS_BIRTH, DAYS_EMPLOYED, DAYS_EMPLOYED_PERC, REGION_POPULATION_RELATIVE, DAYS_REGISTRATION, ANNUITY_INCOME_PERC, DAYS_LAST_PHONE_CHANGE, AMT_GOODS_PRICE, AMT_ANNUITY, INCOME_CREDIT_PERC, AMT_CREDIT, INCOME_PER_PERSON, AMT_INCOME_TOTAL, HOUR_APPR_PROCESS_START, CODE_GENDER, DEF_30_CNT_SOCIAL_CIRCLE, 
AMT_REQ_CREDIT_BUREAU_YEAR, NAME_INCOME_TYPE_Working, OBS_60_CNT_SOCIAL_CIRCLE, OBS_30_CNT_SOCIAL_CIRCLE, WEEKDAY_APPR_PROCESS_START_TUESDAY, WEEKDAY_APPR_PROCESS_START_FRIDAY, REGION_RATING_CLIENT, 
REGION_RATING_CLIENT_W_CITY, CNT_CHILDREN, NAME_TYPE_SUITE_Unaccompanied)

    proba_loan_is_granted = customer_score[0]

    if proba_loan_is_granted >=  0.75:
        st.success('Loan granted !', icon="✅")
    else:
        st.error('Loan not granted :-)', icon="🚨")
            
            
    st.write("The loan's score of client: ", proba_loan_is_granted)
    st.write('Number of children the client has : ', CNT_CHILDREN)
    st.image("Number of children by clients who have obtained a loan.png")
    st.write('Rate of the region where client lives (1,2,3) : ', REGION_RATING_CLIENT)
    st.image("Regional rating of clients who have obtained.png")

















