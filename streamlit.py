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
    
    return prediction[0]


st.title("Loan's Attribution Predictor")
st.image("Bank Loan.jpg")
st.header("Entrez les données: ")
EXT_SOURCE_3 = st.number_input('EXT_SOURCE_3:', min_value=0.000527, max_value=0.885488, value=0.499630)
EXT_SOURCE_2 = st.number_input('EXT_SOURCE_2:', min_value=0.000476, max_value=0.855000, value=0.521936)
PAYMENT_RATE = st.number_input('PAYMENT_RATE:', min_value=0.025278, max_value=0.124429, value=0.052869)
DAYS_ID_PUBLISH = st.number_input('DAYS_ID_PUBLISH:', min_value=-6228.000000, max_value=0.000000, value=-2864.390808)
DAYS_BIRTH = st.number_input('DAYS_BIRTH:', min_value=-25075.000000, max_value=-7742.000000, value=-14888.709785)
DAYS_EMPLOYED = st.number_input('DAYS_EMPLOYED:', min_value=-15632.000000, max_value=-17.000000	, value=-2479.178664)
DAYS_EMPLOYED_PERC = st.number_input('DAYS_EMPLOYED_PERC:', min_value=0.001032, max_value=0.695770, value=0.162370)
REGION_POPULATION_RELATIVE = st.number_input('REGION_POPULATION_RELATIVE:', min_value=0.000938, max_value=0.072508, value=0.020819)
DAYS_REGISTRATION = st.number_input('DAYS_REGISTRATION:', min_value=-20981.000000, max_value=0.000000, value=-4649.857556)
ANNUITY_INCOME_PERC = st.number_input('ANNUITY_INCOME_PERC:', min_value=0.008333, max_value=0.818433, value=0.176294)
DAYS_LAST_PHONE_CHANGE = st.number_input('DAYS_LAST_PHONE_CHANGE:', min_value=-3856.000000, max_value=0.000000, value=-1002.629432)
AMT_GOODS_PRICE = st.number_input('AMT_GOODS_PRICE:', min_value=4.500000e+04, max_value=2.925000e+06, value=5.605670e+05)
AMT_ANNUITY = st.number_input('AMT_ANNUITY:', min_value=2844.000000, max_value=135936.000000, value=27987.123725)
INCOME_CREDIT_PERC = st.number_input('INCOME_CREDIT_PERC:', min_value=0.0400000, max_value=6.000000, value=0.401858)
AMT_CREDIT = st.number_input('AMT_CREDIT:', min_value=4.500000e+04, max_value=2.925000e+06, value=6.241625e+05)
INCOME_PER_PERSON = st.number_input('INCOME_PER_PERSON:', min_value=9.000000e+03, max_value=1.035000e+06, value=9.361403e+04)
AMT_INCOME_TOTAL = st.number_input('AMT_INCOME_TOTAL:', min_value=2.700000e+04, max_value=1.350000e+06, value=1.773478e+05)
HOUR_APPR_PROCESS_START = st.number_input('HOUR_APPR_PROCESS_START:', min_value=1.000000, max_value=23.000000, value=12.252169)
CODE_GENDER = st.number_input('CODE_GENDER:', min_value=0.000000, max_value=1.000000, value=0.629280)
DEF_30_CNT_SOCIAL_CIRCLE = st.number_input('DEF_30_CNT_SOCIAL_CIRCLE:', min_value=0.000000, max_value=5.000000, value=0.145640)
AMT_REQ_CREDIT_BUREAU_YEAR = st.number_input('AMT_REQ_CREDIT_BUREAU_YEAR:', min_value=0.000000, max_value=11.000000, value=1.846294)
NAME_INCOME_TYPE_Working = st.number_input('NAME_INCOME_TYPE_Working:', min_value=0.000000, max_value=1.000000, value=0.642368)
OBS_60_CNT_SOCIAL_CIRCLE = st.number_input('OBS_60_CNT_SOCIAL_CIRCLE:', min_value=0.000000, max_value=25.000000, value=1.416223)
OBS_30_CNT_SOCIAL_CIRCLE = st.number_input('OBS_30_CNT_SOCIAL_CIRCLE:', min_value=0.000000, max_value=25.000000, value=1.432354)
WEEKDAY_APPR_PROCESS_START_TUESDAY = st.number_input('WEEKDAY_APPR_PROCESS_START_TUESDAY:', min_value=0.000000, max_value=1.000000, value=0.166793)
WEEKDAY_APPR_PROCESS_START_FRIDAY = st.number_input('WEEKDAY_APPR_PROCESS_START_FRIDAY:', min_value=0.000000, max_value=1.000000, value=0.166641)
REGION_RATING_CLIENT = st.number_input('REGION_RATING_CLIENT:', min_value=1.000000, max_value=3.000000, value=2.041394)
REGION_RATING_CLIENT_W_CITY = st.number_input('REGION_RATING_CLIENT_W_CITY:', min_value=1.000000, max_value=3.000000, value=2.019480)
CNT_CHILDREN = st.number_input('CNT_CHILDREN:', min_value=0.000000, max_value=7.000000, value=0.508751)
NAME_TYPE_SUITE_Unaccompanied = st.number_input('NAME_TYPE_SUITE_Unaccompanied:', min_value=0.000000, max_value=1.000000, value=0.818140)

if st.button('Predict Loan'):
    response_predicted = predict(EXT_SOURCE_3, EXT_SOURCE_2, PAYMENT_RATE, DAYS_ID_PUBLISH, DAYS_BIRTH, DAYS_EMPLOYED, DAYS_EMPLOYED_PERC, REGION_POPULATION_RELATIVE, DAYS_REGISTRATION, ANNUITY_INCOME_PERC, DAYS_LAST_PHONE_CHANGE, AMT_GOODS_PRICE, AMT_ANNUITY, INCOME_CREDIT_PERC, AMT_CREDIT, INCOME_PER_PERSON, AMT_INCOME_TOTAL, HOUR_APPR_PROCESS_START, CODE_GENDER, DEF_30_CNT_SOCIAL_CIRCLE, 
AMT_REQ_CREDIT_BUREAU_YEAR, NAME_INCOME_TYPE_Working, OBS_60_CNT_SOCIAL_CIRCLE, OBS_30_CNT_SOCIAL_CIRCLE, WEEKDAY_APPR_PROCESS_START_TUESDAY, WEEKDAY_APPR_PROCESS_START_FRIDAY, REGION_RATING_CLIENT, 
REGION_RATING_CLIENT_W_CITY, CNT_CHILDREN, NAME_TYPE_SUITE_Unaccompanied)
    
    if response_predicted == 1:
        st.succes('Loan granted !', icon="✅")
    else:
        st.error('Loan not granted :-)', icon="🚨")

















