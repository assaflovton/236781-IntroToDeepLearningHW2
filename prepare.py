import numpy as np
import pandas as pd


def preprare_data(data, training_data):
    data_res = data.copy()
    data_res = pd.concat([data_res, pd.get_dummies(data_res['blood_type'], prefix="blood_type")], axis=1)
    cough, fever, headache, low_appetite, shortness_of_breath = [0] * data_res.shape[0], [0] * data_res.shape[0], [0] * data_res.shape[
        0], [0] * data_res.shape[0], [0] * data_res.shape[0]
    for i, row in enumerate(data['symptoms']):
        if row == np.NaN:  # in case that it is None
            continue
        if 'cough' in str(row):
            cough[i] = 1
        if 'fever' in str(row):
            fever[i] = 1
        if 'headache' in str(row):
            headache[i] = 1
        if 'low_appetite' in str(row):
            low_appetite[i] = 1
        if 'shortness_of_breath' in str(row):
            shortness_of_breath[i] = 1
    # we decided to not add other symptoms
    data_res['symptoms_cough'] = cough
    data_res['symptoms_fever'] = fever
    data_res['symptoms_shortness_of_breath'] = shortness_of_breath
    PCR = ['PCR_01', 'PCR_02', 'PCR_03', 'PCR_06' , 'PCR_08','PCR_10']

    continuous = ['PCR_01', 'PCR_02', 'PCR_03', 'PCR_06', 'PCR_08', 'PCR_10','household_income', 'sugar_levels','weight']
    categorial = ['num_of_siblings', 'sex']

    #drop unwanted features
    drop_list = ['patient_id', 'age','blood_type','blood_type_B+','blood_type_AB+','blood_type_B-','blood_type_O+','blood_type_O-'
                 ,'blood_type_AB-','blood_type_A-','address','current_location','job','happiness_score','pcr_date',
                 'symptoms','sport_activity','PCR_04','PCR_05','PCR_09','conversations_per_day']
    data_res.drop(columns=drop_list,inplace=True)

    # treat outiers
    for i, column in enumerate(PCR, 1):
        upper_limit = training_data[column].quantile(0.97)
        lower_limit = training_data[column].quantile(0.03)
        data_res.loc[(data[column] > upper_limit), column] = upper_limit
        data_res.loc[(data[column] < lower_limit), column] = lower_limit

    #fill misssing values
    for c in continuous:
        data_res[c].fillna((training_data[c].mean()), inplace=True)
    for c in categorial:
        data_res.fillna(training_data.mode().iloc[0], inplace=True)
    data_res['covid'] = data_res['covid'].replace({True: 1, False: -1})
    data_res['spread'] = data_res['spread'].replace({'High': 1, 'Low': -1})
    data_res['risk'] = data_res['risk'].replace({'High': 1, 'Low': -1})
    data_res['sex'] = data_res['sex'].replace({'M': 1, 'F': 0})
    #normalize
    for c in continuous:
        data_res[c] = (data_res[c]-data_res[c].mean())/data_res[c].std()
    for c in categorial:
        data_res[c] = (data_res[c]-data_res[c].min())/(data_res[c].max()-data_res[c].min())
    return data_res
