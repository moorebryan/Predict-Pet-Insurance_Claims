import pandas as pd
from flask import Flask, render_template, request
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pickle

# For this notebook treat inf as nan for convenience
pd.set_option('use_inf_as_na', True)

app = Flask(__name__)

#label Data folder for use on Cloud
DATA_FOLDER='static/data'
app.config['DATA_FOLDER']=DATA_FOLDER
ALLOWED_EXTENSIONS = set(['csv'])
svr_dog = pickle.load(open('static/data/svr_dog.pickle','rb'))
svr_cat = pickle.load(open('static/data/svr_cat.pickle','rb'))    

def clean_up_df(df1,df2, pred_date_end, pred_date_begin):
    df_joint=pd.merge(df1,df2,on='PetId', how='outer')
    df_joint['ClaimAmount'].fillna(0, inplace=True)
    df_joint['Count']=1
    df_joint['EnrollDate'] =  pd.to_datetime(df_joint['EnrollDate'], infer_datetime_format=True)
    df_joint['CancelDate'] =  pd.to_datetime(df_joint['CancelDate'], infer_datetime_format=True)
    df_joint['ClaimDate'] =  pd.to_datetime(df_joint['ClaimDate'], infer_datetime_format=True)
    df_joint['Claim_Year']= pd.DatetimeIndex(df_joint['ClaimDate']).year
    df_joint['Claim_Month']= pd.DatetimeIndex(df_joint['ClaimDate']).month
    
    replace_dict = {'0-7 weeks old':1, '8 weeks to 12 months old':6, '1 year old':12, '2 years old':24, \
                '3 years old':36, '4 years old':48, '5 years old':60, '6 years old':72, '7 years old':84, \
                '8 years old':96, '9 years old':108, '10 years old':120, '11 years old':132, \
                '12 years old':144, '13 years old':156}

    df_joint['AgeAtEnroll'] = df_joint['AgeAtEnroll'].map(replace_dict)
    df_joint['Age_At_Claim'] = ((df_joint['ClaimDate'] - df_joint['EnrollDate'])/np.timedelta64(1, 'M') + \
                            df_joint['AgeAtEnroll'])

    # Note pets with Age_At_Claim of NaN mean they never actually made a claim
    # We therefore lower their Count to  0
    df_joint.loc[np.isnan(df_joint['Age_At_Claim']), ['Count']] = 0
    # We also get NaN if the pet joined in the same month
    # To address this I will manually set these to zero
    df_joint.loc[np.isnan(df_joint['Age_At_Claim']), ['Age_At_Claim']] = 0

    df_joint['Time_Since_Enrollment'] = ((df_joint['ClaimDate'] - df_joint['EnrollDate'])/np.timedelta64(1, 'M'))

    # Replace Time_Since_Enrollment with CancelDate if they never placed a claim
    df_joint.loc[np.isnan(df_joint['Time_Since_Enrollment']), ['Time_Since_Enrollment']] = (df_joint['CancelDate'] - df_joint['EnrollDate'])/np.timedelta64(1, 'M')
    # If still NaN replace with chosen pred month as this means they have not cancelled (use 1st day of following month for calculation)
    df_joint.loc[np.isnat(df_joint['CancelDate']), ['Time_Since_Enrollment']] = (pred_date_end - df_joint['EnrollDate'])/np.timedelta64(1, 'M')
    
    df_joint['Previous_Costs'] = df_joint.sort_values(by  = 'ClaimDate').groupby(df_joint['PetId'])['ClaimAmount'].cumsum()
    # As this method counts the cost incurred on that day as well I will manually remove this
    df_joint['Previous_Costs'] = df_joint['Previous_Costs'] - df_joint['ClaimAmount']
    
    df_joint['Num_Claims_To_Date'] = df_joint.sort_values(by  = 'ClaimDate')['Count'].groupby(df_joint['PetId']).cumsum()
    # Find  total number of claims made for each pet
    df_total = df_joint.pivot_table(index=['PetId'],values=['Count'], aggfunc=(np.sum))
    df_total.reset_index(inplace=True)

    # Rename column appropriately
    df_total.rename(columns={'Count': 'Total_Num_Claims'}, inplace=True)

    # Combine this with our main  table
    # To prevent re-running errors drop 'Total_Num_Claims' column if it already exists in df_joint
    df_joint.drop(['Total_Num_Claims'], axis=1, inplace=True, errors='ignore')

    # Join total number of claims to main table
    df_joint = pd.merge(df_joint, df_total, on='PetId', how='outer')
    df_joint['Norm_Claims_To_Date'] = (df_joint['Num_Claims_To_Date']-1) / df_joint['Total_Num_Claims']

    # Set to zero to correct for NaN calculations for no claim filed
    df_joint.loc[np.isnan(df_joint['Norm_Claims_To_Date']), ['Norm_Claims_To_Date']] = 0
    # Do same if inf
    df_joint.loc[~np.isfinite(df_joint['Norm_Claims_To_Date']), ['Norm_Claims_To_Date']] = 0
    
    df_joint['Claims_Per_Month']  = df_joint['Total_Num_Claims'] / df_joint['Time_Since_Enrollment']

    # Set to zero to correct for NaN calculations for no claim filed
    df_joint.loc[np.isnan(df_joint['Claims_Per_Month']), ['Claims_Per_Month']] = 0
    # Do same for divide by zero error
    df_joint.loc[~np.isfinite(df_joint['Claims_Per_Month']), ['Claims_Per_Month']] = 0
    
    df_joint['Time_Since_Last_Claim'] = df_joint.sort_values(by  = 'ClaimDate').groupby(df_joint['PetId'])['Age_At_Claim'].diff(periods=1)

    # Note that  with above code the first claim will have NaN
    # Address by setting that time as period between that claim and the enroll date
    df_joint['Time_Since_Last_Claim'].fillna(df_joint['Time_Since_Enrollment'], inplace=True)
    
    df_joint['Avg_Claim_Per_Month'] = df_joint['Previous_Costs'] / (df_joint['Time_Since_Enrollment'])

    # Set to zero to correct for NaN calculations for no claim filed
    df_joint.loc[np.isnan(df_joint['Avg_Claim_Per_Month']), ['Avg_Claim_Per_Month']] = 0
    # Address divide by zero error as well if they just joined
    df_joint.loc[~np.isfinite(df_joint['Avg_Claim_Per_Month']), ['Avg_Claim_Per_Month']] = 0
    
    df_dog = df_joint.loc[df_joint['Species'] == 'Dog']
    df_cat = df_joint.loc[df_joint['Species'] == 'Cat']
    df_dog = pd.concat([df_dog,pd.get_dummies(df_dog.Breed)],axis=1)
    df_cat = pd.concat([df_cat,pd.get_dummies(df_cat.Breed)],axis=1)
        # Drop these features
    df_dog.drop(['Time_Since_Enrollment', 'Num_Claims_To_Date', 'ClaimId','Count', \
                 'Claim_Month','Claim_Year', 'Norm_Claims_To_Date', 'Time_Since_Last_Claim'], axis=1, inplace=True, errors='ignore')

    df_cat.drop(['Time_Since_Enrollment', 'Num_Claims_To_Date','ClaimId','Count', \
             'Claim_Month','Claim_Year', 'Norm_Claims_To_Date', 'Time_Since_Last_Claim'], axis=1, inplace=True, errors='ignore')
    df_dog_1 = df_dog.copy(deep=True)
    df_cat_1 = df_cat.copy(deep=True)
    # Create table with max_Previous_Costs for each pet (excluding dates greater than chosen month)
    # As we do not want to count any rows greater than chosen month as a claim we manually find those and set agg to 0
    df_dog_1.loc[df_dog_1['ClaimDate']>=pred_date_begin, ['Previous_Costs']] = 0
    df_dog_1.loc[df_dog_1['ClaimDate']>=pred_date_begin, ['ClaimAmount']] = 0

    df_cat_1.loc[df_cat_1['ClaimDate']>=pred_date_begin, ['Previous_Costs']] = 0
    df_cat_1.loc[df_cat_1['ClaimDate']>=pred_date_begin, ['ClaimAmount']] = 0

    # Now grab row corresponding to the maximum remaining Previous_Costs, noting we have removed any after chosen month
    df_dog_max_claim = df_dog_1.pivot_table(index=['PetId'],values=['Previous_Costs'], aggfunc=(np.max))
    df_dog_max_claim.reset_index(inplace=True)

    df_cat_max_claim = df_cat_1.pivot_table(index=['PetId'],values=['Previous_Costs'], aggfunc=(np.max))
    df_cat_max_claim.reset_index(inplace=True)

    # Now do inner join to only keep rows corresponding to last claim information for each pet
    df_dog_Pred=pd.merge(df_dog,df_dog_max_claim,on=['PetId','Previous_Costs'], how='inner')
    df_cat_Pred=pd.merge(df_cat,df_cat_max_claim,on=['PetId','Previous_Costs'], how='inner')

    # Calculate estimated age of each pet in chosen month (Use 1st day of following month)
    df_dog_Pred['Age_At_Claim_Pred'] = ((pred_date_end - df_dog_Pred['EnrollDate'])/np.timedelta64(1, 'M') + df_dog_Pred['AgeAtEnroll']).astype(int)
    df_cat_Pred['Age_At_Claim_Pred'] = ((pred_date_end - df_cat_Pred['EnrollDate'])/np.timedelta64(1, 'M') + df_cat_Pred['AgeAtEnroll']).astype(int)

    # Calculate time spent as member through end of chosen month
    df_dog_Pred['Time_Since_Enrollment_Pred'] = ((pred_date_end - df_dog_Pred['EnrollDate'])/np.timedelta64(1, 'M')).astype(int)
    df_cat_Pred['Time_Since_Enrollment_Pred'] = ((pred_date_end - df_cat_Pred['EnrollDate'])/np.timedelta64(1, 'M')).astype(int)

    # Note that if this value is negative it means the pet joined after the prediction period
    # We must thus remove them from our dataframes as predictions do not make sense
    df_dog_Pred.drop(df_dog_Pred[df_dog_Pred['Time_Since_Enrollment_Pred'] < 0].index, inplace=True)
    df_cat_Pred.drop(df_cat_Pred[df_cat_Pred['Time_Since_Enrollment_Pred'] < 0].index, inplace=True)

    # Assuming no more claims have been made, calculate Avg_Claim_Per_Month
    df_dog_Pred['Claims_Per_Month_Pred'] = df_dog_Pred['Total_Num_Claims'] / df_dog_Pred['Time_Since_Enrollment_Pred']
    df_cat_Pred['Claims_Per_Month_Pred'] = df_cat_Pred['Total_Num_Claims'] / df_cat_Pred['Time_Since_Enrollment_Pred']
    # For this we get NaN if the pet joined in the same month
    # To address this I will manually set these to zeroo
    df_dog_Pred.loc[np.isnan(df_dog_Pred['Claims_Per_Month_Pred']), ['Claims_Per_Month_Pred']] = 0
    df_cat_Pred.loc[np.isnan(df_cat_Pred['Claims_Per_Month_Pred']), ['Claims_Per_Month_Pred']] = 0
    # Do same for infinite values
    df_dog_Pred.loc[~np.isfinite(df_dog_Pred['Claims_Per_Month_Pred']), ['Claims_Per_Month_Pred']] = 0
    df_cat_Pred.loc[~np.isfinite(df_cat_Pred['Claims_Per_Month_Pred']), ['Claims_Per_Month_Pred']] = 0

    # Calculate previous costs for each pet (adding in previous claim amount)
    df_dog_Pred['Previous_Costs_Pred'] = df_dog_Pred['Previous_Costs'] + df_dog_Pred['ClaimAmount']
    df_cat_Pred['Previous_Costs_Pred'] = df_cat_Pred['Previous_Costs'] + df_cat_Pred['ClaimAmount']

    # Calculate average cost per month
    df_dog_Pred['Avg_Claim_Per_Month_Pred'] = df_dog_Pred['Previous_Costs_Pred'] / (df_dog_Pred['Time_Since_Enrollment_Pred'])
    df_cat_Pred['Avg_Claim_Per_Month_Pred'] = df_cat_Pred['Previous_Costs_Pred'] / (df_cat_Pred['Time_Since_Enrollment_Pred'])
    # Set to zero to correct for NaN calculations for no claim filed
    df_dog_Pred.loc[np.isnan(df_dog_Pred['Avg_Claim_Per_Month_Pred']), ['Avg_Claim_Per_Month_Pred']] = 0
    df_cat_Pred.loc[np.isnan(df_cat_Pred['Avg_Claim_Per_Month_Pred']), ['Avg_Claim_Per_Month_Pred']] = 0
    # Do same for infinite values
    df_dog_Pred.loc[~np.isfinite(df_dog_Pred['Avg_Claim_Per_Month_Pred']), ['Avg_Claim_Per_Month_Pred']] = 0
    df_cat_Pred.loc[~np.isfinite(df_cat_Pred['Avg_Claim_Per_Month_Pred']), ['Avg_Claim_Per_Month']]=0
    
    col_order_dog=['PetId', 'Species', 'Breed', 'CancelDate','Age_At_Claim_Pred','Previous_Costs_Pred','Claims_Per_Month_Pred', \
           'Avg_Claim_Per_Month_Pred','Chihuahua','French Bulldog','Golden Retriever','Great Dane','Mixed Breed']
    col_order_cat=['PetId', 'Species', 'Breed', 'CancelDate','Age_At_Claim_Pred','Previous_Costs_Pred','Claims_Per_Month_Pred', \
           'Avg_Claim_Per_Month_Pred','Mixed Breed','Ragdoll']
    # Reorder to match trained models
    df_dog_Pred = df_dog_Pred[col_order_dog]
    df_cat_Pred = df_cat_Pred[col_order_cat]
    


    return df_cat_Pred,df_dog_Pred



          
   
@app.route('/')
def homepage():
    return render_template('homepage.html')


    


@app.route('/select', methods=['POST','GET']) 
def submitpage():
  # get form submit data
    global df_cat_cleaned,df_dog_cleaned
    return render_template('select.html')
                           
@app.route('/results', methods=['POST','GET']) 
def resultspage():
    global df_cat_cleaned,df_dog_cleaned, df_pet, df_claims, pred_date_end, pred_date_begin, df_chsn, df_all_pred
    
    scaler = MinMaxScaler()
    
    # First check if new csv files have been uploaded
    try:
        file_pet = request.files['pet_data']
        file_claims=request.files['claims_data']
        #pred_date = request.form["select_date"]
        pred_date = '2019-07-12'
        
        # Turn into a pandas datetime object
        pred_date = pd.to_datetime(pred_date, infer_datetime_format=True)
        
        # Calculate first day of this month
        pred_date_begin = pred_date - pd.offsets.MonthBegin(1)
        # Calculate 1st day of next month
        pred_date_end = pred_date_begin + pd.offsets.MonthBegin(1)
    
        #df_pet=load_data_file(file_to_localfile(file_pet))
        df_pet = pd.read_csv(file_pet)
        #print('Past')
        #df_claims=load_data_file(file_to_localfile(file_claims))
        df_claims = pd.read_csv(file_claims)
        
        # Set chosen pet ID to None as not choice made yet
        chosen_pet_id = None
        
        df_cat_cleaned,df_dog_cleaned=clean_up_df(df_pet,df_claims, pred_date_end, pred_date_begin)
        
        # Make predictions for dogs
        dog_feature = scaler.fit_transform(df_dog_cleaned.drop(['PetId','CancelDate', 'Species', 'Breed'], axis=1).values)
        pred_dog = svr_dog.predict(dog_feature)
        df_dog_cleaned_2 = df_dog_cleaned[['PetId','CancelDate', 'Species', 'Breed', 'Age_At_Claim_Pred']]
        df_dog_cleaned_2['Pred_Cost_Chsn_Month'] = pred_dog.tolist()
        
        # Make predictions for cats
        cat_feature = scaler.fit_transform(df_cat_cleaned.drop(['PetId','CancelDate', 'Species', 'Breed'], axis=1).values)
        pred_cat = svr_cat.predict(cat_feature)
        df_cat_cleaned_2 = df_cat_cleaned[['PetId','CancelDate', 'Species', 'Breed', 'Age_At_Claim_Pred']]
        df_cat_cleaned_2['Pred_Cost_Chsn_Month'] = pred_cat.tolist()
        
        # Now merge these two so we have all predictions in one dataframe
        df_all_pred = df_dog_cleaned_2.append(df_cat_cleaned_2, ignore_index=True)
        # Put in order of PetID again
        df_all_pred.sort_values(by='PetId', inplace=True)
        
        # Round predicted claim amount (if claim made)
        df_all_pred['Pred_Cost_Chsn_Month'] = df_all_pred['Pred_Cost_Chsn_Month'].round(2)
        
        df_all_pred.rename(columns={'Age_At_Claim_Pred': 'Age at Pred (Months)'}, inplace=True)
        
        # If someone has already cancelled prior to this date then the predicted cost is obviously NaN
        df_all_pred.loc[df_all_pred['CancelDate'] < pred_date_begin, ['Pred_Cost_Chsn_Month']] = 'Already Cancelled'
        df_all_pred.loc[df_all_pred['CancelDate'] < pred_date_begin, ['Age at Pred (Months)']] = 'Already Cancelled'
        
        df_all_pred.drop(['CancelDate'], axis=1, inplace=True)
                
        
    except:
        # If not csv files were uploaded see if a new PetID has been provided
        try:
            chosen_pet_id = request.form.get('PetID')
            # Manually raise error if nothing entered so we go to except
            if chosen_pet_id == None:
                raise ValueError('No PetID has been entered')
            
        except:
            # If neither is the case then load the csv files as submit button was pressed with nothing in it
            pred_date = '2019-07-12'
            # Turn into a pandas datetime object
            pred_date = pd.to_datetime(pred_date, infer_datetime_format=True)
            
            # Calculate first day of this month
            pred_date_begin = pred_date - pd.offsets.MonthBegin(1)
            # Calculate 1st day of next month
            pred_date_end = pred_date_begin + pd.offsets.MonthBegin(1)
            
            # Read from uploaded files
            df_claims = pd.read_csv('static/data/claimdata.csv')
            df_pet = pd.read_csv('static/data/petdata.csv')
            
            df_cat_cleaned,df_dog_cleaned=clean_up_df(df_pet,df_claims, pred_date_end, pred_date_begin)
    
            # Make predictions for dogs
            dog_feature = scaler.fit_transform(df_dog_cleaned.drop(['PetId','CancelDate', 'Species', 'Breed'], axis=1).values)
            pred_dog = svr_dog.predict(dog_feature)
            df_dog_cleaned_2 = df_dog_cleaned[['PetId','CancelDate', 'Species', 'Breed', 'Age_At_Claim_Pred']]
            df_dog_cleaned_2['Pred_Cost_Chsn_Month'] = pred_dog.tolist()
            
            # Make predictions for cats
            cat_feature = scaler.fit_transform(df_cat_cleaned.drop(['PetId','CancelDate', 'Species', 'Breed'], axis=1).values)
            pred_cat = svr_cat.predict(cat_feature)
            df_cat_cleaned_2 = df_cat_cleaned[['PetId','CancelDate', 'Species', 'Breed', 'Age_At_Claim_Pred']]
            df_cat_cleaned_2['Pred_Cost_Chsn_Month'] = pred_cat.tolist()
            
            # Now merge these two so we have all predictions in one dataframe
            df_all_pred = df_dog_cleaned_2.append(df_cat_cleaned_2, ignore_index=True)
            # Put in order of PetID again
            df_all_pred.sort_values(by='PetId', inplace=True)
            
            # Round predicted claim amount (if claim made)
            df_all_pred['Pred_Cost_Chsn_Month'] = df_all_pred['Pred_Cost_Chsn_Month'].round(2)
            
            df_all_pred.rename(columns={'Age_At_Claim_Pred': 'Age at Pred (Months)'}, inplace=True)
            
            # If someone has already cancelled prior to this date then the predicted cost is obviously NaN
            df_all_pred.loc[df_all_pred['CancelDate'] < pred_date_begin, ['Pred_Cost_Chsn_Month']] = 'Already Cancelled'
            df_all_pred.loc[df_all_pred['CancelDate'] < pred_date_begin, ['Age at Pred (Months)']] = 'Already Cancelled'
            
            df_all_pred.drop(['CancelDate'], axis=1, inplace=True)
                
    if chosen_pet_id == None:
        chosen_pet_id = 2
    else:
        chosen_pet_id = int(chosen_pet_id)
    
    # Grab row for chosen Pet ID
    try:
        df_chsn = df_all_pred.loc[df_all_pred['PetId'] == chosen_pet_id]
    except:
        print('Nothing')

    return render_template('results.html',table_1=[df_chsn.to_html(classes='data', index=False, justify='center')],
                           table_2=[df_all_pred.to_html(classes='data', index=False)], titles=df_all_pred.columns.values)                                  
                              
if __name__ == '__main__':
	app.run()                           
                           
                           
                          
    