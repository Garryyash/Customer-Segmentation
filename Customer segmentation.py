# -*- coding: utf-8 -*-
"""
Created on Wed Jun 22 09:55:04 2022

@author: caron
"""

#%% IMPORTS

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as ss
import numpy as np
import pickle
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import TensorBoard
import datetime
from tensorflow.keras.utils import plot_model
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Dense,Dropout,BatchNormalization
from tensorflow.keras import Sequential,Input

from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import ConfusionMatrixDisplay
#%% FUNCTIONS

def plot_cat(df,cat_columns):
    for plot_cat in cat_columns:
        plt.figure()
        sns.countplot(df[plot_cat])
        plt.show()

def plot_con(df,con_columns):
    for plot_con in con_columns:
        plt.figure()
        sns.distplot(df[plot_con])
        plt.show()

def cramers_corrected_stat(confusion_matrix):
    """ calculate Cramers V statistic for categorial-categorial association.
        uses correction from Bergsma and Wicher, 
        Journal of the Korean Statistical Society 42 (2013): 323-328
    """
    chi2 = ss.chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum()
    phi2 = chi2/n
    r,k = confusion_matrix.shape
    phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))    
    rcorr = r - ((r-1)**2)/(n-1)
    kcorr = k - ((k-1)**2)/(n-1)
    return np.sqrt(phi2corr / min( (kcorr-1), (rcorr-1)))



#%% STATICS

PATH = os.path.join(os.getcwd(),'Train.csv')

JOB_TYPE_ENCODER_PATH = os.path.join(os.getcwd(),'job_type_encoder.pkl')
MARITAL_ENCODER_PATH = os.path.join(os.getcwd(),'marital_encoder.pkl')
EDUCATION_ENCODER_PATH = os.path.join(os.getcwd(),'education.pkl')
DEFAULT_ENCODER_PATH = os.path.join(os.getcwd(),'default_encoder.pkl')
HOUSING_LOAN_ENCODER_PATH = os.path.join(os.getcwd(),'housing_loan_encoder.pkl')
PERSONAL_LOAN_ENCODER_PATH = os.path.join(os.getcwd(),'personal_loan_encoder.pkl')
COMMUNICATION_TYPE_ENCODER_PATH = os.path.join(os.getcwd(),'communication_type_encoder.pkl')
MONTH_ENCODER_PATH = os.path.join(os.getcwd(),'month_encoder.pkl')
PREV_CAMPAIGN_OUTCOME_ENCODER_PATH = os.path.join(os.getcwd(),'prev_campaign_outcome_encoder.pkl')
TERM_DEPOSIT_SUBSCRIBED_ENCODER_PATH = os.path.join(os.getcwd(),'term_deposit_subscribed_encoder.pkl')


OHE_FILE_NAME = os.path.join(os.getcwd(),'Customer_segmen.pkl')
log_dir = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
LOG_FOLDER_PATH = os.path.join(os.getcwd(),'logs',log_dir)

MODEL_SAVE_PATH = os.path.join(os.getcwd(),'model.h5')
#%% Performing EDA

# STEP 1) LOAD DATA
df = pd.read_csv(PATH)

# STEP 2) DATA INSPECTION
df.info()
df.describe().T

# Visualization
df.boxplot()

#Defining CATEGORICAL & CONTINOU
cat_columns =['job_type','marital','education','default','housing_loan',
              'personal_loan','communication_type','month','prev_campaign_outcome',
              'term_deposit_subscribed']
con_columns =['customer_age','balance','day_of_month','last_contact_duration'
              ,'num_contacts_in_campaign','num_contacts_prev_campaign']

plot_cat(df, cat_columns)
plot_con(df, con_columns)

df.groupby(['education','term_deposit_subscribed']).agg({'term_deposit_subscribed':'count'}).plot(kind='bar')
df.groupby(['job_type','term_deposit_subscribed']).agg({'term_deposit_subscribed':'count'}).plot(kind='bar')

for i in cat_columns:
    plt.figure()
    sns.countplot(df[i],hue=df['term_deposit_subscribed'])
    plt.show()


#%% Regression analysis

# Cramer's V - Categorical Vs Categorical
for i in cat_columns:
    print(i)
    confussion_mat = pd.crosstab(df[i],df['term_deposit_subscribed']).to_numpy()
    print(cramers_corrected_stat(confussion_mat))


# Logistic Regression Categorical Vs Continuos
# Every sample shall be in integers
# Change Categorical data into integers with LabelEncoding 

le = LabelEncoder()
# Check for NaNs
df.isna().sum()  # NaNs is found in Marital and personal_loan which is categorical 
# Cannot do direct label encoding when there is NaNs

paths = [JOB_TYPE_ENCODER_PATH,MARITAL_ENCODER_PATH,EDUCATION_ENCODER_PATH,
         DEFAULT_ENCODER_PATH,HOUSING_LOAN_ENCODER_PATH,PERSONAL_LOAN_ENCODER_PATH,
         COMMUNICATION_TYPE_ENCODER_PATH,MONTH_ENCODER_PATH,PREV_CAMPAIGN_OUTCOME_ENCODER_PATH,
         TERM_DEPOSIT_SUBSCRIBED_ENCODER_PATH]

for index, i in enumerate(cat_columns):
    temp = df[i]
    temp[temp.notnull()]=le.fit_transform(temp[temp.notnull()])
    df[i]=pd.to_numeric(temp,errors='coerce')
    with open(paths[index],'wb') as file:
        pickle.dump(le,file)



# STEP 3) DATA CLEANING
# Labels dropping
df = df.drop(labels='id',axis =1) #Drop ID column from Data Frame
# Dropping "days_since_prev_campaign_contact" label because contains mostly NaNs that has no significance in the training 
df = df.drop(labels='days_since_prev_campaign_contact',axis =1)


# Check for NaNs
df.isna().sum()  # Number of NaNs

# Cheack for duplicated data
df.duplicated().sum() # 0 duplication found 

# To drop NaNs found in customer_age,marital,balance,personal_loan,last_contact_duration,num_contacts_in_campaign

# Mode for categorical data and median for continuos data
#fill mode for NaNs value

# Categorical
for i in ['marital','personal_loan']:
    df[i].fillna(df[i].mode()[0], inplace=True) 
 
#Continous data
df['customer_age'] = df['customer_age'].fillna(df['customer_age'].median())
df['balance'] = df['balance'].fillna(df['balance'].median())
df['last_contact_duration'] = df['last_contact_duration'].fillna(df['last_contact_duration'].median())
df['num_contacts_in_campaign'] = df['num_contacts_in_campaign'].fillna(df['num_contacts_in_campaign'].median())
# some NaNs has not been filled

# Cramer's V - Categorical Vs Categorical
for i in cat_columns:
    print(i)
    confussion_mat = pd.crosstab(df[i],df['term_deposit_subscribed']).to_numpy()
    print(cramers_corrected_stat(confussion_mat))



# finding Correlation & relationship between cont Vs Cat With Logistic Regresion

for con in con_columns:
    print(con)
    lr = LogisticRegression()
    lr.fit(np.expand_dims(df[con],axis=-1),df['term_deposit_subscribed'])
    print(lr.score(np.expand_dims(df[con],axis=-1),df['term_deposit_subscribed']))


# STEP 4) FEATURES SELECTION

# defining the features(X) and target(y)
X = df.loc[:,['customer_age','balance','day_of_month','last_contact_duration',
              'num_contacts_in_campaign','num_contacts_prev_campaign','prev_campaign_outcome']]

y = df['term_deposit_subscribed']




# ONE_HOT_ENCODING


ohe = OneHotEncoder(sparse=False)
y = ohe.fit_transform(np.expand_dims(y, axis=-1))
with open (OHE_FILE_NAME,'wb') as file:
        pickle.dump(ohe,file)




X_train,X_test,y_train,y_test = train_test_split(X,y,
                                                 test_size=0.3,
                                                 random_state=128)


# STEP 5) DATA PREPROCESSING


nb_features = np.shape(X)[1:]
nb_classes = len(np.unique(y_train,axis=0))

model = Sequential()
model.add(Input(shape=(nb_features)))
model.add(Dense(32,activation ='relu',name='Hidden_layer1'))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Dense(32,activation ='relu',name='Hidden_layer2'))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Dense(nb_classes,activation='softmax',name='Output_layer'))
model.summary()

# Wrapping the container
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics='acc')

#CALLBACKS
tensorboard_callback = TensorBoard(log_dir=LOG_FOLDER_PATH)
#early stop
early_stopping_callbacks = EarlyStopping(monitor='loss',patience=5)


# Model training
hist = model.fit(x=X_train,y=y_train,
                      batch_size=128,epochs=100,validation_data=(X_test,y_test),
                      callbacks=[tensorboard_callback,early_stopping_callbacks])


hist.history.keys()
training_loss = hist.history['loss']
training_acc = hist.history['acc']
validation_acc = hist.history['val_acc']
validation_loss = hist.history['val_loss']



plt.figure
plt.plot(training_loss)
plt.plot(validation_loss)
plt.legend(['train_loss','val_loss'])
plt.show()

plt.figure
plt.plot(training_acc)
plt.plot(validation_acc)
plt.legend(['train_acc','val_acc'])
plt.show()


#%% Model Evaluation

results = model.evaluate(X_test,y_test)

print(results)


pred_y = np.argmax(model.predict(X_test),axis=1)
true_y = np.argmax(y_test,axis=1)

cm = confusion_matrix(true_y,pred_y)
cr = classification_report(true_y,pred_y)
print(cm)
print(cr)

labels = ['Zero','One','Two','Three','Four','Five','Six','Seven','Eight','Nine']
disp= ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=labels)
disp.plot(cmap=plt.cm.Blues)
plt.show()

#%% model saving

model.save(MODEL_SAVE_PATH)

#%% TensorBoard plot on browser

plot_model(model,show_shapes=True,show_layer_names=(True))




