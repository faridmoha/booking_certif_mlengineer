#Importing the required packages..
import numpy as np
import pandas as pd
import pathlib
import re 
import datetime
import time 
import tensorflow as tf
from sklearn.feature_selection import SelectKBest, mutual_info_regression,f_regression
#Importing tensorflow 2.6
from tensorflow import keras
from tensorflow.keras import layers

print(tf.__version__)
#Reading data from the gcs bucket
dataset = pd.read_csv('train_booking.csv')
#dataset = pd.read_csv(r"gs://vertex-ia-custom/CrabAgePrediction.csv")
dataset.tail()
#BUCKET = 'gs://machine-learning-433720-bucket'
dataset.isna().sum()
#dataset = dataset.dropna()
#Data transformation..
#nettoyange de la variable texte clearning fee 
def cleaning_fee(x):
    fst_let = str(x)[0:2]
    if fst_let == 'na' :
        return 4
    elif fst_let =='0_':
        return 0
    elif fst_let =='1_':
        return 1
    elif fst_let == '2_':
        return 2
    elif fst_let == 'O3':
        return 3
    else:
        return 4
dataset['cleaning_fee_clean'] = [cleaning_fee(x) for x in dataset['cleaning_fee']]
dataset['seurity_deposit_clean'] = [cleaning_fee(x) for x in dataset['security_deposit']]
# fonction : création flag pour la variable cleaning fee (frais de menage compris ou pas) 
def costornot(x):
    fst_let = str(x)[0:2]
    if fst_let== 'na':
        return 0
    elif fst_let == '0_':
        return 0
    elif fst_let == '1_':
        return 1
    elif fst_let == '2_':
        return 1
    elif fst_let == '03' :
        return 1
    else:
        return 1
dataset['cleaning_fee_flag'] = [costornot(x) for x in dataset['cleaning_fee']]
dataset['seurity_deposit_flag'] = [costornot(x) for x in dataset['security_deposit']]
# fonction : compteur d'équipements, liste avec virgule et host verification  
def cpt_list(lst):
    lst_virgul = lst.split(',')
    return len(lst_virgul)

dataset['nb_amenities'] = [cpt_list(str(x)) for x in dataset['amenities']]
dataset['nb_host_verifications'] = [cpt_list(str(x)) for x in dataset['host_verifications']]
# fonction : test sur la disponibilité : s'il y a + de 154 jour disponibles 0 sinon 1 
def availibity(x):
    if x > 154 : 
        return 0
    else:
        return 1
dataset['availibity_365_ok'] = [availibity(x) for x in dataset['availability_365']]
# fonction :Comptage du nombre de notes remplies 
def note_description(descript,neigh,notes,transit,acess,interaction,house_rules):
    cpt=0
    for elm in [descript,neigh,notes,transit,acess,interaction,house_rules]:
        if elm in [None,np.nan]:
            cpt +=0
        elif len(str(elm)) > 0:
            cpt +=1
        else:
            cpt +=0
    return cpt
dataset['grade_description'] = [note_description(
                            dataset['description'][x],
                            dataset['neighborhood_overview'][x],
                            dataset['notes'][x],
                            dataset['transit'][x],
                            dataset['access'][x],
                            dataset['interaction'][x],
                            dataset['house_rules'][x])
                            for x in range(dataset.shape[0])]
dataset['len_description'] = [len(str(dataset['description'][x]))+
                            len(str(dataset['neighborhood_overview'][x]))+
                            len(str(dataset['notes'][x]))+
                            len(str(dataset['transit'][x]))+
                            len(str(dataset['access'][x]))+
                            len(str(dataset['interaction'][x]))+
                            len(str(dataset['house_rules'][x]))
                            for x in range(dataset.shape[0])]

dataset =dataset.drop(['description','neighborhood_overview','notes','name','transit','access','interaction', 'house_rules','host_id','id','host_verifications','amenities','calendar_updated','geolocation','geopoint_announce','city','country','first_review','last_review','host_since','department'],axis =1)
def features_engenering(x):
    df_cat = x.select_dtypes("object")
    df_num = [i for i in x if i not in df_cat]
    df_num = x[df_num]
    df_dummies = pd.get_dummies(df_cat, drop_first=True , dtype=float)
    df_concat = pd.concat([df_num,df_dummies],axis =1)
    return df_concat
df_final = features_engenering(dataset)
# Remplacer en utilisant la médiane pour zipcode et les autres variables par la moyenne 
median = df_final['zipcode'].median()
df_final['zipcode'].fillna(median, inplace=True)
df_final = df_final.fillna(df_final.describe().mean())
print(df_final.isnull().sum().sum())
#dataset = pd.get_dummies(dataset, prefix='', prefix_sep='')
#dataset.tail()
#Dataset splitting..
train_dataset = df_final.sample(frac=0.8,random_state=0)
test_dataset = df_final.drop(train_dataset.index)
train_stats = train_dataset.describe()
#Removing age column, since it is a target column
train_stats.pop("price")
train_stats = train_stats.transpose()
train_stats
#Removing age column from train and test data
train_labels = train_dataset.pop('price')
test_labels = test_dataset.pop('price')

def norma_data(x):
    #To normalise the numercial values
    return (x - train_stats['mean']) / train_stats['std']

normed_train_data = norma_data(train_dataset)
normed_test_data = norma_data(test_dataset)
# selector k variables
selector = SelectKBest(score_func=mutual_info_regression, k=10)
selector.fit(normed_train_data, train_labels)
print(selector.get_feature_names_out())
normed_train_data_selector = normed_train_data[selector.get_feature_names_out()]
normed_train_data_selector = normed_test_data[selector.get_feature_names_out()]
def build_model():
    #model building function
    model = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=[len(normed_train_data_selector.keys())]),
    layers.Dense(64, activation='relu'),
    layers.Dense(1)
  ])

    optimizer = tf.keras.optimizers.RMSprop(0.001)

    model.compile(loss='mse',
                optimizer=optimizer,
                metrics=['mae', 'mse'])
    return model

#model = build_model()

#model.summary()

model = build_model()
EPOCHS = 10

early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)

early_history = model.fit(normed_train_data_selector, train_labels,
                    epochs=EPOCHS, validation_split = 0.2,
                    callbacks=[early_stop])
#model.save(BUCKET + '/model')