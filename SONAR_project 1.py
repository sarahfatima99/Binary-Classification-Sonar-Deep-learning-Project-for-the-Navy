#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as p
from keras.models import Sequential
from keras.layers import Dense
from sklearn.pipeline import Pipeline 

from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder 
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import StratifiedKFold


# In[2]:


seed=7
np.random.seed(seed)


# In[3]:


dataframe=p.read_csv("sonar.csv",header=None)
dataset=dataframe.values


X=dataset[:,0:60].astype(float)
Y=dataset[:,60]
print(X.shape)
print(Y.shape)
print(X)
print(Y)


# In[4]:


from sklearn.preprocessing import LabelEncoder
labelencoder=LabelEncoder()
encoded_y = labelencoder.fit_transform(Y)



# In[5]:


def build_model():
    
    model = Sequential()
    model.add(Dense(10, activation='relu',  input_shape=(60,)))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(1, activation='sigmoid')  )    
    from keras import optimizers
    model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
    return model
estimator= KerasClassifier(build_fn=build_model,epochs=100,batch_size=5,verbose=0)
kfold=StratifiedKFold(n_splits=10,shuffle=True,random_state=seed)
results=cross_val_score(estimator,X,encoded_y,cv=kfold)
print("Results:%.2f%%(%.2f%%)"%(results.mean()*100,results.std()*100))
from sklearn.preprocessing import StandardScaler
np.random.seed(seed)
estimators=[]
estimators.append(('standardize',StandardScaler()))
estimators.append(('mlp',KerasClassifier(build_fn=build_model,epochs=100,batch_size=5,verbose=0)))
pipeline=Pipeline(estimators)
Kfold=StratifiedKFold(n_splits=10,shuffle=True,random_state=seed)
result=cross_val_score(pipeline,X,encoded_y,cv=Kfold)
print("standardize:%.2f%%(%.2f%%)"%(result.mean()*100,result.std()*100))


# # SMALLER MODEL

# In[ ]:


def build_model(): #DONOT USE BOTTEL NECK; IE DONOT USE SMALLER WIEGHS THEN LARGER THEN SMALLER, DECREASE WOEGHS IN SEGWUENCE
    
    model = Sequential()
    model.add(Dense(3, activation='relu',  input_shape=(60,)))
    model.add(Dense(40, activation='relu'))
    model.add(Dense(1, activation='sigmoid')  )    
    from keras import optimizers
    model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
    return model
estimator= KerasClassifier(build_fn=build_model,epochs=100,batch_size=5,verbose=0)
kfold=StratifiedKFold(n_splits=10,shuffle=True,random_state=seed)
results=cross_val_score(estimator,X,encoded_y,cv=kfold)
print("Results:%.2f%%(%.2f%%)"%(results.mean()*100,results.std()*100))


# # LARGER MODEL

# In[ ]:


def build_model():
    
    model = Sequential()
    model.add(Dense(60, activation='relu',  input_shape=(60,)))
    model.add(Dense(60, activation='relu'))  
    model.add(Dense(30, activation='relu'))
    model.add(Dense(1, activation='sigmoid')  )    
    from keras import optimizers
    model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
    return model
estimator= KerasClassifier(build_fn=build_model,epochs=100,batch_size=5,verbose=0)
kfold=StratifiedKFold(n_splits=10,shuffle=True,random_state=seed)
results=cross_val_score(estimator,X,encoded_y,cv=kfold)
print("Results:%.2f%%(%.2f%%)"%(results.mean()*100,results.std()*100))
def build_model():
    
    model = Sequential()
    model.add(Dense(605, activation='relu',  input_shape=(60,)))
    model.add(Dense(608, activation='relu'))  
    model.add(Dense(309, activation='relu'))
    model.add(Dense(1, activation='sigmoid')  )    
    from keras import optimizers
    model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
    return model
estimator= KerasClassifier(build_fn=build_model,epochs=500,batch_size=5,verbose=0)
kfold=StratifiedKFold(n_splits=10,shuffle=True,random_state=seed)
results=cross_val_score(estimator,X,encoded_y,cv=kfold)
print("Results:%.2f%%(%.2f%%)"%(results.mean()*100,results.std()*100))


# # over fits
# 

# In[ ]:





# In[ ]:


def build_model():
    
    model = Sequential()
    model.add(Dense(60, activation='relu',  input_shape=(60,)))
    model.add(Dense(30, activation='relu'))  
    model.add(Dense(1, activation='sigmoid')  )    
    from keras import optimizers
    model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
    return model
estimator= KerasClassifier(build_fn=build_model,epochs=100,batch_size=5,verbose=0)
kfold=StratifiedKFold(n_splits=10,shuffle=True,random_state=seed)
results=cross_val_score(estimator,X,encoded_y,cv=kfold)
print("Results:%.2f%%(%.2f%%)"%(results.mean()*100,results.std()*100))


# In[ ]:


def build_model():
    model = Sequential()
    model.add(Dense(6059, activation='relu',  input_shape=(60,)))
    model.add(Dense(698, activation='relu'))  
    model.add(Dense(3090, activation='relu'))
    model.add(Dense(1, activation='sigmoid')  )    
    from keras import optimizers
    model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
    return model
estimator= KerasClassifier(build_fn=build_model,epochs=500,batch_size=5,verbose=0)
kfold=StratifiedKFold(n_splits=10,shuffle=True,random_state=seed)
results=cross_val_score(estimator,X,encoded_y,cv=kfold)
print("Results:%.2f%%(%.2f%%)"%(results.mean()*100,results.std()*100))


# # TUNNING

# In[ ]:





# #  FUNCTIONAL API

# In[ ]:





# In[ ]:


from keras.models import Model
from keras.layers import Input, Dense
def build_model():
    input_tensor = Input(shape=(60,))
    x = Dense(74, activation='relu')(input_tensor)
    x = Dense(63, activation='relu')(x)
    output_tensor = Dense(1, activation='sigmoid')(x)
    model = Model(input_tensor, output_tensor)
    model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
    return model
estimator= KerasClassifier(build_fn=build_model,epochs=100,batch_size=5,verbose=0)
kfold=StratifiedKFold(n_splits=10,shuffle=True,random_state=seed)
results=cross_val_score(estimator,X,encoded_y,cv=kfold)


# In[ ]:





# # WITH KERAS

# In[5]:


np.random.shuffle(Y)
np.random.shuffle(X)
data=Y
string = 'MR'
char_to_int = dict((c, i) for i, c in enumerate(string))
y= [char_to_int[char] for char in data]
print(y)


# In[6]:


def build_model():
    
    model = Sequential()
    model.add(Dense(10, activation='relu',  input_shape=(60,)))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(1, activation='sigmoid')  )    
    from keras import optimizers
    model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
    return model


# In[7]:


xtrain=X[:156]
ytrain=y[:156]
k=4
num_val_samples = len(xtrain)//k
num_epochs = 100
all_mae_histories = []
for i in range(k):
    print('processing fold #', i)
    val_data = xtrain[i * num_val_samples: (i + 1) * num_val_samples]
    print(xtrain.shape)
    val_targets = ytrain[i * num_val_samples: (i + 1) * num_val_samples]
    partial_train_data = np.concatenate(
    [xtrain[:i * num_val_samples],
    xtrain[(i + 1) * num_val_samples:]],
    axis=0)
    partial_train_targets = np.concatenate(
    [ytrain[:i * num_val_samples],
    ytrain[(i + 1) * num_val_samples:]],
    axis=0)
    model = build_model()
    model.fit(partial_train_data, partial_train_targets,
    epochs=num_epochs, batch_size=5,verbose=0)
    results = model.evaluate(val_data,val_targets)


# In[8]:


def build_model():
    
    model = Sequential()
    model.add(Dense(10, activation='relu',  input_shape=(60,)))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(1, activation='sigmoid')  )    
    from keras import optimizers
    model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
    return model
model=build_model()
model.fit(X[156:], y[156:], epochs=160, batch_size=512)
results = model.evaluate(X[156:], y[156:])


# # MODEL SUBCLASSING
# 
