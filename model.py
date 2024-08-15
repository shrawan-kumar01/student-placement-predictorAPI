import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split as tts

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle


df = pd.read_csv('placement-dataset.csv')

x = df.drop(columns=['placement'])
y = df['placement']

x_train,x_test,y_train,y_test = tts(x,y,test_size = 0.2,random_state = 2)


#  creating random forest instance
rf = RandomForestClassifier()
rf.fit(x_train,y_train)
y_predict = rf.predict(x_test)
accuracy_score(y_test,y_predict)

# make  pickle
# The '.pk1' extension is commonly used to denote files that contain Python objects serialized using the pickle module.
pickle.dump(rf,open('model_placement.pk1','wb'))