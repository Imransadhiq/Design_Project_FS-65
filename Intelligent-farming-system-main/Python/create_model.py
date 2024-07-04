import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
import pickle

df = pd.read_csv('I:\Intelligent-farming-system-main\Intelligent-farming-system-main\Python\datasets.csv')

Components = ['CropType', 'CropDays', 'SoilMoisture', 'temperature', 'Humidity']
Features = df[Components]
Target = df['Irrigation']

Xtrain, Xtest, Ytrain, Ytest = train_test_split(Features, Target, test_size=0.2, random_state=2)
scaler = MinMaxScaler()
Xtrain = scaler.fit_transform(Xtrain)
Xtest = scaler.transform(Xtest)

RF = RandomForestClassifier(n_estimators=20, random_state=0)
RF.fit(Xtrain, Ytrain)

RF_pkl_filename = 'RandomForest.pkl'
with open(RF_pkl_filename, 'wb') as file:
    pickle.dump(RF, file)

model = pickle.load(open(RF_pkl_filename, 'rb'))
data = np.array([[2, 3, 189, 24, 50]])
prediction = model.predict(data)
print(prediction)
