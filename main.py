import pandas as pd
from Network import Network
from sklearn.preprocessing import StandardScaler


df = pd.read_csv('./data/data.csv')
df = df.drop('id', axis=1)
df['diagnosis'] = df['diagnosis'].map({'B':0, 'M':1})
X=df.drop('diagnosis', axis=1)
y=df['diagnosis']

X_train = X.loc[:500]
y_train = y.loc[:500]
X_test = X.loc[500:]
y_test = y.loc[500:]


sc = StandardScaler()
X_train_std = sc.fit_transform(X_train)
X_test_std = sc.transform(X_test)

network = Network([30, 50, 2])
network.SGD(X_train_std, y, 5, 1, 0.001, test_data=(X_test_std, y_test))
