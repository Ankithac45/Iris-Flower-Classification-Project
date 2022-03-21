import pandas as pd
df = pd.read_csv('iris[1].csv')
df.head()
df.shape
df.info()
df['Species'].unique()
from sklearn.preprocessing import LabelEncoder
le_x =LabelEncoder()
df['Species_target']=le_x.fit_transform(df['Species'])
df.head()
from matplotlib import pyplot as plt
df01=df[df['Species']=='Iris-setosa']
df01.shape
df02 = df[df.Species =='Iris-versicolor']
df03 = df[df.Species =='Iris-virginica']
plt.scatter(df01['PetalLengthCm'],df01['PetalWidthCm'], color='red', marker='+')
plt.scatter(df02['PetalLengthCm'],df02['PetalWidthCm'], color='blue', marker='*')
plt.scatter(df03['PetalLengthCm'],df03['PetalWidthCm'], color='green', marker='.')
plt.xlabel('Petal Length')
plt.ylabel('Petal Width')
plt.show()

X = df.drop(['Species','Species_target'],axis=1)
y = df['Species_target']
#split into train and test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=1)
from sklearn.svm import SVC
model_svc =SVC()
model_svc.fit(X_train, y_train)
model_svc.score(X_test,y_test)