import numpy as np

X=np.array([0,10,20,30,40]).reshape(-1,1)
y=np.array([32,50,68,86,104])

from sklearn.model_selection import  train_test_split
from sklearn.linear_model import LinearRegression

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.4,random_state=42)

model=LinearRegression()

model.fit(X_train,y_train)

a=model.predict(np.array([[100]]))
print(a[0])

# import pickle
# with open('model.pkl', 'wb') as files:
#     pickle.dump(model, files)