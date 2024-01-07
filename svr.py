import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sklearn
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("ising_data_25.csv")
df = df.drop(columns = df.columns[0])

df = df.sample(frac=1)

X_train, X_test, y_train, y_test = train_test_split(df, df['h'], test_size=0.2)

X_train['magnetization_density'], X_test['magnetization_density'] = X_train['magnetization_density']**2, X_test['magnetization_density']**2

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)

svr_model = SVR(kernel="rbf", C=100)

svr_model.fit(X_train, y_train)

y_pred = svr_model.predict(X_test)
mse = sklearn.metrics.mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

plt.scatter(y_test, y_pred)
plt.plot([0,1.2], [0,1.2])
plt.xlabel("truth")
plt.ylabel("pred")
plt.show()

# plt.scatter(X_train['energy_density'], X_train['h'])
# plt.show()

# plt.scatter(X_train['magnetization_density'], X_train['h'])
# plt.show()