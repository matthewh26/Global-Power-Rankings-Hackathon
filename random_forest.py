#%% imports
import pandas as pd
import numpy as np
import sklearn 
import sklearn.ensemble as en
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler



# %% load data
df = pd.read_csv('lec_spring_2023.csv')
df.head()

# %% drop unnecessary columns
code_cols = [col for col in df.columns if "_code" in col]
rolling_cols =[col for col in df.columns if "_rolling" in col]
features = ["start_time"] + code_cols + rolling_cols
X = df[features]
y = df["result"]

# %% preprocessing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)
scaler = StandardScaler()
X_train_scale = scaler.fit_transform(X_train)
X_test_scale = scaler.transform(X_test)

# %% random forest instance
random_forest = en.RandomForestClassifier(n_estimators=1000)
random_forest.fit(X_train, y_train)
print(random_forest.feature_importances_)


# %%
y_pred = random_forest.predict(X_test)
from sklearn.metrics import accuracy_score
accuracy_score(y_test, y_pred)

# %%
