#%% imports
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np


#%% read data
df = pd.read_csv('lec_spring_2023.csv')
df.head()

#%% split up X and y 
code_cols = [col for col in df.columns if "_code" in col]
rolling_cols =[col for col in df.columns if "_rolling" in col]
X_cols = ["start_time"] + code_cols
X = df[X_cols]
X_np = np.array(X,dtype=np.float32).reshape(-1,len(X_cols))
y = df['result']
y_np = np.array(y,dtype=np.float32).reshape(-1,1)
y_np

# %% train test split
X_train, X_test, y_train, y_test = train_test_split(X_np,y_np,test_size=0.2,random_state=345)
y_test

# %% Dataset
class LEC_Dataset(Dataset):
    def __init__(self, X_train, y_train):
        self.X = torch.from_numpy(X_train)
        self.y = torch.from_numpy(y_train)

    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    
# %% dataloader
lec_data = LEC_Dataset(X_train, y_train)
train_loader = DataLoader(lec_data, batch_size = 4)

#%% check
print(lec_data.X.shape, lec_data.X.shape)

# %% model class
class MultiClassNet(nn.Module):
    def __init__(self, features, hidden_features):
        super(MultiClassNet, self).__init__()
        self.linear1 = nn.Linear(features, hidden_features)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_features,1)
        self.activation = nn.Sigmoid()

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return self.activation(x)

# %% define network params
features = lec_data.X.shape[1]
hidden_features = 500
num_epochs = 1000
lr = 0.002

#%% model instance
model = MultiClassNet(features,hidden_features)


# %% loss and optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=lr)
criterion = nn.BCELoss()

# %% training
losses = []
for epoch in range(num_epochs):
    for X, y in train_loader:
        optimizer.zero_grad()
        y_pred = model(X)
        loss = criterion(y_pred,y)
        loss.backward()
        optimizer.step()
    losses.append(float(loss.data.detach().numpy()))

# %% model validation
X_test_torch = torch.from_numpy(X_test)
y_pred = model(X_test_torch)
y_pred = y_pred.detach().numpy()
y_pred_bool = np.round(y_pred)
print(y_pred_bool)
print(accuracy_score(y_pred_bool,y_test))


# %%
print(losses)

# %%
sum(y_pred_bool)

# %%
len(y_pred_bool)

print(model.state_dict)
# %%
