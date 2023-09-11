#%% imports
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


#%% read data
df = pd.read_csv('lec_spring_2023.csv')
df = df.iloc[:-1,:]
df.shape

#%% split up X and y 
code_cols = [col for col in df.columns if "_code" in col]
rolling_cols =[col for col in df.columns if "_rolling" in col]
X_cols = ["start_time"] + code_cols + rolling_cols
X = df[X_cols]
X_np = np.array(X,dtype=np.float32).reshape(-1,len(X_cols))
y = df['result']
y_np = np.array(y,dtype=np.float32).reshape(-1,1)

# %% train test split
X_train, X_test, y_train, y_test = train_test_split(X_np,y_np,test_size=0.2,random_state=345)
scaler = StandardScaler()
X_train_scale = scaler.fit_transform(X_train)
X_test_scale = scaler.transform(X_test)

#%% 
X_train.shape

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


# %% model class
class MultiClassNet(nn.Module):
    def __init__(self, features, hidden_features):
        super(MultiClassNet, self).__init__()
        self.linear = nn.Linear(features, hidden_features)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_features,1)
        self.bn1 = nn.BatchNorm1d(hidden_features)
        self.activation = nn.Sigmoid()

    def forward(self, x):
        x = self.bn1(self.linear(x))
        x = self.relu(x)
        x = self.linear2(x)
        return self.activation(x)

# %% define network params
features = lec_data.X.shape[1]
hidden_features = 100
num_epochs = 1000
lr = 0.005

#%% initialise weights and bias function
def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.01)


#%% model instance
model = MultiClassNet(features,hidden_features)
model.apply(init_weights)


# %% loss and optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=lr)
criterion = nn.BCELoss(reduction='mean')

# %% training
losses = []
for epoch in range(num_epochs):
    for X,y in train_loader:
        optimizer.zero_grad()
        y_pred = model(X)
        loss = criterion(y_pred,y)
        loss.backward()
        optimizer.step()
    losses.append(float(loss.data.detach().numpy()))

print('training complete!')


# %% model evaluation
X_test_torch = torch.from_numpy(X_test)
X_train_torch = torch.from_numpy(X_train)

def accuracy_measure(X, y):
    y_pred = model(X)
    y_pred = y_pred.detach().numpy()
    y_pred_bool = np.round(y_pred)
    print('accuracy: %.3f' % (accuracy_score(y_pred_bool,y)))


print('test accuracy: ')
accuracy_measure(X_test_torch, y_test)
print('train accuracy: ')
accuracy_measure(X_train_torch,y_train)


# %%
print(losses)

#%% graph losses
import seaborn as sns
sns.lineplot(x=range(num_epochs),y=losses)

# %%

