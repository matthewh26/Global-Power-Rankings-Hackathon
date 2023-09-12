#%%
import numpy as np
import pandas as pd
import torch
from nn_first_try import BinaryClassNet

#%%
def team_data(side,df,team):
    #grab a team's latest match on the specified side
    team_row = df[df[f'{side}_team_code_{team}']==1].iloc[-1,:].to_frame().T
    cols = [col for col in team_row.columns if f'{side}' in col]
    team_row = team_row[cols]
    team_row.reset_index(drop=True,inplace=True)
    return team_row


def create_X(blue_row, red_row, order, bo):
    # combine the data for the two teams into an X of shape (1,718)
    # that can be inserted into the model yielding a predicted winner
    X = pd.concat([blue_row,red_row],axis=1)
    print(X.shape)
    if bo == 1:
        X['best_of_code_1'] = 1
    else:
        X['best_of_code_1'] = 0
    if bo == 3:
        X['best_of_code_3'] = 1
    else:
        X['best_of_code_3'] = 0
    if bo == 5:
        X['best_of_code_5'] = 1
    else:
        X['best_of_code_5'] = 0
    X = X[order]
    
    return X


def play_match(team_1,team_2,side,bo,order,model):
    # play a match and see who wins!!
    # return 1 = win for blue side
    # return 0 = win for red side

    if side == 'blue':
        blue_team = team_1
        red_team = team_2
    else:
        blue_team = team_2
        red_team = team_1

    blue_row = team_data('blue',df_to_model,blue_team)
    red_row = team_data('red',df_to_model,red_team)
    X = create_X(blue_row,red_row,order,bo)
    X = np.array(X, dtype=np.float32).reshape(1,len(X_cols))
    X = torch.from_numpy(X)

    with torch.no_grad():
        pred = 0
        pred = model(X)
        pred_bool = np.round(pred)
        pred_bool = pred_bool.detach().numpy()[0][0]
    return pred_bool


#--------------------------------------------------------------------------------------------------------------------------------------
#%%
model = BinaryClassNet(718,200,100)
model.load_state_dict(torch.load('model_state_dict.pth'))
model.eval()
print(model.state_dict)

#%%
#read in data
df = pd.read_csv('data_one_hot.csv')
df['date'] = pd.to_datetime(df['date'], dayfirst=True)

#make a dataframe of only cols used in model
code_cols = [col for col in df.columns if "_code" in col]
rolling_cols =[col for col in df.columns if "_rolling" in col]
X_cols = code_cols + rolling_cols
df_to_model = df[X_cols]
order = list(df_to_model.columns)

#only keep teams from 2023 summer
    #current_teams_df = df[df['date']>'2023-05-01']
#hard coded FOR NOW
teams = ['G2','XL','TH','FNC','MAD','BDS','AST','SK','KOI','VIT']
team_wins = dict.fromkeys(teams,0)

#%% play matches loop
for i in range(len(teams)):
    for j in range(len(teams)):
        if i != j:
            results = 0
            results += play_match(teams[i],teams[j],side='blue',
                                bo='bo1',order=order,
                                model=model)
            results += play_match(teams[i],teams[j],side='red',
                                bo='bo1',order=order, 
                                model=model)
            results += play_match(teams[i],teams[j],side='red',
                                bo='bo5',order=order,
                                model=model)
            results += play_match(teams[i],teams[j],side='blue',
                                bo='bo5',order=order,
                                model=model)
            team_wins[teams[i]] += results
    

#%%
print(team_wins)

#rank teams
rankings = sorted(team_wins, key=team_wins.get, reverse=True)

print(rankings)


