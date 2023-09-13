import json
from datetime import datetime, date
import pandas as pd
import matplotlib.pyplot as plt
import warnings
#warnings.filterwarnings("ignore")
import os

def differences(df):
    roles = ["top","jng","mid","adc","sup"]
    times= ["600","840"]

    for side, opp_side in zip(["blue","red"],["red","blue"]):
        for role in roles:
            for time in times:
                for stat in ["cs", "totalGold"]:
                    df[f"{side}_{role}_{stat}_diff_{time}"] = df[f"{side}_{role}_{stat}_{time}"] - df[f"{opp_side}_{role}_{stat}_{time}"]
    return(df)


def statspermin(df):
    roles = ["top","jng","mid","adc","sup"]

    for side in ["blue","red"]:
        for role in roles:
            for stat in ["cs", "totalGold", "VISION_SCORE", "TOTAL_DAMAGE_DEALT_TO_CHAMPIONS"]:
                df[f"{side}_{role}_{stat}_per_min"] = df[f"{side}_{role}_{stat}_end"] /(df["duration"] / 60)
    return(df)


with open("C:/GLOBAL POWER RANKINGS/esports-data/tournaments.json", "r") as json_file:
                tournaments_json = json.load(json_file)
        
with open("C:/GLOBAL POWER RANKINGS/esports-data/leagues.json", "r") as json_file:
                leagues_json = json.load(json_file)

#need to add: team names? not necessary for model (would just need to input team id instead)
#make sure team id is seen as a string
#get team tag and remove it from player names

df= pd.DataFrame()
for tournament in os.listdir("tournaments"):
    print(f" processing {tournament}")
    tounament_df = pd.read_csv(f"tournaments/{tournament}")
    
    df = pd.concat([df,tounament_df])

df = df.sort_values(by =["date", "start_time"])
df.reset_index(drop=True, inplace=True)

df = differences(df)
df = statspermin(df)

blue_cols = ["date","start_time", "duration"] +[col for col in df.columns if "blue" in col]
red_cols = ["date","start_time", "duration"] +[col for col in df.columns if "red" in col]
team_cols =[col.replace("blue_","") for col in blue_cols]
#print(team_cols)
blue = df[blue_cols]
red = df[red_cols]

#Get the ids of all of the teams
teams = df["blue_team"].drop_duplicates()

#separating the data for individual teams to get rolling averages
team_data = pd.DataFrame()
for team in teams:
    for index,row in df.iterrows():
        if row["blue_team"] == team:
            blue_renamed = blue.rename(columns=dict(zip(blue_cols, team_cols)))
            team_data = pd.concat([team_data, blue_renamed.iloc[[index]]])
        elif row["red_team"] == team:
            red_renamed = red.rename(columns=dict(zip(red_cols, team_cols)))
            team_data = pd.concat([team_data, red_renamed.iloc[[index]]])
            
grouped_games = team_data.groupby("team")

def rolling_avg(group,cols,new_cols):
    group = group.sort_values("date")
    rolling_stats = group[cols].rolling(3, closed ='left').mean() #assigns avg from previous 3 games onto the 4th game
    group[new_cols]=rolling_stats
    group = group.dropna(subset = new_cols) #drops the first 3 games without rolling averages
    return group

cols = [col for col in team_data.columns if "_end" in col or "_300" in col or "_840" in col or "per_min" in col] + ["duration"] 
new_cols = [f"{c}_rolling" for c in cols]

games_rolling = grouped_games.apply(lambda x:rolling_avg(x,cols,new_cols))
games_rolling = games_rolling.droplevel('team')

games = games_rolling.sort_index()

# Recombining separate game data into the main set
rolling_cols = [col for col in games.columns if "rolling" in col]
blue_rolling_cols = [f"blue_{col}" for col in rolling_cols]
red_rolling_cols = [f"red_{col}" for col in rolling_cols]
#print(red_rolling_cols)
result_df = pd.DataFrame()
for team in teams:
    for index, row in df.iterrows():
        try:
            if index in games.index and row["blue_team"] == team:
                # Copy the original game DataFrame and rename columns
                game = games.rename(columns=dict(zip(rolling_cols, blue_rolling_cols)))
                # Adding in the rolling columns
                game = game[(game.index == index) & (game["team"] == team)]
                #print(game.index)
                new_cols = game.loc[index,blue_rolling_cols]
                df.loc[index, blue_rolling_cols]= new_cols
            elif index in games.index and row["red_team"] == team:
                # Copy the original game DataFrame and rename columns
                game = games.rename(columns=dict(zip(rolling_cols, red_rolling_cols)))
                game = game[(game.index == index) & (game["team"] == team)]
                game
                new_cols = game.loc[index,red_rolling_cols]
                df.loc[index, red_rolling_cols]= new_cols
        except KeyError:
            #its skipping the first 3 games for "new" teams as this data has been removed
            print(f"{index} for {team} not found")
            continue

df = df.dropna()

roles = ["top","jng","mid","adc","sup"]
code_cols = []
for side in ["blue","red"]:
    for role in roles:
        code_cols += [col for col in df.columns if f"{side}_{role}" == col]
    code_cols += [f"{side}_team", f"{side}_league"]
code_cols += ["best_of"]
new_code_cols =[name + '_code' for name in code_cols]
df.rename(columns = dict(zip(code_cols,new_code_cols)), inplace=True)

df.best_of_code = df.best_of_code.astype(str)
df.blue_team_code = df.blue_team_code.astype(str)
df.red_team_code = df.red_team_code.astype(str)
dates= df["date"]
df= df.drop(columns=["date", "patch"])
df_dummies = pd.get_dummies(df)
df_dummies["date"] = dates
df["date"] = dates

df_dummies.to_csv("data_one_hot.csv",index=False)