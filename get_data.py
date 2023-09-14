import requests
import json
import gzip
import shutil
import time
import os
from datetime import datetime, date
import pandas as pd
from io import BytesIO

S3_BUCKET_URL = "https://power-rankings-dataset-gprhack.s3.us-west-2.amazonaws.com"

def sanitize_file_name(file_name):
    # Replace invalid characters (e.g., colon) with underscores
    sanitized_name = file_name.replace(":", "_")
    return sanitized_name

def download_gzip_and_write_to_json(file_name,output_name):
   # If file already exists locally do not re-download game
   
   sanitized_file_name = sanitize_file_name(output_name)
   
   if os.path.isfile(f"{sanitized_file_name}.json"):
        return
    
   response = requests.get(f"{S3_BUCKET_URL}/{file_name}.json.gz")
   
   if response.status_code == 200:
        try:
            gzip_bytes = BytesIO(response.content)
            with gzip.GzipFile(fileobj=gzip_bytes, mode="rb") as gzipped_file:
                output_file_name = f"{sanitized_file_name}.json"
                with open(output_file_name, 'wb') as output_file:
                    shutil.copyfileobj(gzipped_file, output_file)
                print(f"{sanitized_file_name}.json written")
        except Exception as e:
            print("Error:", e)
   else:
        print(f"Failed to download {file_name}")

def download_gzip_and_parse_json(file_name):
    response = requests.get(f"{S3_BUCKET_URL}/{file_name}.json.gz")

    if response.status_code == 200:
        try:
            gzip_bytes = BytesIO(response.content)
            with gzip.GzipFile(fileobj=gzip_bytes, mode="rb") as gzipped_file:
                gzipped_content = gzipped_file.read().decode("utf-8")
                json_data = json.loads(gzipped_content)
                #print("JSON data loaded into a variable:")
                #print(json_data)
        except Exception as e:
            print("Error:", e)
    else:
        print(f"Failed to download {file_name}")
    return json_data

def download_esports_files():
    directory = "esports-data"
    if not os.path.exists(directory):
        os.makedirs(directory)

    esports_data_files = ["leagues", "tournaments", "players", "teams", "mapping_data"]
    for file_name in esports_data_files:
        download_gzip_and_write_to_json(f"{directory}/{file_name}",f"{directory}/{file_name}")
        
def get_date(date_str):
    return datetime.strptime(date_str, "%Y-%m-%dT%H:%M:%S.%fZ").date()

def convert_time_to_seconds(time_str):
    time = datetime.strptime(time_str, "%Y-%m-%dT%H:%M:%S.%fZ").time()
    return time.hour * 3600 + time.minute * 60 + time.second


def get_league(league_id):
    for league in leagues_json:
        if league["id"] ==league_id:
            league_name = league["name"]
            if league["region"] == "INTERNATIONAL":
                return None

            return league_name
    return "other"

def get_game_info(game_df, match, tournament,stage, league_id):
    
    game_info = match[0]
    game_end = match[-1]
    #game_id = int((game_end["gameName"]).split("|")[0])                 
    game = game_df
    
    start_time = convert_time_to_seconds(game_info["eventTime"])
    #print(start_time)
    end_time = game_end["gameTime"]/1000
    date = get_date(game_info["eventTime"])

    game["date"] = [date]
    game["start_time"] = [start_time]
    #game["end_time"] = [end_time]
    game["duration"] = [end_time]

    patch = game_info["gameVersion"]
    game["patch"] =[patch]
    game["tournament"] = tournament
    participants =game_info["participants"]
    roles = ["top","jng","mid","adc","sup","top","jng","mid","adc","sup"]
    
    for player,role in zip(participants,roles):
        #summonerName =player["summonerName"].split(" ")
        #team_tag = summonerName.pop(0)
        player_name = player["summonerName"]
        if player["teamID"]==100:
            side = "blue"
        else:
            side = "red"
        #game[f"{side}_team"]= [team_tag]
        game[f"{side}_{role}"]= [player_name]
    game["stage"]=[stage]
    if game_end["winningTeam"]== 100:
        game["result"]=[1]
    else:
        game["result"]=[0]
    for side in ["blue","red"]:
        team = game.iloc[0][f"{side}_team"]
        if team in league_names:
            #print(league_names[team])
            game[f"{side}_league"]=league_names[team]
        else:
            league = get_league(league_id)
            league_names[team]= league
            game[f"{side}_league"] = league
            print(f"added {team} to league_names")
    return(game)


def get_status_updates(match,game):
    start_time = game.iloc[0]["start_time"]
    #print(start_time)
    roles = ["top","jng","mid","adc","sup","top","jng","mid","adc","sup"]
    end_time = int(match[-2]["gameTime"]) //1000
    columns_to_concat = []
    for event in match:
        try:
            event_time = int(event["gameTime"]) //1000
        except:
            continue
        for time in ["600", "840", end_time]: #can add more times, 300 sec for 5 mins game time
            if (event["eventType"] == "stats_update") and (event_time == int(time)):
                if time == end_time:
                    time = "end"
                    end_time = 8 #some end times were twice so doing this makes it only happen once
                participants = event["participants"]
                variables =["totalGold","level"] #, "shutdownValue"
                #other stats that have been taken out "WARD_KILLED", "WARD_PLACED", "TOTAL_DAMAGE_DEALT", "TOTAL_DAMAGE_TAKEN"
                stats= ["NEUTRAL_MINIONS_KILLED_ENEMY_JUNGLE", "CHAMPIONS_KILLED","NUM_DEATHS", "ASSISTS", "VISION_SCORE", "TOTAL_DAMAGE_DEALT_TO_CHAMPIONS"]
                pings = [ "BASIC_PINGS", "COMMAND_PINGS","DANGER_PINGS","GET_BACK_PINGS","RETREAT_PINGS", "ON_MY_WAY_PINGS", "ASSIST_ME_PINGS", "ENEMY_MISSING_PINGS", "PUSH_PINGS", "ALL_IN_PINGS", "HOLD_PINGS", "BAIT_PINGS", "VISION_CLEARED_PINGS", "ENEMY_VISION_PINGS", "NEED_VISION_PINGS"]
                team_stats = ["inhibKills","towerKills","baronKills","dragonKills"]
                minions = ["MINIONS_KILLED","NEUTRAL_MINIONS_KILLED","NEUTRAL_MINIONS_KILLED_YOUR_JUNGLE","NEUTRAL_MINIONS_KILLED_ENEMY_JUNGLE"]
                
                red_pings=0
                blue_pings =0
                for player,role in zip(participants,roles):
                    if player["teamID"]==100:
                        side = "blue"
                    else:
                        side = "red"
                    for var in variables:
                        #game[f"{side}_{role}_{var}_{time}"] = [player[var]]
                        columns_to_concat.append(pd.Series(player[var], name=f"{side}_{role}_{var}_{time}"))
                    stat_list = player["stats"]
                    cs =0
                    for stat in stat_list:
                        name = stat["name"]
                        if name in minions:
                            cs += stat["value"]
                            
                        if name in stats:
                            #game[f"{side}_{role}_{name}_{time}"] = stat["value"]
                            columns_to_concat.append(pd.Series(stat["value"], name=f"{side}_{role}_{name}_{time}"))
                            
                        elif name in pings:
                            if side =="blue":
                                blue_pings += stat["value"]
                            else:
                                red_pings += stat["value"]
    
                    columns_to_concat.append(pd.Series(cs, name=f"{side}_{role}_cs_{time}"))        
                    #game[f"{side}_{role}_cs_{time}"] = [cs]
            
                columns_to_concat.append(pd.Series(blue_pings, name=f"blue_pings_{time}"))
                columns_to_concat.append(pd.Series(red_pings, name=f"red_pings_{time}"))
                #game[f"blue_pings_{time}"] = blue_pings
                #game[f"red_pings_{time}"] = red_pings
                for stat in team_stats:
                    #game[f"blue_{stat}_{time}"] =[event["teams"][0][stat]]
                    #game[f"red_{stat}_{time}"] =[event["teams"][1][stat]]
                    columns_to_concat.append(pd.Series(event["teams"][0][stat], name=f"blue_{stat}_{time}"))
                    columns_to_concat.append(pd.Series(event["teams"][1][stat], name=f"red_{stat}_{time}"))
    game = pd.concat([game] + columns_to_concat, axis=1)
    return(game)

def download_games(year, df):
    start_time = time.time()
    with open("esports-data/tournaments.json", "r") as json_file:
        tournaments_data = json.load(json_file)
    with open("esports-data/mapping_data.json", "r") as json_file:
        mappings_data = json.load(json_file)

    #base_directory = "games"
    #if not os.path.exists(base_directory):
        #os.makedirs(base_directory)

    mappings = {
        esports_game["esportsGameId"]: esports_game for esports_game in mappings_data
    }

    game_counter = 0
    if not os.path.exists("tournaments"):
        os.makedirs("tournaments")

    for tournament in tournaments_data:
        if os.path.isfile(f"tournaments/{tournament['slug']}.csv"):
             continue
        
        df = pd.DataFrame()
        start_date = tournament.get("startDate", "")
        if start_date.startswith(str(year)):
            tournament_slug = tournament['slug']
            league_id = tournament["leagueId"]
            print(league_id)
            print(f"Processing {tournament_slug}")
            for stage in tournament["stages"]:
                stage_name = stage['name']   
                        
                for section in stage["sections"]:
                    for match in section["matches"]:
                        best_of = match["strategy"]["count"]
                        
                        for game in match["games"]:
                             if game["state"] == "completed":
                                try:
                                     platform_game_id = mappings[game["id"]]["platformGameId"]
                                except KeyError:
                                    print(f"{game['id']} not found in the mapping table")
                                    continue
                                print(platform_game_id)
                                #download_gzip_and_write_to_json(f"games/{platform_game_id}",f"{directory}/{platform_game_id}")
                                try:
                                    blue_team_id = str(game["teams"][0]["id"])
                                    red_team_id = str(game["teams"][1]["id"])
                                    #print(blue_team_id)
                                    #print(red_team_id)
                                    game_data = pd.DataFrame()
                                    game_data["blue_team"]= [blue_team_id]
                                    game_data["red_team"]= [red_team_id]
                                    #print(game_data["blue_team"])
                                    match_json = download_gzip_and_parse_json(f"games/{platform_game_id}")
                                    game_data = get_game_info(game_data, match_json,tournament_slug,stage_name, league_id)
                                
                                    game_data["best_of"] =best_of
                                    game_data["tournament"] = tournament_slug
                                    game_data= get_status_updates(match_json,game_data)
                                    #df =pd.concat([df,game_data])
                                    #df = df.reset_index(drop=True)
                                    #print(game_data.columns)
                                    #game_data.to_csv(f"game_data.csv",index=False)
                                    df = pd.concat([df, game_data], axis=0, ignore_index=True)
                                    game_counter += 1
                                except ValueError as e:
                                    print(e)
                                    print("skipping game ")
                                    continue
                                # KeyError
                                except KeyError as e:
                                    print(e)
                                    print(type(e))
                                    print("skipping game ")
                                    continue

                                if game_counter % 10 == 0:
                                    print(
                                        f"----- Processed {game_counter} games, current run time: "
                                        f"{round((time.time() - start_time)/60, 2)} minutes")
        
            df.to_csv(f"tournaments/{tournament['slug']}.csv",index=False)
            with open('league_names.json', 'w') as f:
                json.dump(league_names, f)                            
            print(f"Completed {tournament['slug']}")
        
if __name__ == "__main__":
    while True:
        try:
            if os.path.isfile("league_names.json"):
                with open('league_names.json', 'r') as f:
                    league_names= json.load(f)
            else:
                league_names = {}
                
            df = pd.DataFrame()
            download_esports_files()
            
            with open("esports-data/leagues.json", "r") as json_file:
                leagues_json = json.load(json_file)
                
            download_games(2023,df)
            download_games(2022,df)
            #download_games(2021)
            #download_games(2020)
            break # stop the loop if the function completes sucessfully
        except Exception as e:
            print(e)
            print("Retrying in 15 sec ")
            time.sleep(15)
        