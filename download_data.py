import requests
import json
import gzip
import shutil
import time
import os
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

   # if response.status_code == 200:
   #      try:
   #          gzip_bytes = response.content
   #          with gzip.GzipFile(fileobj=BytesIO(gzip_bytes), mode="rb") as gzipped_file:
   #              json_data = gzipped_file.read().decode("utf-8")
   #              # Parse the JSON data
   #              data = json.loads(json_data)

   #              # Write the data to a JSON file with indents (e.g., 4 spaces)
   #              with open(f"{sanitized_file_name}.json", "w", encoding="utf-8") as output_file:
   #                  json.dump(data, output_file, ensure_ascii=False, indent=4)  # Add 'indent' parameter

   #              print(f"{sanitized_file_name}.json written")
   #      except Exception as e:
   #          print("Error:", e)
   # else:
   #      print(f"Failed to download {file_name}")
   
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


def download_esports_files():
   directory = "esports-data"
   if not os.path.exists(directory):
       os.makedirs(directory)

   esports_data_files = ["leagues", "tournaments", "players", "teams", "mapping_data"]
   for file_name in esports_data_files:
       download_gzip_and_write_to_json(f"{directory}/{file_name}",f"{directory}/{file_name}")


def download_games(year):
    start_time = time.time()
    with open("esports-data/tournaments.json", "r") as json_file:
        tournaments_data = json.load(json_file)
    with open("esports-data/mapping_data.json", "r") as json_file:
        mappings_data = json.load(json_file)

    base_directory = "games"
    if not os.path.exists(base_directory):
        os.makedirs(base_directory)

    mappings = {
        esports_game["esportsGameId"]: esports_game for esports_game in mappings_data
    }

    game_counter = 0

    for tournament in tournaments_data:
        #if tournament['slug'] == 'lec_spring_2023' or tournament['slug'] == 'lec_winter_2023':
        if tournament['slug'] == 'lec_spring_2023':
            
            start_date = tournament.get("startDate", "")
            if start_date.startswith(str(year)):
                print(f"Processing {tournament['slug']}")
                for stage in tournament["stages"]:
                    
                    directory= f"{base_directory}/{tournament['slug']}/{stage['name']}"
                    if not os.path.exists(directory):
                        os.makedirs(directory)
                        
                    for section in stage["sections"]:
                        for match in section["matches"]:
                            for game in match["games"]:
                                if game["state"] == "completed":
                                    try:
                                        platform_game_id = mappings[game["id"]]["platformGameId"]
                                    except KeyError:
                                        print(f"{game['id']} not found in the mapping table")
                                        continue

                                    download_gzip_and_write_to_json(f"games/{platform_game_id}",f"{directory}/{platform_game_id}")
                                    game_counter += 1

                                    if game_counter % 10 == 0:
                                        print(
                                            f"----- Processed {game_counter} games, current run time: "
                                            f"{round((time.time() - start_time)/60, 2)} minutes"
                                        )
            print(f"Completed {tournament['slug']}")
if __name__ == "__main__":
   #download_esports_files()
   download_games(2023)
