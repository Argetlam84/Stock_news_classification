# %%
import pandas as pd
import os
import datetime as dt
import glob

# %%
today_date = dt.datetime.today().strftime("%Y-%m-%d")

def merge_with_csv(data):
    main_file = "datasets/clustered.csv"
    #sentences = "datasets/sentences.csv"

    
    if isinstance(data, pd.DataFrame):
        new_file = "datasets/temp_new_file.csv"
        data.to_csv(new_file, index=False)
    else:
        
        new_file = data

    if not os.path.exists(main_file):
        
        pd.read_csv(new_file).to_csv(main_file, index=False)
        
    
    main_df = pd.read_csv(main_file)
    new_df = pd.read_csv(new_file)
    #sentences_df = pd.read_csv(sentences)
    combined_df = pd.concat([main_df, new_df], ignore_index=True)
    combined_df.to_csv(main_file, index=False)

    
    if os.path.exists("datasets/temp_new_file.csv"):
        os.remove("datasets/temp_new_file.csv")

    
    files_to_delete = glob.glob("datasets/clustered_*.csv")
    files_to_delete = [f for f in files_to_delete if f != main_file]
    for file_path in files_to_delete:
        os.remove(file_path)
        print(f"{file_path} removed")

    
    files_to_delete = glob.glob("datasets/stock_market_news_*.csv")
    for file_path in files_to_delete:
        os.remove(file_path)
        print(f"{file_path} removed")

    return combined_df

