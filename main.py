from web_scrapper import fetch_data
from news_clustering import process_data
from combine import merge_with_csv
from models import train_ml, train_deep
from datetime import datetime


def main():

    data = fetch_data()
    print("Data collected from Yahoo Finance news")

    processed_data = process_data(data)
    print("Data processed successfully")

    final_data = merge_with_csv(processed_data)
    print("All datas merged together")

    if datetime.now() == 1:

        train_ml(final_data)
        
        train_deep(final_data)
        
    else:
        print("Training models function passed because it will be work on each month beginings")



if __name__=="__main__":
    main()