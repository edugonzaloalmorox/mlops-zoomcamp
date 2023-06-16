
import requests
import os


def download_data(url, filename, folder_path):
# Create the data folder if it doesn't exist
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    file_path = os.path.join(folder_path, filename)

    response = requests.get(url)
    if response.status_code == 200:
        with open(file_path, 'wb') as file:
            file.write(response.content)
        print(f"File {filename} downloaded and saved successfully.")
    else:
        print(f"Failed to download the file {filename}")
        
        
download_data('https://d37ci6vzurychx.cloudfront.net/trip-data/green_tripdata_2023-01.parquet',
              "green_tripdata_2023-01.parquet", 
              'data')

download_data('https://d37ci6vzurychx.cloudfront.net/trip-data/green_tripdata_2023-02.parquet',
              'green_tripdata_2023-02.parquet',
              'data')

download_data('https://d37ci6vzurychx.cloudfront.net/trip-data/green_tripdata_2023-03.parquet',
              'green_tripdata_2023-03.parquet',
              'data')