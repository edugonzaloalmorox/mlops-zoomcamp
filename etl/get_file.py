
import atoma 
import pandas as pd
import requests as re
from pprint import pprint  
from bs4 import BeautifulSoup
import feedparser
import urllib.request
import zipfile
import os

pd.set_option('display.max_columns', None)


def get_atom(year_month):
    url = f'https://contrataciondelsectorpublico.gob.es/sindicacion/sindicacion_643/licitacionesPerfilesContratanteCompleto3_{year_month}.zip'
    file_name = f'data/{year_month}.zip'
    
    urllib.request.urlretrieve(url, file_name)
    
    destination_folder = f'data/{year_month}'
    
    # Create the folder if it doesn't exist
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)
    
    with zipfile.ZipFile(file_name, "r") as zip_ref:
        zip_ref.extractall(destination_folder) 
        print('extraction completed')
    
   



def read_atom(year_month):
    lst_docs = []
    folder_path = f'data/{year_month}'
    file_names = os.listdir(folder_path)
    for file in file_names:
        x = file
        lst_docs.append(x)
    
    return(lst_docs)
    
        

def parse_values(docs, n):
    
    
    objeto = docs.entries[n].title.value
    perfil_contratante = docs.entries[n].id_
    date_update = docs.entries[n].updated
    link = docs.entries[n].links[0].href
    licitacion = docs.entries[n].summary.value
    
    doc = docs 

    dict = {
        'objeto': objeto,
        'perfil_contratante': perfil_contratante,
        'fecha_actualización': date_update,
        'link': link,
        'licitacion': licitacion, 
        'doc': doc     
    } 
    
    dict_df = pd.DataFrame(dict,index=[0])
    
    return dict_df



def clean_licitacion(df):
    lic = df[['licitacion']]
    lic[['id', 'organo', 'importe', 'estado']] = lic.licitacion.str.split(';', expand=True)
    
    lic['id'] = lic['id'].str.replace('Id licitación: ', '')
    lic['organo'] = lic['organo'].str.replace('Órgano de Contratación: ', '')
    lic['importe'] = lic['importe'].str.replace('Importe: ', '')
    lic['importe'] = lic['importe'].str.replace(' EUR', '')
    lic['estado'] = lic['estado'].str.replace('Estado: ', '')
    
    
    
    df_new = pd.concat([df, lic], axis=1)
    

    return(df_new)




def etl_web_to_data(year_month):
    zip_file = get_atom(year_month)
    
    
    
    
    lst = []
    n_values = list(range(1, len(feed.entries)))
    for n in n_values:
        dict_df = parse_values(feed, n)
        lst.append(dict_df)


    result_df = pd.concat(lst, ignore_index=True)

    final_df = clean_licitacion(result_df)
    
    return final_df



x = read_atom(202307)
print(x)
    





