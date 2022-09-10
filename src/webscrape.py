import pandas as pd
import numpy as np
import re
import requests
from bs4 import BeautifulSoup
import googlesearch

data = pd.read_csv("data/out/cinemaFeatures.csv")
print(data.head)
print(data.shape)
data = data.drop(data[data.release_date.str.len()!=10].index)
print(data.shape)
data = data.reset_index(drop=True)
data = data.drop(data[data.NumberShows < 10].index)
data = data.reset_index(drop=True)
data["LogGross$"] = np.log(data["Gross$"])
data["release_date"] = pd.to_datetime(data["release_date"])
data = data.sort_values(by="release_date", ascending=True)
data = data.reset_index(drop=True)
data.dropna(subset=['revenue'], inplace=True)
data = data.reset_index(drop=True)
data.dropna(subset=['LogGross$'], inplace=True)
print(data.shape)
data = data.drop(data[data["revenue"] == 0].index)
data = data.reset_index(drop=True)

print(data.Title[0:5])

# Create an URL object
url = 'https://en.wikipedia.org/wiki/12_Years_a_Slave_(film)'
tables = pd.read_html(url)

page = requests.get(url)
soup = BeautifulSoup(page.text, 'lxml')
#print(soup.prettify())
# headers = ['Premise', 'Plot', 'Cast', 'Actors']
headers = ['Premise', 'Reception', 'Production', 'Plot', 'Release', 'Cast', 'Filming', 'Music', 'Soundtrack',
           'People appearing in the film', 'Production and release', 'Critical reception', 'Synopsis', 'Reviews',
           'Themes', 'Historical accuracy']

selector = ', '.join([f'h2:-soup-contains("{header}") + p' for header in headers])
paragraphs_list = [i.text for i in soup.select(selector)]
#print(paragraphs_list)

targets = soup.select('h2', id='Plot')
for target in targets:
    for sib in target.find_next_siblings():
        if sib.name=="p":
            print(sib.text)
        else:
            break
#####################################################################################################

# try:
#     from googlesearch import search
# except ImportError:
#     print("No module named 'google' found")
#
# # to search
# query = "meru film wiki"
#
# urls = []
# for j in search(query, tld="com", num=10, stop=10, pause=2):
#     urls.append(j)
# print(urls[0])
#
#
#
# test_movies = data.Title[0:5]

# for i in range(len(test_movies)):
#     movie = test_movies[i]
#     query = movie + " film wiki"
#     urls = []
#     for search_res in search(query, tld="com", num=10, stop=10, pause=2):
#         urls.append(search_res)
#
#     #create page object
#     page = requests.get(urls[0])
#     soup = BeautifulSoup(page.text, 'lxml')
#     # Obtain information from tag <table>
#     table1 = soup.find('table')
#
#     info = []
#     for j in table1.find_all('tr')[1:]:
#         row_data = j.find_all('td')
#         row = [i.text for i in row_data]
#         info.append(row)
#     print(info)
#
#     #mydata.loc[i] = info
#
# #print(mydata)






