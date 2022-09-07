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
url = 'https://en.wikipedia.org/wiki/Meru_(film)'
# Create object page
page = requests.get(url)

# parser-lxml = Change html to Python friendly format
# Obtain page's information
soup = BeautifulSoup(page.text, 'lxml')

# Obtain information from tag <table>
table1 = soup.find('table')

# Obtain every title of columns with tag <th>
headers = []
for i in table1.find_all('th'):
    title = i.text
    headers.append(title)

print(headers)

# Create a dataframe
mydata = pd.DataFrame(columns = headers)

info = []
for j in table1.find_all('tr')[1:]:
    row_data = j.find_all('td')
    row = [i.text for i in row_data]
    info.append(row)

mydata.loc[0] = info
print(mydata)

# select all paragraphs
paragraphs1=soup.select("div p")
#print(paragraphs1)
# select all paragraphs preceded by a header
paragraphs2 = soup.select("h2 ~ p")
#print(paragraphs2)
#select paragraph with header "Premise"
premise_p = soup.select('h2:-soup-contains("Premise") + p')
# print(premise_p)
#parse wiki page for paragraphs with headers: Premise, Reception, Plot and Awards
soup = BeautifulSoup(page.text, 'lxml')
headers = ['Premise', 'Reception', 'Awards', 'Plot']
selector = ', '.join([f'h2:-soup-contains("{header}") + p' for header in headers])
paragraphs_list = [i.text for i in soup.select(selector)]
#print(paragraphs_list)

header= 'premise'
print(f'h2:-soup-contains("{header}") + p')
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
# for j in search(query,tld="com", num=10, stop=10, pause=2):
#     urls.append(j)
# print(urls[0])
#
#
#
# test_movies = data.Title[0:5]
#
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
#     #print(info)
#
#     #mydata.loc[i] = info
#
# print(mydata)
#
#
#
#
#
#
