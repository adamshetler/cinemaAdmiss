import matplotlib.pyplot as plt
import pandas as pd
import requests
import os
from tmdbv3api import TMDb
tmdb = TMDb()
tmdb.api_key = 'e36c3208e2664800c506328a3303432b'
tmdb.language = 'en'
tmdb.debug = True
from tmdbv3api import Movie

# import cinema data
df = pd.read_excel('data/in/cinemaGrossSummary.xlsx')
df = df.rename(columns={"Film Title": "Title", "Avg Per Show": "AvgPerShow$", "Net": "Net$", "Gross": "Gross$"})
# # add calculated column for number of showings
df = df.assign(NumberShows=df['Gross$'] / df['AvgPerShow$'])

# # create plots showing overall numbers (include all movies)
df["NumberShows"].hist(bins=50)
df2 = df.loc[:, ["AvgPerShow$", "Net$", "Gross$", "Admissions"]]
df2.hist(bins=50)
plt.show()
color = {
    "boxes": "DarkGreen",
    "whiskers": "DarkOrange",
    "medians": "DarkBlue",
    "caps": "Gray",
}
df.loc[:, ["Net$", "Gross$"]].plot.box(color=color, sym="r+")
plt.show()
df["AvgPerShow$"].plot.box(color=color, sym="r+")
plt.show()
df["Admissions"].plot.box(color=color, sym="r+")
plt.show()
df["NumberShows"].plot.box(color=color, sym="r+")
plt.show()


# use api to pull feature data on each film
# genre_dict = {
#     'title':df["Title"],
#     'genre1':[],
#     'genre2':[]
# }
# genre_lookup = pd.dataframe(genre_dict)
df["Admissions"] = pd.to_numeric(df["Admissions"])
df = df[df["Admissions"]>10]
df = df.reset_index(drop=True)
#df["Title"] = df["Title"].str.replace(r"[\"\',]", '')
for i in range(len(df)):
    movie = Movie()
    title = df.loc[i, "Title"]
    search = movie.search(title)
    for res in search:
        title_id = res.id
        BASE_URL = "https://api.themoviedb.org/3/movie/" + str(res.id)
        endpoint = f"{BASE_URL}?api_key={'e36c3208e2664800c506328a3303432b'}"
        r = requests.get(endpoint)
        data = r.json()
        df.loc[i, "vote_average"] = data["vote_average"]
        df.loc[i, "popularity"] = data["popularity"]
        df.loc[i, "adult"] = data["adult"]
        df.loc[i, "release_date"] = data["release_date"]
        df.loc[i, "original_language"] = data["original_language"]
        df.loc[i, "revenue"] = data["revenue"]
        df.loc[i, "runtime"] = data["runtime"]
        df.loc[i, "budget"] = data["budget"]
        #df[i, "genres"] = data["genres"]
        #df[i, "production_companies"] = data["production_companies]
        #df[i, "production_countries"] = data["production_countries"]
        #df[i, "tagline"] = data["tagline"]
        #df[i, "overview"] = data["overview"]
        #df[i, "poster_path"] = data["poster_path"]
        #df[i, "homepage] = data["homepage"]
    print(str(i) + ": " + title + " Complete")

df.to_csv("data/out/cinemaFeatures.csv")




