import pandas as pd
from pytube import YouTube
from youtubesearchpython import VideosSearch

data = pd.read_csv("data/out/cinemaFeatures.csv")
for i in range(len(data)):
    movie = data.Title[i].strip()
    movie_file = movie.replace(" ", "_")
    videoSearch = VideosSearch(f"{movie} film trailer", limit=1)
    videoSearchRes = videoSearch.result()
    videoResList = videoSearchRes["result"]
    videoResDict = videoResList[0]
    videoURL = videoResDict["link"]

    try:
        # object creation using YouTube
        yt = YouTube(videoURL)
    except:
        print(f"Connection Error - cant create connection for film {movie}")  # to handle exception

    # download video as mp4 and set filename
    try:
        yt.streams.filter(progressive=True,
                          file_extension="mp4").first().download(output_path="data\download",
                                                                 filename=f"{movie_file}_trailer.mp4")
    except:
        print(f"download error - can't download trailer for film {movie}")
    print(f'{movie} Trailer downloaded')



