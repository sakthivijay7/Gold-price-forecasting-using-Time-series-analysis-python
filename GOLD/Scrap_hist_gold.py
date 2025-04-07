import pandas as pd
import requests 
from bs4 import BeautifulSoup

hist_url=f"https://finance.yahoo.com/quote/GC=F/history?period1=0&period2=9999999999&interval=1d&filter=history&frequency=1d"

def hist_gold():
    response=requests.get(hist_url,headers={"User-Agent":"Mozilla/5.0"})
    soup=BeautifulSoup(response.text,"html.parser")

    rows=soup.find_all("tr")
    history_data=[]
    for row in rows:
        cols=row.find_all("td")
        if len(cols)<6:
            continue
        date=cols[0].text.strip()
        closing_price=cols[-2].text.strip()
        history_data.append({"Date":date,"Closing price XAU/USD":closing_price})

    dataset=pd.DataFrame(history_data)
    dataset.to_csv("new_hist_gold.csv",index=False)
    print(dataset.head())

hist_gold()
