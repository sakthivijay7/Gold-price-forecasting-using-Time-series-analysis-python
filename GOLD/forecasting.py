import pandas as pd
import joblib
from datetime import timedelta
import matplotlib.pyplot as plt

with open("D:/ALL/Gold price/Trained_models.pkl","rb") as file:
    loaded_model=joblib.load(file)
date=pd.to_datetime(input("Starting_Date:"))
periods=int(input("Periods:"))

future_date=pd.date_range(start=date,periods=periods)
future_df=pd.DataFrame({"Date":future_date,
                        "year":future_date.year,
                        "month":future_date.month,
                        "day":future_date.day,
                        "weekday":future_date.dayofweek,
                        "is_weekend":(future_date.dayofweek >=5).astype(int),
                        "week":future_date.isocalendar().week,
                        "is_leap_year":(future_date.is_leap_year).astype(int)})
features=future_df[["year","month","day","weekday", "is_weekend","week","is_leap_year"]]
future_df["Forecast_Price XAU/USD"]=loaded_model["XGBRegressor"].predict(features)      

future_df.to_csv("Forecasting.csv",index=False)

plt.figure(figsize=(12,5))
plt.plot(future_df["Date"],future_df["Forecast_Price XAU/USD"],marker="o",color="gold")
plt.title("Forecasting Gold price")
plt.xlabel("DATE")
plt.ylabel("Price XAU/USD Troy Ounce")
plt.grid(True)
plt.tight_layout()
plt.savefig("Gold_forecast.png")
plt.show()