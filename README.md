# Gold-price-forecasting-using-Time-series-analysis-python
## DATE:Apr7 -2025
Historical data of closing price of gold XAU/USD to Timeseries forecasting using python 

- **Data Collection:**
Use requests and Beautifulsoup to Yahoo finance link to scrap historical global gold closing XAU/USD  price over the 25 years.

- **Data handle:**
pandas to read the dataset and remove unwanted symbols(doller sign)

- **Model Train:**
Train a model with multiple Regression alogrithmns for comparision.

- **Model save:**
Trained models store in the pickle file.

- **Result:**
Model got `99% R^2 Score`.

- **Forecasting:**
DATETIME to features extraction than trainded model to perform time series forecasting.
![Gold_forecast](https://github.com/user-attachments/assets/446300fa-a548-4c09-ba09-067bafe595e2)


