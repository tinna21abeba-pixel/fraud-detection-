import yfinance as yf
import matplotlib.pyplot as plt
import talib

data= yf.download("AAPL", start="2020-01-01", end="2026-3-30")
print(data.head())

data["Close"].plot(figsize=(10,5))
plt.title("AAPL Closing Price")
plt.show()

#find average closing price for 365 days
data["MA365"]=data["Close"].rolling(window=365).mean()
data[["Close", "MA365"]].plot(figsize=(10, 5))  
plt.title("Close Price and one year Moving Average")
plt.show()    
 # RSI(relative strength index)
 