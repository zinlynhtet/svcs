from pathlib import Path

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('seaborn-v0_8-darkgrid')
import warnings
warnings.filterwarnings('ignore')
import yfinance as yf

DATA_DIR = Path(__file__).resolve().parent / 'data'
df = pd.read_csv(DATA_DIR / 'SPY.csv', index_col=0, parse_dates=True)
df.index = pd.to_datetime(df.index)
print(df.head())

df['Open/Close'] = df.Open / df.Close
df['High/Low'] = df.High / df.Low
X = df[['Open/Close', 'High/Low']]
X.head()

y = np.where(df['Close'].shift(-1) > df['Close'], 1, -1)
split_percent = 0.8
split = int(split_percent * len(df))
X_train = X[:split]
X_test = X[split:]
y_train = y[:split]
y_test = y[split:]
cls = SVC().fit(X_train, y_train)
y_pred = cls.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

df['Predicted_Signal'] = cls.predict(X)
df = df[split:]
df['Return'] = df.Close.pct_change()
df['Strategy_Return'] = df.Return * df.Predicted_Signal.shift(1)
df['Cumulative_Market_Returns'] = (1 + df['Strategy_Return']).cumprod().shift().fillna(1)
plt.title("Cumulative Returns Plot", fontsize=16)
plt.ylabel("Cumulative Returns")
plt.xlabel("Date")
df['Cumulative_Market_Returns'].plot(figsize=(15, 7),color='g')

plt.show()