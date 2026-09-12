from pathlib import Path
import json

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.style.use('seaborn-v0_8-darkgrid')
import warnings

warnings.filterwarnings('ignore')


class MarketDataLoader:
    """Read market data from disk and return a clean pandas DataFrame."""

    def __init__(self, data_dir=None, file_name='SPY.csv'):
        self.data_dir = Path(data_dir) if data_dir else Path(__file__).resolve().parent / 'data'
        self.file_name = file_name

    def load(self):
        data_path = self.data_dir / self.file_name
        if not data_path.exists():
            raise FileNotFoundError(f"Market data file not found: {data_path}")

        df = pd.read_csv(data_path, index_col=0, parse_dates=True)
        df.index = pd.to_datetime(df.index)
        return df


class FeatureEngineer:
    """Create technical signal features used by the strategy model."""

    def __init__(self, df):
        self.df = df.copy()

    def build_features(self):
        required_columns = {'Open', 'Close', 'High', 'Low'}
        missing = required_columns.difference(self.df.columns)
        if missing:
            raise ValueError(f"Missing required columns for feature engineering: {sorted(missing)}")

        self.df['Open/Close'] = self.df['Open'] / self.df['Close']
        self.df['High/Low'] = self.df['High'] / self.df['Low']
        return self.df


class StrategyModel:
    """Train an SVM strategy model and compute the strategy return series."""

    def __init__(self, df, split_percent=0.8, model=None):
        self.df = df.copy()
        self.split_percent = split_percent
        self.model = model or SVC()
        self.accuracy = None
        self.model_trained = None

    def set_target(self):
        self.df['Target'] = np.where(self.df['Close'].shift(-1) > self.df['Close'], 1, -1)

    def train_and_score(self):
        self.set_target()

        X = self.df[['Open/Close', 'High/Low']]
        y = self.df['Target']

        split = int(self.split_percent * len(self.df))
        X_train = X[:split]
        X_test = X[split:]
        y_train = y[:split]
        y_test = y[split:]

        self.model_trained = self.model.fit(X_train, y_train)
        y_pred = self.model_trained.predict(X_test)
        self.accuracy = accuracy_score(y_test, y_pred)
        print(f"Accuracy: {self.accuracy * 100:.2f}%")

        self.df['Predicted_Signal'] = self.model_trained.predict(X)
        self.df['Return'] = self.df['Close'].pct_change()
        self.df['Strategy_Return'] = self.df['Return'] * self.df['Predicted_Signal'].shift(1)
        self.df['Cumulative_Market_Returns'] = (1 + self.df['Strategy_Return']).cumprod().shift().fillna(1)

        return self.model_trained, self.accuracy


class StrategyPlotter:
    """Create a simple plot of the cumulative strategy return series."""

    def __init__(self, df):
        self.df = df

    def plot(self):
        plt.figure(figsize=(15, 7))
        plt.title("Cumulative Returns Plot", fontsize=16)
        plt.ylabel("Cumulative Returns")
        plt.xlabel("Date")
        self.df['Cumulative_Market_Returns'].plot(color='g')
        plt.tight_layout()
        plt.show()


class StrategyArtifactWriter:
    """Save model information and output files to the workspace artifacts directory."""

    def __init__(self, output_dir='artifacts'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def write_metrics(self, accuracy):
        metrics = {
            'accuracy': float(accuracy),
            'model': 'SVC',
            'features': ['Open/Close', 'High/Low'],
        }
        metrics_path = self.output_dir / 'strategy_metrics.json'
        metrics_path.write_text(json.dumps(metrics, indent=2), encoding='utf-8')

    def write_strategy_data(self, df):
        output_path = self.output_dir / 'strategy_returns.csv'
        df[['Return', 'Strategy_Return', 'Cumulative_Market_Returns']].to_csv(output_path)


class StrategyPipeline:
    """End-to-end object-oriented workflow for loading, feature engineering, training and writing outputs."""

    def __init__(self, data_dir=None, file_name='SPY.csv', output_dir='artifacts'):
        self.loader = MarketDataLoader(data_dir=data_dir, file_name=file_name)
        self.writer = StrategyArtifactWriter(output_dir=output_dir)

    def run(self, split_percent=0.8):
        df = self.loader.load()
        print(df.head())

        engineered = FeatureEngineer(df)
        df = engineered.build_features()

        model = StrategyModel(df, split_percent=split_percent)
        model.train_and_score()

        self.writer.write_metrics(model.accuracy)
        self.writer.write_strategy_data(model.df)

        plotter = StrategyPlotter(model.df)
        plotter.plot()

        return model.df


if __name__ == '__main__':
    pipeline = StrategyPipeline()
    pipeline.run()
