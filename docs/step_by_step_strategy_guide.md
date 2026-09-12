# Step-by-Step Strategy Guide

This project now follows a simple object-oriented pattern:

1. `MarketDataLoader`
   - Loads the CSV from the `data` folder.
   - Uses a stable path based on the file location's parent folder.

2. `FeatureEngineer`
   - Creates the ratios `Open/Close` and `High/Low`.
   - Keeps the raw stock-market DataFrame ready for classroom modeling.

3. `StrategyModel`
   - Builds the target signal by comparing the next day close to the current close.
   - Splits the dataset into a training and test section.
   - Trains an `SVC` model and prints an accuracy score.
   - Calculates the strategy return and cumulative return series.

4. `StrategyPlotter`
   - Draws the cumulative returns chart for the strategy.

## Execution Flow

1. Read the `SPY.csv` file from the data folder.
2. Convert the date index into a real pandas datetime index.
3. Add derived dataset features, such as `Open/Close` and `High/Low`.
4. Use the SVM classifier (`SVC`) to fit on train data and test on the remaining rows.
5. Compute the strategy return signal and cumulative return curve.
6. Plot the strategy curve.

## Why this structure helps

- `MarketDataLoader` isolates file and path handling.
- `FeatureEngineer` isolates feature creation.
- `StrategyModel` isolates training, scoring, and return logic.
- `StrategyPlotter` isolates plotting.

This keeps the script easier to read, easier to test, and easier to extend in future projects.
