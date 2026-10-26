import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_theme(style="darkgrid")
import matplotlib
matplotlib.rcParams["figure.max_open_warning"] = 0

SPOTIFY_GREEN = "#1DB954"
SPOTIFY_DARK = "#191414"
SPOTIFY_LIGHT = "#B3B3B3"

df = pd.read_csv("dataset/spotify_songs.csv")

print("First 5 Rows: ")
print(df.head())

print("\nDataset shape: ")
print(df.shape)

print("\nColumn Names: ")
print(df.columns)

print("\nDataset info: ")
df.info()

print("\nMissing Values: ")
print(df.isnull().sum())

df = df.dropna()

print("\nAfter Cleaning:")
print(df.shape)

print("\nMissing Values After Cleaning:")
print(df.isnull().sum())

# Check duplicate rows
print("\nDuplicate Rows:")
print(df.duplicated().sum())

# Basic statistics of song popularity
print("\nPopularity Statistics:")
print(df["track_popularity"].describe())

# Popularity Distribution
print("\nPopularity Skewness:", round(df["track_popularity"].skew(), 3))
print("Songs with popularity = 0:", (df["track_popularity"] == 0).sum())

plt.figure(figsize=(10, 6))
sns.histplot(
    df["track_popularity"],
    bins=30,
    kde=True,
    color=SPOTIFY_GREEN
)
plt.axvline(
    df["track_popularity"].mean(),
    color="#FF6B6B",
    linestyle="--",
    label="Mean"
)
plt.axvline(
    df["track_popularity"].median(),
    color="#FFD166",
    linestyle="--",
    label="Median"
)
plt.xlabel("Track Popularity")
plt.ylabel("Number of Songs")
plt.title("Distribution of Track Popularity")
plt.legend()
plt.show()

plt.figure(figsize=(8, 4))
sns.boxplot(
    x=df["track_popularity"],
    color=SPOTIFY_GREEN
)
plt.title("Boxplot of Track Popularity")
plt.show()

# Popularity categories
bins = [-1, 30, 60, 100]
labels = ["Low (0-30)", "Medium (31-60)", "High (61-100)"]
df["popularity_level"] = pd.cut(df["track_popularity"], bins=bins, labels=labels)
print("\nPopularity Level Counts:")
print(df["popularity_level"].value_counts())

# Top 10 most popular songs
print("\nTop 10 Most Popular Songs:")
print(df[["track_name", "track_artist", "track_popularity"]]
      .sort_values("track_popularity", ascending=False)
      .head(10))

# Relationship between danceability and popularity
print("\nDanceability vs Popularity:")
print(df[["danceability", "track_popularity"]].corr())

# Correlation of numerical features with popularity
print("\nCorrelation with Track Popularity:")

correlation = df.select_dtypes(include="number").corr()["track_popularity"]
print(correlation.sort_values(ascending=False))

# Danceability vs Popularity
plt.scatter(
    df["danceability"],
    df["track_popularity"],
    color=SPOTIFY_GREEN,
    alpha=0.3
)

plt.xlabel("Danceability")
plt.ylabel("Track Popularity")
plt.title("Danceability vs Track Popularity")

plt.show()

# Correlation Heatmap
plt.figure(figsize=(12, 8))

correlation_matrix = df.select_dtypes(include="number").corr()

sns.heatmap(
    correlation_matrix,
    annot=True,
    cmap="Greens",
    fmt=".2f"
)

plt.title("Correlation Heatmap of Spotify Features")
plt.show()

# Top 10 Unique Popular Songs Bar Chart
top_10 = (
    df[["track_name", "track_popularity"]]
    .drop_duplicates(subset="track_name")
    .nlargest(10, "track_popularity")
)

plt.figure(figsize=(10, 6))

plt.bar(
    top_10["track_name"],
    top_10["track_popularity"],
    color=SPOTIFY_GREEN
)

plt.xlabel("Song")
plt.ylabel("Popularity")
plt.title("Top 10 Most Popular Songs")

plt.xticks(rotation=45, ha="right")
plt.tight_layout()

plt.show()

# Inference / Findings
corr = df.select_dtypes(include="number").corr()["track_popularity"].drop("track_popularity")

print("\n========== KEY FINDINGS ==========")
print(f"1. Mean popularity: {df['track_popularity'].mean():.2f}, "
      f"Median: {df['track_popularity'].median():.2f}")
print(f"2. Skewness: {df['track_popularity'].skew():.2f} "
      f"({'right-skewed' if df['track_popularity'].skew() > 0.5 else 'left-skewed' if df['track_popularity'].skew() < -0.5 else 'roughly symmetric'})")
print(f"3. Share of songs with popularity = 0: "
      f"{(df['track_popularity'] == 0).mean() * 100:.1f}%")
print(f"4. Strongest positive correlation: {corr.idxmax()} ({corr.max():.3f})")
print(f"5. Strongest negative correlation: {corr.idxmin()} ({corr.min():.3f})")
print(f"6. Danceability correlation with popularity: {corr['danceability']:.3f}")

# ==========================================
# MACHINE LEARNING: POPULARITY PREDICTION
# ==========================================

import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Select audio features for prediction
features = [
    "danceability",
    "energy",
    "key",
    "loudness",
    "mode",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "valence",
    "tempo",
    "duration_ms"
]

target = "track_popularity"

# Prepare input and output data
X = df[features]
y = df[target]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("\n========== MODEL TRAINING ==========")

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))

# Model 1: Linear Regression
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)

linear_predictions = linear_model.predict(X_test)

linear_mae = mean_absolute_error(y_test, linear_predictions)
linear_rmse = np.sqrt(mean_squared_error(y_test, linear_predictions))
linear_r2 = r2_score(y_test, linear_predictions)

print("\n--- Linear Regression Results ---")
print(f"MAE:  {linear_mae:.2f}")
print(f"RMSE: {linear_rmse:.2f}")
print(f"R2 Score: {linear_r2:.3f}")

# Model 2: Random Forest Regression
random_forest = RandomForestRegressor(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

random_forest.fit(X_train, y_train)

rf_predictions = random_forest.predict(X_test)

rf_mae = mean_absolute_error(y_test, rf_predictions)
rf_rmse = np.sqrt(mean_squared_error(y_test, rf_predictions))
rf_r2 = r2_score(y_test, rf_predictions)

print("\n--- Random Forest Results ---")
print(f"MAE:  {rf_mae:.2f}")
print(f"RMSE: {rf_rmse:.2f}")
print(f"R2 Score: {rf_r2:.3f}")

# Compare both models
print("\n========== MODEL COMPARISON ==========")

if rf_r2 > linear_r2:
    print("Random Forest has a higher R2 score.")
else:
    print("Linear Regression has a higher or equal R2 score.")

# Feature Importance
importance = pd.Series(
    random_forest.feature_importances_,
    index=features
).sort_values(ascending=False)

print("\n========== FEATURE IMPORTANCE ==========")
print(importance)

plt.figure(figsize=(10, 6))

importance.sort_values().plot(
    kind="barh",
    color=SPOTIFY_GREEN
)

plt.xlabel("Importance")
plt.ylabel("Audio Feature")
plt.title("Feature Importance for Spotify Popularity Prediction")
plt.tight_layout()
plt.show()

# Actual vs Predicted Popularity

plt.figure(figsize=(8, 6))

plt.scatter(
    y_test,
    rf_predictions,
    alpha=0.4,
    color=SPOTIFY_GREEN
)

plt.plot(
    [0, 100],
    [0, 100],
    linestyle="--",
    color="red",
    label="Perfect Prediction"
)

plt.xlabel("Actual Popularity")
plt.ylabel("Predicted Popularity")
plt.title("Actual vs Predicted Song Popularity")
plt.legend()
plt.tight_layout()
plt.show()

# Tuned Random Forest Experiment
tuned_forest = RandomForestRegressor(
    n_estimators=150,
    max_depth=20,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)

tuned_forest.fit(X_train, y_train)
tuned_predictions = tuned_forest.predict(X_test)

tuned_mae = mean_absolute_error(y_test, tuned_predictions)
tuned_rmse = np.sqrt(mean_squared_error(y_test, tuned_predictions))
tuned_r2 = r2_score(y_test, tuned_predictions)

print("\n--- Tuned Random Forest Results ---")
print(f"MAE: {tuned_mae:.2f}")
print(f"RMSE: {tuned_rmse:.2f}")
print(f"R2 Score: {tuned_r2:.3f}")
