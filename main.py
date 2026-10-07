import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

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
sns.histplot(df["track_popularity"], bins=30, kde=True)
plt.axvline(df["track_popularity"].mean(), color="red", linestyle="--", label="Mean")
plt.axvline(df["track_popularity"].median(), color="green", linestyle="--", label="Median")
plt.xlabel("Track Popularity")
plt.ylabel("Number of Songs")
plt.title("Distribution of Track Popularity")
plt.legend()
plt.show()

plt.figure(figsize=(8, 4))
sns.boxplot(x=df["track_popularity"])
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
plt.scatter(df["danceability"], df["track_popularity"], alpha=0.3)

plt.xlabel("Danceability")
plt.ylabel("Track Popularity")
plt.title("Danceability vs Track Popularity")

plt.show()

# Correlation Heatmap
plt.figure(figsize=(12, 8))

correlation_matrix = df.select_dtypes(include="number").corr()

sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")

plt.title("Correlation Heatmap of Spotify Features")
plt.show()

# Top 10 Unique Popular Songs Bar Chart
top_10 = (
    df[["track_name", "track_popularity"]]
    .drop_duplicates(subset="track_name")
    .nlargest(10, "track_popularity")
)

plt.figure(figsize=(10, 6))

plt.bar(top_10["track_name"], top_10["track_popularity"])

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

