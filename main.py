import pandas as pd

# Load Spotify dataset
df = pd.read_csv("dataset/spotify_songs.csv")

print("Dataset loaded successfully!")
print(df.head())
print("\nDataset shape:", df.shape)

print("\nColumn names:")
print(df.columns.tolist())

print("\nDataset information:")
print(df.info())

print("\nMissing values:")
print(df.isnull().sum())

print("\nDuplicate rows:")
print(df.duplicated().sum())

df = df.drop_duplicates()

print("\nAfter removing duplicates:")
print("New dataset shape:", df.shape)

print("\nPopularity Analysis:")
print("Average popularity:", df["track_popularity"].mean())
print("Minimum popularity:", df["track_popularity"].min())
print("Maximum popularity:", df["track_popularity"].max())
