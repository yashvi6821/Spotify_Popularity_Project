Spotify Song Popularity Analysis & Prediction

Project Overview

This project analyzes Spotify song features and predicts song popularity using Machine Learning techniques.

Objectives

Analyze Spotify song features.
Visualize song popularity using graphs.
Train and compare Machine Learning models.
Identify important features for predicting song popularity.

Technologies Used

Python
Pandas
NumPy
Matplotlib
Seaborn
Scikit-learn

Machine Learning Models

Linear Regression
Random Forest Regressor

Dataset

The dataset contains Spotify song information and audio features such as danceability, energy, loudness, acousticness, tempo, duration, and popularity.

Model Evaluation Results

|           Model         |  MAE  |  RMSE | R² Score |
|-------------------------|------:|------:|------:|
|   Linear Regression     | 19.97 | 23.96 | 0.065 |
| Random Forest Regressor | 16.52 | 21.16 | 0.271 |
|   Tuned Random Forest   | 17.00 | 21.28 | 0.262 |

The original Random Forest model performed better than Linear Regression and the tuned Random Forest model in the current evaluation.

Conclusion

This project applies data analysis and Machine Learning techniques to study Spotify song popularity. Among the evaluated models, the original Random Forest Regressor achieved the best results. However, the R² score indicates that the models explain only a limited portion of the variation in song popularity, so further improvement is possible.
