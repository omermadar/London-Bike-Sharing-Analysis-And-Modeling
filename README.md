# London-Bike-Sharing-Analysis-And-Modeling

This comprehensive data science project explores the London bike-sharing dataset through several key stages:

1. Data Preparation and Exploration: The project begins with data cleaning and feature engineering to create new variables like seasonal names and weekend indicators. An exploratory data analysis (EDA) is then conducted to calculate descriptive statistics and analyze the correlation between features, such as the relationship between bike count and various weather conditions.

2. Unsupervised Learning with K-Means: The project implements the K-Means clustering algorithm from scratch to segment the data. It focuses on the cnt (bike count) and t1 (real temperature) features, applying min-max scaling and noise to prepare the data for clustering. The results for different numbers of clusters (k=2,3,5) are visualized to demonstrate how the algorithm groups similar data points.

3. Supervised Learning with K-Nearest Neighbors (KNN): To predict outcomes, the project builds a K-Nearest Neighbors (KNN) model, implemented from the ground up, capable of both classification and regression. The model's performance is rigorously evaluated using K-fold cross-validation across a range of k values. The classification task predicts the season using an f1-score metric, while the regression task predicts humidity with the Root Mean Square Error (RMSE) metric. The final visualizations illustrate how the model's accuracy changes as k is adjusted.
