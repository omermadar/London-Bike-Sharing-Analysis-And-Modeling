import numpy as np
import data
import cross_validation
import evaluation
import knn

df = data.load_data('london_sample_2500.csv')
folds = data.get_folds()

y_list = np.array(df['season'])
y_list = data.adjust_labels(y_list)
x_list = np.array(df[['t1', 't2', 'wind_speed', 'hum']])
x_list = data.add_noise(x_list)


k_list = [3, 5, 11, 25, 51, 75, 101]

mean_scores_classification = []

print("Part1 - Classification")
for k in k_list:
    classifier = knn.ClassificationKNN(k)
    scores = cross_validation.cross_validation_scores(classifier, x_list, y_list, folds, evaluation.f1_score)
    mean_score = np.mean(scores)
    std_score = np.std(scores, ddof=1)
    mean_scores_classification.append(mean_score)
    print(f"k={k}, mean score: {mean_score:.4f}, std of scores: {std_score:.4f}" )

evaluation.visualize_results(k_list, mean_scores_classification, "f1 score",
                             "Classification", "plot_classification.png")

print()

y_list_new = np.array(df['hum'])
x_list_new = np.array(df[['t1', 't2', 'wind_speed']])
x_list_new = data.add_noise(x_list_new)

mean_scores_regression = []

print("Part2 - Regression")
for k in k_list:
    classifier = knn.RegressionKNN(k)
    scores = cross_validation.cross_validation_scores(classifier, x_list_new, y_list_new, folds, evaluation.rmse)
    mean_score = np.mean(scores)
    std_score = np.std(scores, ddof=1)
    mean_scores_regression.append(mean_score)
    print(f"k={k}, mean score: {mean_score:.4f}, std of scores: {std_score:.4f}" )

evaluation.visualize_results(k_list, mean_scores_regression, "rmse",
                             "Regression", "plot_regression.png")
