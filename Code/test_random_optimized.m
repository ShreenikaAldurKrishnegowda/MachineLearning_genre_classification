file_path = 'final_dataset.csv';
data = readtable(file_path);

% extraction of predicters
X = data{:, {'acousticness', 'danceability', 'energy', 'instrumentalness', 'liveness', 'loudness', 'speechiness', 'tempo', 'valence'}};
y = categorical(data.broadgenre);

%creating training and test set
rng(42);
cv = cvpartition(y, 'Holdout', 0.2);
random_x_train = X(training(cv), :);
random_y_train = y(training(cv));
random_x_test = X(test(cv), :);
random_y_test = y(test(cv));

loaded_model = load('Random_optimized_model.mat');
random_optimized = loaded_model.rand_forest_optimized;

%predicting against test set
rand_optimized_prediction = predict(random_optimized, random_x_test);

rand_optimized_prediction = categorical(rand_optimized_prediction);

%calculating the final accuracy
rand_optimized_accuracy = sum(rand_optimized_prediction == random_y_test) / numel(random_y_test);
disp(['accuracy of optimized random forest: ', num2str(rand_optimized_accuracy)]);

% ----------------- Visualise ------------------ %



% ---confusion matrix --- %

figure;
confusion_mat = confusionchart(random_y_test, rand_optimized_prediction, 'RowSummary', 'row-normalized', 'ColumnSummary', 'column-normalized');
confusion_mat.Title = 'confusion matrix for random forest';
conf = confusionmat(random_y_test, rand_optimized_prediction);
disp(conf);


% ---- feature important --- %

feature_importance = predictorImportance(loaded_model.rand_forest_optimized);

disp('feature importance scores:');
disp('--------------------------');
for i = 1:numel(feature_importance)
    fprintf('%s: %.4f\n', char(loaded_model.rand_forest_optimized.PredictorNames(i)), feature_importance(i));
end

% Visualize feature importance
figure;
bar(feature_importance);
xlabel('feature');
ylabel('importance Score');
title('Feature Importance');
xticklabels({'acousticness', 'danceability', 'energy', 'instrumentalness', 'liveness', 'loudness', 'speechiness', 'tempo', 'valence'});

% ------- ROC Curve -------- %

[~, scores] = predict(random_optimized, random_x_test);

figure;
for i = 1:numel(random_optimized.ClassNames)
    [X, Y, ~, AUC] = perfcurve(random_y_test, scores(:, i), random_optimized.ClassNames(i));
    fprintf('Class: %s (AUC = %.4f)\n', char(random_optimized.ClassNames(i)), AUC);
    plot(X, Y, 'LineWidth', 1.5, 'DisplayName', ['Class: ', char(random_optimized.ClassNames(i)), ' (AUC = ', num2str(AUC), ')']);
    hold on;
end

title('ROC Curve - Random Forest');
xlabel('false Positive Rate');
ylabel('true Positive Rate');
legend('show');
hold off;
