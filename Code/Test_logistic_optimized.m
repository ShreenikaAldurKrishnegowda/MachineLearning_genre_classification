disp('This code demonstrates the testing phase using pre-trained models.');

file_path = 'final_dataset.csv';
data = readtable(file_path);


% extraction of predicters
X = data{:, {'acousticness', 'danceability', 'energy', 'instrumentalness', 'liveness', 'loudness', 'speechiness', 'tempo', 'valence'}};
y = categorical(data.broadgenre);

%creating training and test set
rng(42);
cv = cvpartition(y, 'Holdout', 0.2);
log_x_train = X(training(cv), :);
log_y_train = y(training(cv));
log_x_temp = X(test(cv), :);
log_y_temp = y(test(cv));

% splitting temporary set into testing and validation
cv_temp = cvpartition(log_y_temp, 'Holdout', 0.5);
log_x_val = log_x_temp(training(cv_temp), :);
log_y_val = log_y_temp(training(cv_temp));
log_x_test = log_x_temp(test(cv_temp), :);
log_y_test = log_y_temp(test(cv_temp));

% feature scaling
log_x_train = zscore(log_x_train);
log_x_val = zscore(log_x_val);
log_x_test = zscore(log_x_test);

%Loading the trained model
loaded_model = load('logistic_optimized_model.mat');
logistic_optimized = loaded_model.best_model;

%predicting the best model against test set
optimized_prediction = predict(logistic_optimized, log_x_test);

%calculating the final accuracy
optimized_accuracy = sum(optimized_prediction == log_y_test) / numel(log_y_test);
disp(['the accuracy of optimized logistic regression is: ', num2str(optimized_accuracy)]);
disp('testing phase is completed for optimized logistic regression');


% --------------- Visualise --------- %

% -- confusion matrix --- %
figure;
confusion_mat = confusionchart(log_y_test, optimized_prediction, 'RowSummary', 'row-normalized', 'ColumnSummary', 'column-normalized');
confusion_mat.Title = 'confusion matrix logistic regression';
conf = confusionmat(log_y_test, optimized_prediction);
disp(conf);

% ------ ROC Curve ---- %

[~, scores] = predict(logistic_optimized, log_x_test);

figure;
for i = 1:numel(logistic_optimized.ClassNames)
    [X, Y, ~, AUC] = perfcurve(log_y_test, scores(:, i), logistic_optimized.ClassNames(i));
    fprintf('Class: %s (AUC = %.4f)\n', char(logistic_optimized.ClassNames(i)), AUC);
    plot(X, Y, 'LineWidth', 1.5, 'DisplayName', ['Class: ', char(logistic_optimized.ClassNames(i)), ' (AUC = ', num2str(AUC), ')']);
    hold on;
end

title('ROC Curve - Logistic Regression');
xlabel('false Positive Rate');
ylabel('true Positive Rate');
legend('show');
hold off;