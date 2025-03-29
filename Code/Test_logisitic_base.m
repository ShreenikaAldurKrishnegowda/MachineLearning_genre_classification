%This code demonstrates the testing phase

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
log_X_test = X(test(cv), :);
log_y_test = y(test(cv));



%Loading the trained model
model = load('logistic_base_model.mat');
saved_mdl = model.logistic_model;
%predicting against test case
y_pred = predict(saved_mdl, log_X_test);


%calculating accuracy of logistic regression model
accuracy_test = sum(y_pred == log_y_test) / numel(log_y_test);
disp(['accuracy of logistic regression model is : ', num2str(accuracy_test)]);
disp('testing phase is completed for logistic regression');
