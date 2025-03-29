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

% logistic regression model training
logistic_model = fitmnr(log_x_train, log_y_train);
%, 'Learners', 'linear', 'Coding', 'onevsall');
%logistic_model = fitmulticlass(log_x_train, log_y_train, 'Learners', 'logistic');

%saving the trained model
save('logistic_base_model.mat', 'logistic_model');

disp('training phase completed,model saved');

