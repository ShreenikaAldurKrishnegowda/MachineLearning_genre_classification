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


%Loading the trained model
loaded_model = load('random_model.mat');
random = loaded_model.random_model;

%predicting against the test set
random_predict = predict(random, random_x_test);

random_predict = categorical(random_predict);

% calculating the accuracy of the model
random_accuracy = sum(random_predict == random_y_test) / numel(random_y_test);
disp(['the accuracy of random forest is :', num2str(random_accuracy)]);
%testing phase for random forest is completed

