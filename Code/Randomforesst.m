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

%random forest without parameter optimization
numTrees = 100; 

random_model = TreeBagger(numTrees, random_x_train, random_y_train, 'Method', 'classification');

save('random_model.mat', 'random_model');

disp('training phase completed,model saved');
