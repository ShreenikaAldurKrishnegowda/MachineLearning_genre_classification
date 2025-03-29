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

%hyperparameter optimization using cross validation
num_of_trees = [50, 100, 150];%number of trees
min_leaf_size = [1, 5, 10];%minimum number of observations


t = templateTree('Reproducible', true);%making the tree reproducable

optimizing_parameters = struct('NumLearningCycles', num_of_trees, 'MinLeafSize', min_leaf_size);
%training the model
rand_forest_optimized = fitcensemble(random_x_train, random_y_train, 'Method', 'bag', 'Learners', t, 'OptimizeHyperparameters', 'auto', 'HyperparameterOptimizationOptions', struct('MaxObjectiveEvaluations', 50, 'ShowPlots', true));


save('Random_optimized_model.mat', 'rand_forest_optimized');

disp('training phase is completed,model saved');

