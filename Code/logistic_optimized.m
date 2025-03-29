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

% we trained two logistic regression models with and without
% hyperparametres


%model without hyperparameters
model = fitcecoc(log_x_train, log_y_train, 'Learners', 'linear', 'Coding', 'onevsall', 'ClassNames', unique(log_y_train));



%model with hyperparameter optimization
options = struct('Optimizer', 'bayesopt', 'MaxTime', 5, 'ShowPlots', true, 'Verbose', 1);
optimized_model = fitcecoc(log_x_train, log_y_train, 'Learners', 'linear', 'Coding', 'onevsall', 'OptimizeHyperparameters', 'auto', 'HyperparameterOptimizationOptions', options, 'ClassNames', unique(log_y_train));


% calculating prediction against validation set
pred_model = predict(model, log_x_val);%model without hyperparameters
pred_optimized_model = predict(optimized_model, log_x_val); %model with hyperparameter optimization

% calculating accuracy against validation set
accuracy_model = sum(pred_model == log_y_val) / numel(log_y_val);%model without hyperparameters
accuracy_optimized_model = sum(pred_optimized_model == log_y_val) / numel(log_y_val);%model with hyperparameter optimization

disp(['accuracy of model without hyperparameters is : ', num2str(accuracy_model)]);
disp(['accuracy of model with hyperparameters is: ', num2str(accuracy_optimized_model)]);

% choosing the best model
if accuracy_model > accuracy_optimized_model
    best_model = model;
else
    best_model = optimized_model;
end

save('logistic_optimized_model.mat ', 'best_model');

disp('Training phase completed,model saved');


