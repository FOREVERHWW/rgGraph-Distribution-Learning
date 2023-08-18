%IISLLDDEMO	The example of IISLLD algorithm.
%
%	Description
%   We establish a maximum entropy model and use IIS algorithm to estimate
%   the parameters. In this way, we can get our LDL model. Then a new 
%   distribution can be predicted based on this model.
% 
%	See also
%	LLDPREDICT, IISLLDTRAIN
%	
%   Copyright: Xin Geng (xgeng@seu.edu.cn)
%   School of Computer Science and Engineering, Southeast University
%   Nanjing 211189, P.R.China
%
clear;
clc;
% Load the data set.
load acm50;

% Initialize the model parameters.
para.minValue = 1e-7; % the feature value to replace 0, default: 1e-7
para.iter = 500; % learning iterations, default: 50 / 200 
para.minDiff = 1e-3; % minimum log-likelihood difference for convergence, default: 1e-7
para.regfactor = 0; % regularization factor, default: 0

metrics = [{'chebyshev'}, {'clark'}, {'canberra'}, {'kldist'}, {'cosine'}, {'intersection'}];
exp_avg_metrics = dictionary(metrics, 0);

num_runs = 1; % number of different training and test maks to average results over

for n = 1:num_runs
    tic;
    train_features = features(train_masks(n, :), :);
    train_labels = labels(train_masks(n, :), :);

    val_features = features(val_masks(n, :), :);
    val_labels = labels(val_masks(n, :), :);
   
    test_features = features(test_masks(n, :), :);
    test_labels = labels(test_masks(n, :), :);

    % The training part of IISLLD algorithm.
    [weights] = iislldTrain(para, train_features, train_labels, val_features, val_labels);
    fprintf('Training time of IIS-LLD: %8.7f \n', toc);
    
    % Prediction
    preDistribution = lldPredict(weights,test_features);
    fprintf('Finish prediction of IIS-LLD. \n');
    
    % find the average metrics over the test set
    avg_metrics = dictionary;
    [disName, distance] = computeMeasures(test_labels(1,:), preDistribution(1,:));
    save("yelp2_50_second_results", "test_labels", "preDistribution") 
    avg_metrics(disName) = distance;
    for i=2:size(test_features,1)
        % show the comparisons between the predicted distribution
        [disName, distance] = computeMeasures(test_labels(i,:), preDistribution(i,:));
        avg_metrics(disName) = avg_metrics(disName) + distance;
    end

    % average out the metrics
    for k = keys(avg_metrics)
        avg_metrics(k) = avg_metrics(k) / size(test_features,1);
    end
    
    for k = keys(avg_metrics)
        exp_avg_metrics(k) = exp_avg_metrics(k) + avg_metrics(k);
    end
end

%average out the metrics
for k = keys(exp_avg_metrics)
    exp_avg_metrics(k) = exp_avg_metrics(k) / num_runs;
end

entries(exp_avg_metrics)

% To visualize two distribution and display some selected metrics of distance
% for i=1:5
%     % Show the comparisons between the predicted distribution
% 	[disName, distance] = computeMeasures(test_labels(i,:), preDistribution(i,:));
%     % Draw the picture of the real and prediced distribution.
%     drawDistribution(test_labels(i,:),preDistribution(i,:),disName, distance);
%     %sign=input('Press any key to continue:');
% end
