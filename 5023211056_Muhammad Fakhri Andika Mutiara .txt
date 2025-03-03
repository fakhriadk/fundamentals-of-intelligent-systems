% Load MNIST data from image file
A = imread("MnistExamples.png");
AGray = mean(A,3); % Convert to grayscale

% Define the dimensions and spacing of the digit images in the file
firstRow = 12; firstCol = 28; lastRow = 38; lastCol = 55;
dataRowNum = 28; dataColNum = 27;
xSpace = 8*ones(1,16); xSpace([3,7,11,15]) = xSpace([3,7,11,15])+1;
ySpace = 5*ones(1,10); ySpace(5) = ySpace(5)+1;

totalDigitImages = [];

% Extract digit images
for row = 1:10
    digitRowImages = [];
    for col = 1:16
        digitImage = AGray(firstRow:lastRow, firstCol:lastCol);
        firstCol = firstCol + xSpace(col) + dataColNum;
        lastCol = lastCol + xSpace(col) + dataColNum;
        digitRowImages = cat(3, digitRowImages, digitImage);
    end
    totalDigitImages = cat(4, totalDigitImages, digitRowImages);
    firstRow = firstRow + ySpace(row) + dataRowNum;
    firstCol = 28; lastRow = lastRow + ySpace(row) + dataRowNum; lastCol = 55;
end

% Prepare data for training
numDigits = size(totalDigitImages, 4);
numExamples = size(totalDigitImages, 3) * numDigits;
data = reshape(totalDigitImages, [dataRowNum*dataColNum, numExamples]);
data = double(data) / 255; % Normalize data to [0, 1]
labels = repmat(0:9, [size(totalDigitImages, 3), 1]);
labels = labels(:);

% Split data into training and test sets
trainRatio = 0.8;
numTrainExamples = round(numExamples * trainRatio);
numTestExamples = numExamples - numTrainExamples;
randIndices = randperm(numExamples);
trainData = data(:, randIndices);
trainLabels = labels(randIndices);
% trainData = data(:, randIndices(1:numTrainExamples));
% trainLabels = labels(randIndices(1:numTrainExamples));

testData = data(:, randIndices(numTrainExamples+1:end));
testLabels = labels(randIndices(numTrainExamples+1:end));


% Convert labels to one-hot encoding
trainLabels_one_hot = full(ind2vec(trainLabels' + 1));
testLabels_one_hot = full(ind2vec(testLabels' + 1));

% ANN architecture
inputSize = dataRowNum * dataColNum;
hiddenLayerSize = [128, 64, 10];
outputSize = 10;

% Initialize weights and biases
weights = {
    rand(hiddenLayerSize(1), inputSize) - 0.5, 
    rand(hiddenLayerSize(2), hiddenLayerSize(1)) - 0.5, 
    rand(outputSize, hiddenLayerSize(2)) - 0.5
};
biases = {
    rand(hiddenLayerSize(1), 1) - 0.5, 
    rand(hiddenLayerSize(2), 1) - 0.5, 
    rand(outputSize, 1) - 0.5
};

% Training parameters
learningRate = 0.01;
epochs = 5000;

% Activation function and its derivative
sigmoid = @(z) 1.0 ./ (1.0 + exp(-z));
sigmoid_prime = @(z) sigmoid(z) .* (1 - sigmoid(z));

% Loss function
cross_entropy_loss = @(y, y_hat) -sum(y .* log(y_hat) + (1 - y) .* log(1 - y_hat));

% Training loop
trainLosses = [];
tic;
for epoch = 1:epochs
    idx = randperm(numTrainExamples);
    shuffledData = trainData(:, idx);
    shuffledLabels = trainLabels_one_hot(:, idx);
    
    total_loss = 0;
    
    for i = 1:numTrainExamples
        % Forward propagation
        a1 = sigmoid(weights{1} * shuffledData(:, i) + biases{1});
        a2 = sigmoid(weights{2} * a1 + biases{2});
        output = sigmoid(weights{3} * a2 + biases{3});
        
        % Calculate error (loss)
        loss = cross_entropy_loss(shuffledLabels(:, i), output);
        total_loss = total_loss + loss;
        
        % Backpropagation
        delta3 = output - shuffledLabels(:, i);
        delta2 = (weights{3}' * delta3) .* sigmoid_prime(weights{2} * a1 + biases{2});
        delta1 = (weights{2}' * delta2) .* sigmoid_prime(weights{1} * shuffledData(:, i) + biases{1});
        
        % Update weights and biases
        weights{3} = weights{3} - learningRate * delta3 * a2';
        biases{3} = biases{3} - learningRate * delta3;
        
        weights{2} = weights{2} - learningRate * delta2 * a1';
        biases{2} = biases{2} - learningRate * delta2;
        
        weights{1} = weights{1} - learningRate * delta1 * shuffledData(:, i)';
        biases{1} = biases{1} - learningRate * delta1;
    end
    
    if mod(epoch, 100) == 0
        trainLosses = [trainLosses, total_loss / numTrainExamples];
        disp(['Epoch ', num2str(epoch), ': Loss = ', num2str(total_loss / numTrainExamples)]);
    end
end
trainingTime = toc;
disp(['Training time: ', num2str(trainingTime), ' seconds.']);

% Evaluation on test set
correct_predictions = 0;
confusionMatrix = zeros(10, 10);
for i = 1:numTestExamples
    a1 = sigmoid(weights{1} * testData(:, i) + biases{1});
    a2 = sigmoid(weights{2} * a1 + biases{2});
    output = sigmoid(weights{3} * a2 + biases{3});
    
    [~, predicted_class] = max(output);
    actual_class = testLabels(i) + 1;
    confusionMatrix(actual_class, predicted_class) = confusionMatrix(actual_class, predicted_class) + 1;
    
    if predicted_class == actual_class
        correct_predictions = correct_predictions + 1;
    end
end

accuracy = correct_predictions / numTestExamples;
disp(['Test set accuracy: ', num2str(accuracy)]);

% Normalize confusion matrix to get percentages
confusionMatrixPercent = 100 * bsxfun(@rdivide, confusionMatrix, sum(confusionMatrix,2));

% Plotting training loss over epochs
figure;
plot(trainLosses, 'LineWidth', 2);
title('Convergence Graph');
xlabel('Epoch');
ylabel('Loss');
grid on;

% Plotting the confusion matrix
figure;
imagesc(confusionMatrixPercent);
colorbar;
title('Confusion Matrix');
xlabel('Predicted class');
ylabel('True class');
axis square;
% Set the ticks and labels for both axes from 0 to 9
set(gca, 'XTick', 1:10, 'YTick', 1:10, 'XTickLabel', 0:9, 'YTickLabel', 0:9);