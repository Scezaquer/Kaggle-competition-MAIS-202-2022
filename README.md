# Kaggle-competition-MAIS-202-2022
Cloth image recognition

# I – Implementation of the model

I chose to use a convolutional neural network for this problem since this is image recognition. However, to boost the performances I implemented the same logic as the random forest algorithm, meaning I have multiple classifiers all making their prediction on the data, and the final prediction is made following a majority vote logic. 
Every model has 5 convolutional layers with 64 filters each and a 3x3 kernel, 1 stride, padding and a relu activation function. There is a 2x2 max-pooling layer between each convolutional layer. Then at the very end of the network we have the classifier, made of 64 perceptron with linear activation, before the output layer of 10 neurons with softmax activation to give us a probability distribution of what the actual thing in the picture is.
We train every model on 5 epochs using the training dataset. The validation is performed on a different part of the dataset for each model, so each one has a slightly different training set, and all the training data is used.
To determine the prediction of each model we take the output with the highest probability. Then we compare it to the prediction of every other model and pick the final answer based on majority vote.

# II – Results

Individual models seem to perform marginally better with more filters per layer and more convolutional layers. However, we are limited by the fact that we are training 10 models each time, making big models not worth the time they take to train since the improvement in performance does not match the majority vote of 10 slightly less precise models.
Creating this majority vote system makes a significant improvement in the performances of the whole, as it jumps from 81-83%  for an individual model to about 87% for 10 together. Increasing the number of models is expected to improve performances further, but training gets proportionally longer with diminishing returns.

# III - To reproduce results

In order to reproduce the results described above, one needs to run this code in the same folder as the data files, namely

```
label_int_to_str_mapping.csv
sample_submission.csv
test_images.npy
train_images.npy
train_labels.csv
```

This code does not save the weights of the trained models, so retraining is needed each time. This is a path for future potential improvements.
Training takes about 3h30. To reduce this, it is recommanded to reduce the `number_of_models` variable, the number of filters per layer, and the number of epochs for the training of each model.
Of course, tworse eprformances are to be expected in counterpart.
