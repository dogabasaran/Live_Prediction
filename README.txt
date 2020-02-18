Simple machine learning pipeline to gather data from computer camera, preprocess it, train a convolutional neural network and evaluate it using a live camera feed.


File Descriptions:
buildtrain.py: runs get_data.py, preprocess.py and netmodel.py in a pipeline with respect to given arguments. Can also do this via running scripts individually.
get_data.py: Open up camera feed to gather data, stores images into data/ folder in folders of the labels provided
preprocess.py: Prepares the data for convnet. Resizes, normalizes, and shuffles the dataset
netmodel.py: defines neural network model and builds it. Dataset is turned into tensors, and split into training and cvalidation sets.
live_predict.py: Opens camera feed, neural network predicts the label of the input feed.



Usage:
python buildtrain.py -b <BUILD_STATE> -i <IMAGES_PER_LABEL> -t <TRAIN_STATE>
python live_predict.py
