# Speech_Emotion_recognition
The neural network model is capable of detecting eight different emotions from audio speeches. (Deep Learning, NLP, Python)
The idea behind creating this project was to build a machine learning model that could detect emotions from the speech we have with each other all the time. Nowadays personalization is something that is needed in all the things we experience every day.
So why not have a emotion detector that will gauge your emotions and in the future recommend you different things based on your mood. This can be used by multiple industries to offer different services like marketing company suggesting you to buy products based on your emotions, automotive industry can detect the persons emotions and adjust the speed of autonomous cars as required to avoid any collisions etc.
# Datasets:
Made use of three different datasets:
RAVDESS. This dataset includes around 1500 audio file input from 24 different actors. 12 male and 12 female where these actors record short audios in 8 different emotions i.e. 1 = neutral, 2 = calm, 3 = happy, 4 = sad, 5 = angry, 6 = fearful, 7 = disgust, 8 = surprised. Each audio file is named in such a way that the 7th character is consistent with the different emotions that they represent.
SAVEE. This dataset contains around 500 audio files recorded by 4 different male actors. The first two characters of the file name correspond to the different emotions that the portray.
Tess. There are a set of 200 target words were spoken in the carrier phrase "Say the word _' by two actresses (aged 26 and 64 years) and recordings were made of the set portraying each of seven emotions (anger, disgust, fear, happiness, pleasant surprise, sadness, and neutral). There are 2800 data points (audio files) in total.
Audio files:
Tested out the audio files by plotting out the waveform and a spectrogram to see the sample audio files.
Waveform  

[©Fabien_Ringeval_PhD_Thesis](https://drive.google.com/file/d/0B2V_I9XKBODhcEtZV1lRWW1fYTg/view).
<br>

Spectrogram
![](images/cnn_acc_and_..png?raw=true)
Feature Extraction
The next step involves extracting the features from the audio files which will help our model learn between these audio files. For feature extraction we make use of the LibROSA library in python which is one of the libraries used for audio analysis.
 Here there are some things to note. While extracting the features, all the audio files have been timed for 3 seconds to get equal number of features.
The sampling rate of each file is doubled keeping sampling frequency constant to get more features which will help classify the audio file when the size of dataset is small.
The extracted features look as follows
 
These are array of values with labels appended to them.
Building Models
Since the project is a classification problem, Convolution Neural Network seems the obvious choice. We also built CNN and Long Short-Term Memory models and they performed with better accuracies.
Building and tuning a model is a very time-consuming process. The idea is to always start small without adding too many layers just for the sake of making it complex. After testing out with layers, the model which gave the max validation accuracy against test data was little more than 85%

                     
Predictions
After tuning the model, tested it out by predicting the emotions for the test data. For a model with the given accuracy these are a sample of the actual vs predicted values.

 
Testing out with live voices
In order to test out our model on voices that were completely different than what we have in our training and test data, we recorded our own voices with different emotions and predicted the outcomes 

 
This is the function which extracts features from audio and take it as numpy array and predicts the emotion.
Conclusion
Building the model was a challenging task as it involved lot of trial and error methods, tuning etc. The model is very well trained to distinguish emotions and it distinguishes with 100% accuracy. The model was tuned to detect emotions with more than 85% accuracy on CNN . Accuracy can be increased by including more audio files for training.
