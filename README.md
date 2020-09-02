# DeepLearning
## cs18s038_PA2 Course Project ##
<b> Task:</b> 
Given an input image of shape 64×64×3 from the dataset, the network will be trained
to classify the image into 1 of 201 classes. Tiny ImageNet (subset) dataset from Kaggle is used here.
Project comptetion was hosted on https://www.kaggle.com/c/dl2019pa2/leaderboard
Experiment involves maximising the accuraccy and following deliverables.
* Providing the Configuration and training details of your best performing model.
* Plotting the learning curve showing iterations on the x-axis and negative log likelihood
over labels on the y-axis.A single plot showing both the training loss and the
validation loss.
* Showing the performance on the test data of the model that performs best on the validation
data.
* Providing the parameter setting which gave you the best results.
* Giving the answers to the following questions-
    * Writing down the dimensions of the input and output at each layer (for example, the
input to Conv1 layer is 3 × 64 × 64)
    * Exactly how many parameters does your network have? How many of these are in the
fully connected layers and how many are in the convolutional layers?
    * Exactly how many neurons does your network have? How many of these are in the
fully connected layers and how many are in the convolutional layers?
    * What was the effect of using batch normalization ?
    *  Plot all the 32 layer-1 (Conv1) filters in an 4 × 8 grid. Do you observe any interesting
patterns?
    * Apply guided back propagation on any 10 neurons in the Conv6 layer and plot the
images which excite this neuron. The idea again is to discover interesting patterns
which excite some neurons.
      <object data="https://github.com/rajanskumarsoni/DeepLearning/blob/master/CS18S038_PA2/report.pdf" width="700px" height="700px">
    <embed src="https://github.com/rajanskumarsoni/DeepLearning/blob/master/CS18S038_PA2/report.pdf">
   <p><b>Note:</b> For more detailed analysis of each observation and inferernce drawn from it, Please have a look at full report  <a href="https://github.com/rajanskumarsoni/DeepLearning/blob/master/CS18S038_PA2/report.pdf">Click here</a>.</p>
    </embed>
   </object>
   
   
   
   
## cs18s038_PA3 Course Project ##
<b> Task: </b> The aim of this project is to train and test a Recurrent Neural Network (LSTM) for text
transliteration from English to Hindi.
In this project sequence to sequence networks is implemented. We will work with
transliteration as an example of a sequence to sequence task.
The goal of transliteration is to write a word in one language using the closest corresponding
letters of a different alphabet or language. More formally, the goal of transliteration is to transform
a name (a string of characters) in one language (and corresponding script) to the target language
(and corresponding script) while ensuring phonemic equivalence (preserving and conforming to the
phonology) and conventions of the target language.

<b> Architecture comprise</b> Encoder-Decoder with attention.

