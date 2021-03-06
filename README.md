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

<b> Architecture comprise</b> Encoder-Decoder with attention, Dropout, early stopping, tanh non-linearity.
Project competition was hosted on https://www.kaggle.com/c/programming-assignment-3/leaderboard .

Experiment involves maximising the accuraccy and following deliverables

   * A plot of the learning curve showing iterations on the x-axis and negative log likelihood over
labels on the y-axis. Making a single plot showing both the training loss and the validation loss.
   * Reporting the parameter setting which gave the best results and the performance on the test
data of the model that performs best on the validation data.)
   * Reported a table with validation accuracies with different hyperparameter settings.
   * Wrote down the dimensions of the input and output at each layer (for example, the input to
INEMBED layer is 20 x 50 x 256)
   * Observing the effect of unidirectional LSTM for the encoder. 
   * Explaining the attention mechanism we used with equations.
   * Observed the effect of using attention mechanism.
   * Observed the visualization of the attention layer weights for a sequence pair. Tried to see meaningful
character alignments.
   * Observed the effect of using 2-layered decoder as compared to single decoder.
   * Observed the effect of using dropout.
   
  <object data="https://github.com/rajanskumarsoni/DeepLearning/blob/master/CS18S038_PA3/main.pdf" width="700px" height="700px">
    <embed src="https://github.com/rajanskumarsoni/DeepLearning/blob/master/CS18S038_PA3/main.pdf">
   <p><b>Note:</b> For more detailed analysis of each observation and inferernce drawn from it, Please have a look at full report  <a href="https://github.com/rajanskumarsoni/DeepLearning/blob/master/CS18S038_PA3/main.pdf">Click here</a>.</p>
    </embed>
   </object>
   
   
## cs18s038_PA4 Course Project
<b> Task: </b> In this assignment we have implemented Restricted Boltzmann machines (RBMs) using only
python and numpy. We were not allowed to use tensorflow, theano or any package which
supports automatic differentiation.
RBMs can be used to learn hidden representations (h) from the raw features (V ). Our
task is to train RBMs using the Contrastive Divergence (CD) algorithm. Specifically, given
the 784 dimensional (V ) binary fashion-MNIST data (attached csv files) we need to learn
a n-dimensional hidden representation (h). We had to convert the real valued fashionMNIST data into binary data by using a threshold of 127 (any pixel having a value less than
127 will be treated as 0 and any pixel having a value greater than or equal to 127 will be
treated as 1). We 
1. We used the training data (60000 images) for training the RBM.
2. After training, we computed the compute the hidden representations for the 10000 test images.

Experiment involves following deliverables

   1. Used t-SNE to plot the learned representations in a 2-dimensional space
   
   2. In every step of stochastic gradient descent (SGD) we run the
   Gibbs Chain for k steps. Studied the effect of using different values of k : 1; 3; 5; 10; 20.
   3. Assumeing that CD takes around m iterations of SGD to converge, where
   m = 6400. Plotted the samples generated by Gibbs chain after every 64 m steps of SGD.
   Used an 8 x 8 grid to plot these 64 samples.
   
   
   <object data="https://github.com/rajanskumarsoni/DeepLearning/blob/master/CS18S038_PA4/main.pdf" width="700px" height="700px">
    <embed src="https://github.com/rajanskumarsoni/DeepLearning/blob/master/CS18S038_PA4/main.pdf">
   <p><b>Note:</b> For more detailed analysis of each observation and inferernce drawn from it, Please have a look at full report  <a href="https://github.com/rajanskumarsoni/DeepLearning/blob/master/CS18S038_PA4/main.pdf">Click here</a>.</p>
    </embed>
   </object>



