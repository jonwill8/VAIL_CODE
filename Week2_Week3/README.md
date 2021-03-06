# Introduction
___
This repository holds all my code/ideas/notes from VAIL Week 2 Assignments
___
# Responses
___
## Reflection - 2/16/21

### Question 1
    I think ML concepts were used pretty accurately in this game. An ML model was created to automate a task humans saw 
    fit for automation (hiring from many qualified canidates in a short amount of time). The hiring ML model was then trained
    on a massive dataset of (CV,hiring-outucome-result) from candiates who applied to work at both our firm and apple
    (the ml model needed more data for training purposes). The fact that the ML model amplified the biases both within 
    our firm's hiring process and within apple hiring process is extremely realistic. If a model is feed biased data,
    the model will inevitably spit out biased results (helping to perpetrate bias).

### Question 2
    I heard a podcast a couple of years back about an AI system in South FL which would predict if a criminal 
    would relaspe into criminality after being released. From my hazy memory, I recall while the model had high overall
    accuracy, it was dramatically overpredicting the ciminality relaspe rate for black criminals and dramatically 
    underpredicting the ciminality relaspe rate for white criminals (if analyzed with a confusion matrix, the 
    low specificity of the model is apparent).

    Unfortunately, I believe this model cannot be made more equitable. All criminal data is drawn from policing data. 
    Policing systems have been proven time and time again to hold bias against minority communities 
    (ex: disparities in arrest rates betwen black/white neighborhoods). Any criminal data feed into this model contains 
    intrinsic bias, thus the final model will spit our biased results. I selected this particular AI model becuase it
    was implemented in the area I live (South FL).

## Reflection - 2/17/21
    Differences:
        * A CNN has Convlution/Pooling Layers which process and condense 2d matrix data, while a FC ANN does not have Convlution/Pooling Layers
        * A CNN is tailored for image classification, while a general FC ANN can perform more general tasks such as regression
        * An FC ANN can only be feed vectorized (nx1) input data while a CNN can be feed (n,n) image data input
        * All CNN's contain a FC ANN as a compnent, but the inverse is not true.

## Reflection - 2/23/21
    The key advantage of using ReLu as opposed to Tanh/Sigmoid as the activation function for hidden layers is that
    Relu is able to fight of the vanishing graident problem which plagues Tanh/Sigmoid. Outside of a small sweet spot
    near a weighted input value of 0, the derivative of a neuron which utilizes Tanh/Sigmoid Activation Functions is 
    close to 0. This means that during backpropogation, the weights inside that neuron will not change much. The vanishing 
    gradient problem only amplifies the "deeper" a NN becomes because during backpropogation, weights are updating using 
    a vector which is depdent on the error in the preceeding layer (if a single neuron close to the output layer has a 
    vanishing gradient, than a near zero error signal will be passed back, and weights will not change dramatically and
    it will take our model a longer time to converge).
        
    

