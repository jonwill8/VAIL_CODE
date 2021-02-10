# Introduction
___
This repository holds all my code/ideas/notes from VAIL Week 1 Assignments
___
# Responses
___
## Reflection - 2/8/21

     I hope to gain a deeper understanding of AI through this course. While I have built math models before that
     fit well to logged data, none of my models could make predictions on new data. I am looking forward to building AI
     models which can accurately make predictions on never before seen data. I am pretty excited to see how
     prior calculus courses I have taken tie into AI models. I am anxious to learn enough Linear Algebra to understand
     AI models.

## Reflection - 2/9/21

### Question 1
    In supervised learning, you are fitting a model on labeled training data so that the model becomes very good 
    at mapping inputs to the outputs you deem are correct (ex: you may train a classifier model on images of 
    apples/oranges such that it becomes really good at seeing a new image of an apple/orange and then 
    categorizing it correctly). In supervised learning, you must balance the bias vs variance of your model 
    (low bias and high variance means high accuracy but low model applicability to different data sets, 
    while high bias and low variance means low accuracy but high model applicability to different data sets)

    In unsupervised learning, you are employing an ML model on unlabeled/messy data such 
    that the model can find patterns in the data which are unapparent to humans. Unsupervised models 
    are generally more complex and it is harder to rate the accuracy of an unsupervised ML model.
### Question 2
    I believe that SKL does NOT have the power to visualize data without a Graphviz, Pandas, or 
    other data analysis libraries because SKL is only concered with ML & data modeling. It is not meant
    for data visualization and thus cannot plot/represent data without the assitance of the helper modules 
    it is built upon such as pandas/matplotlib

## Reflection - 2/10/21

### Question 1
    Tensors are a more generalized form of vectors we are all comfortable with. Tensors are more powerful than 
    vectors becuase they allow us to explore mathematical operations in higher dimensons (ex: calculating
    all forces acting on the interior of a solid object). While vectors can be though of as having components with
    singular direction (i,j,k), tensors can be thought of as having compnents with multiple directions (ex: ii or jk).
    The complexity of a tensor scales with its rank. A tensor of rank n with have 3^n compnents 
    (rank 0 tensor = a 1 component scalar, rank 1 tensor = a 3 component vector, rank 2 tensor will have 9 compnents).

    Tensors are important in ML becuase they are very effective at describing multidimensional datasets. From 
    what I have coded already, it seems tensors/vectors are a much more efficent means to store model parameter 
    values (when you are mapping each weight to its corresponding input variable, you would only have to take a
    dot product between the weight tensor/vector and the input variable value vector). I hope to gain a deeper understanding
    of tensors over time as the topic is nontrivial and hard to grasp for a beginner like myself.

### Question 2
    One thing I noticed is tensorflow computations are applied elementwise to each input vector

