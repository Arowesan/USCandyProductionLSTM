### Machine Learning exercise: Time Series and Uncertainty Estimate


Task: Model the US Candy Production time series 
      using machine learning and compute uncertainty 
	  of a single prediction.

## Prerequisites:
- Python 3 
- Keras with TensorFlow backend
- numpy, scipy, sklearn
- pandas
- candy_production.csv (https://www.kaggle.com/rtatman/us-candy-production-by-month)

Prototyping was done on the Jupyer notebook (USCandyProductionLSTM.ipynb). After being satisfied
with the results I transferred the code into a single python file (USCandayProductionLSTM.py).
This program can be run with the following command:

   `python USCandyProductionLSTM.py`
   
Alternatively, the program can be run on a python IDE so 
long as the above prerequisites are met.
  

## Uncertainty Estimate Implementation Explanation:

In principle, a Bayesian network could be used
to quantify the uncertainty of a single prediction
from a neural network. However, such use of a model 
is intractable. On the other hand, it has 
been shown by Yarin Gal that the use of dropouts 
on a neural network leads to a Bayesian approximation 
of the Gaussian process.

To achieve this Bayesian approximation more precisely, 
one carries out the usual (standard) 
dropout approach during the training phase.
During testing, one uses the so-called Monte Carlo (MC) dropout
where a number of stochastic forward passes is carried out 
in the model. It is important to note that the dropout probabilities 
used during training phase are retained in MC dropout.
This then generates a sample of predictions for a 
single input value, which allows one to 
compute uncertainty estimates of this sample. Since it can be inferred that this sample from MC dropout 
is approximately normally distributed, one could then compute uncertainty estimates such as confidence interval 
with standard methods. My solution does MC dropout 
on the entire test set but extracts one of them 
and calculates statistics on it, including the 95% confidence 
interval where I used the 'norm.interval' method from the 
scipy.stats class.

For the US Candy Production data set where a 
Long Short-Term Memory (LSTM) layer is used in a recurrent
neural network, one applies variational dropout 
to achieve the same Bayesian approximation. 
Fortunately this has already been implemented in the LSTM
layer in Keras(^) but one needs to specify an input, 
recurrent, and output dropout probabilities. 
To implement MC dropout, one can define 
a backend function in Keras that allows the use of
the model which has the same dropout probabilities that were
used from training. Note that the standard 'model.predict' method 
in Keras does not include dropouts in the Sequential model(^^). 
In the present code, this is defined as 'predict_stochastic' 
and can be found on line 128. This function allows one to specify 
which model to use based on the 'learning_phase' parameter
(0 = testing, 1 = training). This function is then used 
on line 136. Statistics calculations are then
performed starting at line 152.


## Other notes and remarks about my solution:
- The given US Candy Production time series has a small
  increasing trend. This trend was removed in order 
  to make the modeling easier. The trend is added back 
  when comparing the predicted value with the expected.
- Since the LSTM layer uses the hyperbolic tangent function 
  as the default activation, which outputs values between -1 and 1, 
  the data set was scaled to this range before modeling.
- The code originally extracts the first output 
  value of the predicted set with monthIdx = 0 (line 138). 
  Change this value to extract prediction for a different month,
  so long as it's less than len(test)=165.
- During prototyping, I've checked the r^2 score of 
  my test predictions with this data set, which is about 0.996 from 
  using the standard model.predict method.
  For MC dropout a somewhat better value of 0.998 was obtained.
- The structure of this code is for demonstration purposes. 
  For software development, I would actually
  wrap the code in a class.


## References:

[1] Y. Gal and Z. Gharamani, "Dropout as Bayesian Approximation:
    Representing Model Uncertainty in Deep Learning," arXiv:1506.02157, 2015.
	
[2] Y. Gal and Z. Gharamani, "A Theoretically Grounded Application of 
    Dropout in Recurrent Neural Networks," arXiv:1512.05287, 2015.

-------
(^) Interestingly Y. Gal helped with the implementation [2]. 

(^^) One can retain dropout probabilities in the standard 'model.predict' method 
by using functional API to build the model and setting 'training=True' but I find 
the MC dropout calculations with this approach very slow.
