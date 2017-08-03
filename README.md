## Zillow Prize: Zillow’s Home Value Prediction (Zestimate)
https://www.kaggle.com/c/zillow-prize-1


In this million-dollar competition, participants will develop
an algorithm that makes predictions about the future sale
prices of homes. The contest is structured into two rounds,
the qualifying round which opens May 24, 2017 and the private
round for the 100 top qualifying teams that opens on
Feb 1st, 2018. In the qualifying round, you’ll be building a
model to improve the Zestimate residual error. In the final
round, you’ll build a home valuation algorithm from the ground
up, using external data sources to help engineer new features
that give your model an edge over the competition.

### Model

The code is a mutli-level model using several optimized base
models that are then stacked with a meta model. The final
 goal is have 2 or three meta models and combine then with
either arithmetic and/or geometric averaging.

Base models:
+ Random forest
+ ADA boost
+ Gradient boost
+ KNeighbours (for 1,5,10)

Meta model:
+ Linear regression

### Running code

Requirements:
+ Python 3.5+
+ Sklearn 0.2
+ Numpy 0.2
+ Panadas

To run just type "python run.py". There is an optimisation
flag when the model is called that can be set to run the find
hyper paramaters function but be warned it will take some
time.
