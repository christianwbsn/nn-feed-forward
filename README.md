# nn-feed-forward
----------------------------------------
IF4074 - Advanced Machine Learning

## Description
----------------------------------------
Implementation of mini-batch gradient descent for Feed Forward Neural Network (FFNN) from scratch using Python

## Quick Start
```
python3 -m src.main
```

## Mathematical Formulation
We want to correct the weight of our model to produce the smallest error possible. To do that, we must use calculus: find the derivative of our loss function with respect to the specific weight that we want to correct, and then substract that weight proportional to the derivative. 

The derivative must be calculated with a chain rule. In the code, `Layer.calc_grad()` is used to calculate a single term in the chain rule.

## Authors
-------------------------------
* **[Senapati Sang Diwangkara](https://github.com/diwangs)** - 13516107
* **[Hafizh Budiman](https://github.com/hafizhbudiman)** - 13516137
* **[Christian Wibisono](https://github.com/christianwbsn)** - 13516147
