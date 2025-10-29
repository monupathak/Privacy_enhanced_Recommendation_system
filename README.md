# Privacy_enhanced_Recommendation_system

Project Overview

This project began as my course project for IE 506: Machine Learning Principles and Techniques at IIT Bombay. Later, I thought — why not improve it and make it open source? That’s how this project evolved.

In this project, I implemented the concepts from the paper [“Applying Differential Privacy to Matrix Factorization”](https://dl.acm.org/doi/pdf/10.1145/2792838.2800173) The paper explores the application of Differential Privacy in Recommendation Systems based on Collaborative Filtering techniques like Matrix Factorization.
You can watch this video to better understand the concept of matrix factorization[you can watch this video to understannd the concept of matrix factorization](https://www.youtube.com/watch?v=ZspR5PZemcs).

Optimization Approaches in Matrix Factorization

There are two main optimization approaches commonly used in Matrix Factorization:

1. Alternating Least Squares (ALS)

ALS is a matrix factorization technique used for collaborative filtering. It iteratively updates user and item latent factor matrices to minimize the error between predicted and actual ratings.

2. Stochastic Gradient Descent (SGD)

SGD is a widely used optimization algorithm in machine learning. It updates model parameters in the direction of the negative gradient of the loss function, using a subset of the data (mini-batch) at each iteration.

I have implemented both ALS and SGD approaches from scratch for this project.

Dataset

I used the MovieLens 100K dataset for training and evaluation.
A detailed report of this project is also attached for reference.










 
