# Turic

Using machine learning + pytorch, attempt to make a train a model to learn to factor numbers. Although it will definitely be inaccurate at times, we have an infinitely sized dataset which can be created relatively quickly. The goal of this project is to create a deep learning model which will reasonably estimate factors of semiprime numbers which should be off by a known error.

## Todo
- [ ]  Compile dataset of 50,000+ big semiprime numbers with their factored form.
- [ ]  Compile test set of 5,000+ big semiprime numbers with their factored form.
- [ ]  Identify the best machine learning model for this project.
- [ ]  Implement the above model.
- [ ]  Add a lot more data.
- [ ]  Write a better algorithm for the deep learning model.
- [ ]  Write it up.

## Potential Problems:

- The model might run slower than an actual factoring algorithm.
    - I think this is not a problem since CNNs do operations on huge arrays.
- The error will be too big for this to be of any use
