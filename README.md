##### Noah X. Deutsch / Flatiron School / Data Science Program / Capstone Project


# Classy

---

## Abstract

#### About
Classy is a python package that lets anybody create their own basic machine learning classifier from any dataset in under 10 lines of code.

#### The Problem

**Machine learning is hard.** Propperly scrubbing, modeling, optimizing, and interpreting results requires a lot of training, tinkering and practice. __What if it could be much simpler?__

#### The Solution

For my final project for the Flatiron School's Data Science Bootcamp, I set out to build a python "chat bot" that can guide anybody step-by-step through a basic ML classification workflow, using the dataset of their choosing. Ultimately, I believe I succeeded in creating a tool that radically simplifies the process of exploring, scrubbing, modeling, interpreting, and predicting data for straightforward classification tasks.

---

## How it works

#### Setup -> **c = Classy()** 
Simply create a new instance of the Classy object to get started.

#### Obtaining Data -> **c.obtain()**
The obtain function detects all the CSV files in the directory and asks you to select one to work with.
This function also asks you to identify which column you are trying to predict (the target).

#### Exploring Data -> **c.explore()**
The explore function allows you to explore both the original dataframe and the scrubbed dataframe (if .explore() is called after .scrub() has been called).
The explore function has two modes:
    - Simple: which calls .head(), .info(), and .describe() for the chosen dataframe.
    - Advanced: which makes use of the pandas profiling library to provide a more robust exploratory environment.

#### Scrubbing Data -> **c.scrub()**
The scrub function walks you step by step through the process of cleaning your data. This function allows you to take a sample of your data, remove specific values and columns, fix column data types, and properly deal with missing data, among other things.

#### Model Data -> **c.model()**
The model function first prepares your scrubbed data for modelling by getting dummy variables and allowing you to normalize or standardize your data. Then, the function allows you select from multiple classification models (as many as you would like), which it proceeds to run, optimize, analyze and visualize the results for you.

#### Interpret Data -> **c.interpret()**
The interpret function plots the test accuracy for each model that has previously been run through the .model() function.

#### Interpret Data -> **c.predict()**
The predict function allows you to input new data and view the predicted class for the best performing model (in addition to the predictions for all previously run models). 

---

## Future Improvements

In future versions of this tool, I would like to make the scrubbing function more robust, add additional classifiers (including neural networks) to the model function, improve the optimization part of the model function to allow it to scale up depending on desired wait time, and add more nuance to the interpretation function.

