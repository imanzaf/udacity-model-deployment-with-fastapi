# Deploying an ML Model with FastAPI

Repository for the 3rd project in the Udacity Machine Learning DevOps Engineer Nanodegree

### Project Description
The project had the following requirements for completion:
- Setting up a GitHub Action for continuous integration
- Training a Machine Learning model to predict an individual's salary using census data
- Evaluating model performance on data slices
- Creating a REST API using the FastAPI library that implements a POST method that does model inference
- Writing unit tests for src functions and test cases for API
- Deploying API on Render.com

### File Descriptions
- src/ - package containing functions written for model training and evaluation
- train_model.py - script to train LGBM model on census data
- model_card.md - model card of trained model
- training_output/ - contains model, encoder, and binarizer objects returned by train_model.py
- test_src.py - unit tests for src code
- get_slice_metrics.py - script to get model metrics on data slices for categorical features
- slice_output/ - contains output of get_slice_metrics.py
- app/
  - main.py - code for creating API
  - test_main.py - test cases for API
  - live_post.py - live example for API post method
- screenshots/ - contains required screenshots

<br/>

#### [Link to Render Dashboard](https://dashboard.render.com/web/srv-cia6e4lgkuvusaobkckg/events)
#### [Link to GitHub Repository](https://github.com/imanzaf/udacity-model-deployment-with-fastapi)

