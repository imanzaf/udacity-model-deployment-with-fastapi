# Model Card

## Model Details
The model trained is a Light Gradient Boosting Machine (LGBM) Classifier.
It was trained on census data containing 6 numeric features and 9 categorical features.
The model uses these features to categorise an individual to be earning either <= 50k or >50k,
where a prediction of 1 represents the former.

## Intended Use
The intended use of the model is to predict whether an individual is earning less than / equal to 
or more than 50k.

## Training Data
A sample of 80% of the census data was used to train the model.

## Evaluation Data
The remaining sample of 20% of the census data was used to evaluate the model.

## Metrics
The metrics used to evaluate the model, and the model's score for these metrics are detailed as follows:
<br/> - _Precision_ = TP / (FP + TP) = 0.791
<br/> - _Recall_ = TP / (FN + TP) = 0.683
<br/> - _fbeta_ = (1+β^2)(precision*recall) / ((β^2)precision+recall) = 0.733
<br/> - _Accuracy_ = (TP+TN) / (TP+TN+FP+FN) = 0.880

## Ethical Considerations
- Data Bias: This phenomenon occurs when data isn't representative, 
leading to model predictions that are based on incorrect assumptions. 
This could mean decisions made using the model may lead to unfavourable outcomes.
- Model Bias: Due to the fact that no data checks were conducted, and 
limited hyperparameter tuning was conducted, the trained model may have
systematic errors that would lead to incorrect predictions.


## Caveats and Recommendations
In addition to the ethical considerations, some caveats of the model include lack of:
- k-fold cross validation 
- stratified sampling 
- extensive hyperparameter tuning 
- experimentation with other models (e.g., logistic regression)
<br/> 

The above caveats can be taken into account to improve the model training process for better results.
