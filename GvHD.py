# Import the required packages
import numpy as np
import pandas as pd
import os
from sklearn.ensemble import RandomForestClassifier

# Identify working directory
os.getcwd()
# Place data file in working directory

# Import and read data file
data_frame = pd.read_csv('mock_data.csv', sep=',', header=None)

# Introduction
print("\n")
print("This script will first create a Random Forest model using 2/3rd of the data from the 'mock_data.csv' file.")
print("\n")
print("It will then run the model on the remaining 1/3rd of the dataset to allow us to assess accuracy of the model.")
print("\n")
print("The user may then input their own patient data to receive predictions of aGvHD occurence.")
print("\n")
print("\n")

# Split dataframe into training data and test data
training = data_frame.iloc[0:67]
testing = pd.merge(data_frame.head(1), data_frame.iloc[67:101], how='outer')
print("The training dataset is:\n")
print(training)
print("\n")
print("The test dataset is:\n")
print(testing)
print("\n")

# Select target variable for random forest
target = np.array(training[1][1:])

# Select features to be included in random forest model
features = np.array(training.iloc[1:,2:])

# Set forest model settings
forest = RandomForestClassifier(min_samples_split=5, n_estimators = 100, random_state = 1)

# Fit the model to the training data
forest_fit = forest.fit(features, target)

# Print the score of the fitted random forest
#print(forest_fit.score(features, target))

# Print the relative importance of each variable in the model
#print(forest_fit.feature_importances_)

# Set features to be included in the test run
test_features = np.array(testing.iloc[1:,2:])

# Make your prediction using the test set
prediction = forest_fit.predict(test_features)
prediction = list(map(int, prediction))

# Create a data frame with four columns: ID, predicted aGvHD, actual aGvHD, and outcome
ID = np.array(testing[0][1:])
prediction_df = pd.DataFrame({"DR_ID" : ID,
                             "predicted_aGvHD" : prediction})
actual = np.array(testing[1][1:])
actual = list(map(int, actual))
actual_df = pd.DataFrame({"DR_ID" : ID,
                         "actual_aGvHD" : actual})
predicted_vs_actual = pd.merge(prediction_df, actual_df, how = 'left', on = "DR_ID")

def outcome(df):
  if df['predicted_aGvHD'] == 1 & df['actual_aGvHD'] == 1:
    return 'True Positive'
  elif df['predicted_aGvHD'] == 0 & df['actual_aGvHD'] == 0:
    return 'True Negative'
  elif df['predicted_aGvHD'] == 1 & df['actual_aGvHD'] == 0:
    return 'False Positive'
  else:
    return 'False Negative'

predicted_vs_actual['outcome'] = predicted_vs_actual.apply(outcome, axis=1)

print("The model predictions are placed alongside actual data to assess accuracy:\n")
print(predicted_vs_actual)

# Calculate sensitivity and specificity
outcome_list = predicted_vs_actual['outcome']

true_pos = 0
true_neg = 0
false_pos = 0
false_neg = 0

for i in outcome_list:
    if i == "True Positive":
        true_pos = true_pos+1
    elif i == "True Negative":
        true_neg = true_neg+1
    elif i == "False Positive":
        false_pos = false_pose+1
    elif i == "False Negative":
        false_neg = false_neg+1
    else:
        print("Unknown Outcome")

print("\n")
print("Sensitivity: "+str(round(true_pos/(true_pos+false_pos), 2)))
print("Specificity: "+str(round(true_neg/(true_neg+false_neg), 2)))
print("Overall Accuracy: "+str(round((true_pos/(true_pos+false_pos)+true_neg/(true_neg+false_neg))/2, 2)))
print("\n")
print("\n")

# Instructions to user
print("Load your patient data as a csv file in the same format as the 'mock_data.csv' file.\n")
print("Assign it to the variable 'patient_data' and execute the model using the code provided at the end of this script.\n")
print("Your output is below:\n\n")

# Load your patient data and assign it to the variable 'patient_data'
# Unhash the code below and run the script

#patient_data = open("patient_data.csv").read()
#application_features = np.array(patient_data.iloc[1:,2:])
#prediction2 = forest_fit.predict(application_features)
#ID2 = np.array(patient_data[0][1:])
#prediction_df2 = pd.DataFrame({"DR_ID" : ID,
#                             "predicted_aGvHD" : prediction})