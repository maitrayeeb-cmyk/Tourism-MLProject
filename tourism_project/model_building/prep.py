# for data manipulation
import pandas as pd
import sklearn
# for creating a folder
import os
# for data preprocessing and pipeline creation
from sklearn.model_selection import train_test_split
# for hugging face space authentication to upload files
from huggingface_hub import login, HfApi

# Define constants for the dataset and output paths
api = HfApi(token=os.getenv("HF_TOKEN"))
DATASET_PATH = "hf://datasets/mrhea/tourism-project/tourism.csv"
tourism_dataset = pd.read_csv(DATASET_PATH)
print("Dataset loaded successfully.")

# Define the target variable for the classification task
target='ProdTaken'

# List of numerical features in the dataset
numeric_features = [
    'Age', 'DurationOfPitch', 'MonthlyIncome', 'NumberOfFollowups',
    'NumberOfPersonVisiting', 'NumberOfChildrenVisiting','NumberOfTrips', 'PitchSatisfactionScore'
]

# List of categorical features in the dataset
categorical_features = [
    'TypeofContact',
    'CityTier',
    'Occupation',
    'Gender',
    'PreferredPropertyStar',
    'MaritalStatus',
    'Passport',
    'OwnCar',
    'Designation',
    'ProductPitched'
]

####--------------CLEANUP OF DATA #################
data=tourism_dataset.copy()

## ONE
#Drop columns Unnamed and CustomerId from dataset, if present
if 'Unnamed: 0' in data.columns:
    data = data.drop(columns=['Unnamed: 0'],axis=1)
if 'CustomerID' in data.columns:
    data = data.drop(columns=['CustomerID'],axis=1)
print("Columns Unnamed and CustomerId dropped successfully.")

## TWO
#Drop col AgeCategory, if present, in dataset
if 'AgeCategory' in data.columns:
    data = data.drop(columns=['AgeCategory'],axis=1)
    print("Column 'AgeCategory' dropped successfully.")
else:
    print("Column 'AgeCategory' not found in the DataFrame.")

#Converting the Age data into categories of age 
# Young Adult: 18 to 25 
# Early Mid-Aged: 26 to 35
# Mid-Aged: 36 to 45
# Late Mid-Aged: 46 to 55 
# Senior: > 55              and adding this data as AgeCategory field in dataset
def classify_age(age):
    if age < 25:
        return 'Young Adult'
    elif age < 35:
        return 'Early Mid-Aged'
    elif age < 45:
        return 'Mid-Aged'
    elif age < 55:
        return 'Late Mid-Aged'
    else:
        return 'Senior'

data['AgeCategory'] = data['Age'].apply(classify_age)
categorical_features.append('AgeCategory')

## THREE
# Replacing Gender value of 'Fe Male' with 'Female'
data['Gender'] = data['Gender'].replace('Fe Male', 'Female')

## FOUR
#Replacing MaritalStatus of 'Single' to 'Unmarried'
data['MaritalStatus'] = data['MaritalStatus'].replace('Single', 'Unmarried')


## FIVE
# Checking for duplicate records in data
if data.duplicated().sum()>0:
    print(f"{"Duplicate records found in the dataset.","Number of duplicate records:",data.duplicated().sum()}")
    # Dropping the duplicate records from data
    data.drop_duplicates(inplace=True)
    print("Duplicate records dropped successfully.")
else:
    print("No duplicate records found in the dataset.")




##################################################


# Define predictor matrix (X) using selected numeric and categorical features
X = data[numeric_features + categorical_features]

# Define target variable
y = data[target]


# Split dataset into train and test
# Split the dataset into training and test sets
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y,              # Predictors (X) and target variable (y)
    test_size=0.2,     # 20% of the data is reserved for testing
    random_state=42    # Ensures reproducibility by setting a fixed random seed
)

Xtrain.to_csv("Xtrain.csv",index=False)
Xtest.to_csv("Xtest.csv",index=False)
ytrain.to_csv("ytrain.csv",index=False)
ytest.to_csv("ytest.csv",index=False)


files = ["Xtrain.csv","Xtest.csv","ytrain.csv","ytest.csv"]

for file_path in files:
    api.upload_file(
        path_or_fileobj=file_path,
        path_in_repo=file_path.split("/")[-1],  # just the filename
        repo_id="mrhea/tourism-project",
        repo_type="dataset",
    )
