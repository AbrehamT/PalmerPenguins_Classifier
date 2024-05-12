# This is an SVM method to calculate the accuracies of the penguins file
# using different libraries and training the data to get the most accuracy
# Import all the libraries we need 
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score

# Set the dataset to a variable where it holds all the data
penguin_dataset = pd.read_csv("palmerpenguins_original.csv")

# This dropna is so that it can drop the rows that don't have any data
penguin_dataset = penguin_dataset.dropna()

# This uses the label encoder to set and encode each of the penguin datasets
encode_svm = LabelEncoder()
penguin_dataset['species'] = encode_svm.fit_transform(penguin_dataset['species'])
penguin_dataset['island'] = encode_svm.fit_transform(penguin_dataset['island'])
penguin_dataset['sex'] = encode_svm.fit_transform(penguin_dataset['sex'])

# Split features and target variable
species1 = penguin_dataset.drop(['species'], axis=1)
species2 = penguin_dataset['species']

# standard scaler to scale each of the features and then transform it
features_slr = StandardScaler()
species1_scaled = features_slr.fit_transform(species1)

# Categorize the data into two and split it between the training and testing sets
species1_train, species1_test, species2_train, species2_test = train_test_split(species1_scaled, species2, test_size=0.2, random_state=42)

# Train SVM model
done_train = SVC(kernel='linear')
done_train.fit(species1_train, species2_train)

# Take the already trained data and predict that 
species2_pred = done_train.predict(species1_test)

# Calculate accuracy and print it in percentage value 
accuracy = accuracy_score(species2_test, species2_pred) *100
print(f"The accuracy percentage using SVM algorithm: {accuracy:6f}%")

