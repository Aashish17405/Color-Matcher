# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import LabelEncoder
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.metrics import accuracy_score

# # Load the dataset
# file_path = 'updated_color_matcher.csv'
# data = pd.read_csv(file_path)

# # Encode the 'top_color' and 'bottom_color' as categorical values
# top_color_encoder = LabelEncoder()
# bottom_color_encoder = LabelEncoder()

# data['top_color_encoded'] = top_color_encoder.fit_transform(data['top_color'])
# data['bottom_color_encoded'] = bottom_color_encoder.fit_transform(data['bottom_color'])

# # Separate the features (X) and target (y)
# X = data[['top_color_encoded']]  # Input: top color
# y = data['bottom_color_encoded']  # Output: bottom color

# # Split the data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Initialize the KNN classifier
# knn = KNeighborsClassifier(n_neighbors=5)

# # Train the KNN model
# knn.fit(X_train, y_train)

# # Make predictions on the test set
# y_pred = knn.predict(X_test)

# # Evaluate the model's accuracy
# accuracy = accuracy_score(y_test, y_pred)
# print(f"Model Accuracy: {accuracy * 100:.2f}%")

# # Example prediction: predict bottom color for a new top color
# new_top_color = '#FF7F50'
# new_top_color_encoded = top_color_encoder.transform([new_top_color])
# predicted_bottom_color_encoded = knn.predict([new_top_color_encoded])
# predicted_bottom_color = bottom_color_encoder.inverse_transform(predicted_bottom_color_encoded)

# print(f"Suggested bottom color for {new_top_color}: {predicted_bottom_color[0]}")


# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import LabelEncoder
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import accuracy_score

# # Load the dataset
# file_path = 'updated_color_matcher.csv'
# data = pd.read_csv(file_path)

# # Encode the 'top_color' and 'bottom_color' as categorical values
# top_color_encoder = LabelEncoder()
# bottom_color_encoder = LabelEncoder()

# data['top_color_encoded'] = top_color_encoder.fit_transform(data['top_color'])
# data['bottom_color_encoded'] = bottom_color_encoder.fit_transform(data['bottom_color'])

# # Separate the features (X) and target (y)
# X = data[['top_color_encoded']]  # Input: top color
# y = data['bottom_color_encoded']  # Output: bottom color

# # Split the data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Initialize the Random Forest classifier
# rf = RandomForestClassifier(n_estimators=100, random_state=42)

# # Train the Random Forest model
# rf.fit(X_train, y_train)

# # Make predictions on the test set
# y_pred = rf.predict(X_test)

# # Evaluate the model's accuracy
# accuracy = accuracy_score(y_test, y_pred)
# print(f"Random Forest Model Accuracy: {accuracy * 100:.2f}%")

# # Example prediction: predict bottom color for a new top color
# new_top_color = '#FF7F50'
# new_top_color_encoded = top_color_encoder.transform([new_top_color])
# predicted_bottom_color_encoded = rf.predict([new_top_color_encoded])
# predicted_bottom_color = bottom_color_encoder.inverse_transform(predicted_bottom_color_encoded)

# print(f"Suggested bottom color for {new_top_color}: {predicted_bottom_color[0]}")



import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load the dataset
file_path = 'updated_color_matcher.csv'
data = pd.read_csv(file_path)

# Encode the 'top_color' and 'bottom_color' as categorical values
top_color_encoder = LabelEncoder()
bottom_color_encoder = LabelEncoder()

data['top_color_encoded'] = top_color_encoder.fit_transform(data['top_color'])
data['bottom_color_encoded'] = bottom_color_encoder.fit_transform(data['bottom_color'])

# Separate the features (X) and target (y)
X = data[['top_color_encoded']]  # Input: top color
y = data['bottom_color_encoded']  # Output: bottom color

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Logistic Regression model
lr = LogisticRegression(max_iter=200)

# Train the Logistic Regression model
lr.fit(X_train, y_train)

# Make predictions on the test set
y_pred = lr.predict(X_test)

# Evaluate the model's accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Logistic Regression Model Accuracy: {accuracy * 100:.2f}%")

# Example prediction: predict bottom color for a new top color
new_top_color = '#87CEEB'
new_top_color_encoded = top_color_encoder.transform([new_top_color])
predicted_bottom_color_encoded = lr.predict([new_top_color_encoded])
predicted_bottom_color = bottom_color_encoder.inverse_transform(predicted_bottom_color_encoded)

print(f"Suggested bottom color for {new_top_color}: {predicted_bottom_color[0]}")
