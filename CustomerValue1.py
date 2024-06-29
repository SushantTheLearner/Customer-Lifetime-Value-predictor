import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load the customer purchase data from a CSV file
customer_data = pd.read_csv('customer_data.csv')

# Display the first few rows of the dataset to understand its structure
print(customer_data.head())

# Ensure the dataset contains the necessary columns
required_columns = ['CustomerID', 'AveragePurchaseValue', 'PurchaseFrequency', 'FirstPurchaseDate', 'LastPurchaseDate']
for column in required_columns:
    if column not in customer_data.columns:
        raise ValueError(f"Missing expected column: {column}")

# Convert date columns from strings to datetime objects
try:
    customer_data['FirstPurchaseDate'] = pd.to_datetime(customer_data['FirstPurchaseDate'], errors='coerce')
    customer_data['LastPurchaseDate'] = pd.to_datetime(customer_data['LastPurchaseDate'], errors='coerce')
except Exception as e:
    print(f"Date conversion error: {e}")
    raise

# Remove rows with invalid date entries
if customer_data['FirstPurchaseDate'].isnull().any() or customer_data['LastPurchaseDate'].isnull().any():
    print("Dropping rows with invalid date entries")
    customer_data = customer_data.dropna(subset=['FirstPurchaseDate', 'LastPurchaseDate'])

# Calculate the lifespan of each customer in days
customer_data['CustomerLifespan'] = (customer_data['LastPurchaseDate'] - customer_data['FirstPurchaseDate']).dt.days

# Calculating the total value spent by the cutomers
customer_data['TotalValue'] = customer_data['AveragePurchaseValue'] * customer_data['PurchaseFrequency']

# Selection of (x) variables and directing(y)
features = customer_data[['AveragePurchaseValue', 'PurchaseFrequency', 'CustomerLifespan']]
target = customer_data['TotalValue']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Creating train and linear regression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Make predictions on the test set
predictions = regressor.predict(X_test)

# calcukating Mean Absolute Error and Mean Squared Error
mae_value = mean_absolute_error(y_test, predictions)
mse_value = mean_squared_error(y_test, predictions)
print(f'Mean Absolute Error: {mae_value}')
print(f'Mean Squared Error: {mse_value}')

# Calculate and print the R-squared score if there are enough test samples
if len(X_test) > 1:
    r2_value = r2_score(y_test, predictions)
    print(f'R-squared: {r2_value}')
