from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd

housing_data = pd.read_csv('C:/Users/ELCOT/Desktop/Task 1/housing_data.csv')

housing_data = housing_data.drop(columns=['Unnamed: 0'])

X = housing_data.drop(columns=['Price'])
y = housing_data['Price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), ['Area', 'No. of Bedrooms']),
        ('cat', OneHotEncoder(handle_unknown='ignore'), ['City', 'Location'])
    ])

model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(random_state=42))
])

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
print(f"Mean Absolute Error: {mae}")

def predict_price(new_data):
    """
    Predicts the price of a house given the features.

    new_data: dict, contains 'City', 'Area', 'Location', 'No. of Bedrooms'
    """
    new_df = pd.DataFrame([new_data])
    predicted_price = model.predict(new_df)
    return predicted_price[0]

new_house = {'City': 'Bangalore', 'Area': 1200, 'Location': 'JP Nagar Phase 1', 'No. of Bedrooms': 3}
predicted_price = predict_price(new_house)
print(f"Predicted Price for the new house: {predicted_price}")
