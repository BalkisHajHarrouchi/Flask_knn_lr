from flask import Flask, render_template, request, redirect
import joblib
import pandas as pd
from sklearn.impute import SimpleImputer

app = Flask(__name__, template_folder='template')

# Load the KNN model
model_path_knn = "model/knn_model.joblib"
knn = joblib.load(model_path_knn)

# Load the Linear Regression model
lr_model = joblib.load("model/linear_regression_model.joblib")

# Route for the homepage
@app.route('/')
def home():
    return render_template('navigation.html')

# Route for the KNN model
@app.route('/knn')
def knn_home():
    return render_template('index.html')

# Route for the KNN prediction
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data
        features_knn = [
            float(request.form['High']),
            float(request.form['Low']),
            float(request.form['Open_Price']),
        ]

        # Create a DataFrame with the features
        input_data_knn = pd.DataFrame([features_knn], columns=['High', 'Low', 'Open_Price'])

        # Predict with the KNN model
        prediction_knn = knn.predict(input_data_knn)

        # Render the result to the HTML page
        return render_template('index.html', prediction=prediction_knn[0])

    except Exception as e:
        # Handle errors and display them in the HTML page
        error_message = f"Error: {str(e)}"
        return render_template('index.html', error=error_message)

# Route for the Linear Regression homepage
@app.route('/linear_regression')
def linear_regression_home():
    return render_template('index3.html')

# Route for the Linear Regression prediction
@app.route('/predict_lr', methods=['POST'])
def predict_lr():
    try:
        # Get user input from the form
        stock_price = float(request.form['Stock_Price'].replace(',', ''))

        # Create a DataFrame with the single feature
        input_df_lr = pd.DataFrame({'Stock Price': [stock_price]})

        # Handle NaN values using SimpleImputer
        imputer = SimpleImputer(strategy='mean')
        input_df_lr_no_nan = imputer.fit_transform(input_df_lr)

        # Make predictions using the loaded Linear Regression model
        prediction_lr = lr_model.predict(input_df_lr_no_nan)

        # Display the prediction on the result page
        return render_template('index3.html', prediction_lr=prediction_lr[0])

    except Exception as e:
        # Log the exception for debugging
        app.logger.error(f"Exception: {str(e)}")

        # Display a user-friendly error message
        return render_template('index3.html', error_lr="Error occurred. Please check your input.")

# Run the Flask application
if __name__ == '__main__':
    app.run(debug=True)
