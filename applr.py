from flask import Flask, render_template, request, redirect
import joblib
import pandas as pd

app_lr = Flask(__name__, template_folder='template')

# Load the Linear Regression model and scaler
lr_model = joblib.load("model/linear_regression_model.joblib")
lr_scaler = joblib.load("model/linear_regression_scaler_X.joblib")

# Assuming selected columns from feature selection
selected_columns_lr = ['Stock Price']

# Update the linear_regression_home route
@app_lr.route('/')
def linear_regression_home():
    return redirect('/linear_regression')

# Keep the existing linear_regression_home route for accessing '/linear_regression'
@app_lr.route('/linear_regression', methods=['GET'])
def linear_regression_form():
    return render_template('index3.html')

@app_lr.route('/predict_lr', methods=['POST'])
def predict_lr():
    try:
        # Get user input from the form
        input_data_lr = request.form.to_dict()

        # Convert numerical values to float and remove commas
        input_data_lr = {key: float(value.replace(',', '')) for key, value in input_data_lr.items()}

        # Prepare input data as a DataFrame with feature names
        input_df_lr = pd.DataFrame(data=input_data_lr, index=[0], columns=selected_columns_lr)

        # Use the scaler fitted during training for Linear Regression
        input_scaled_lr = lr_scaler.transform(input_df_lr)

        # Make predictions using the loaded Linear Regression model
        prediction_lr = lr_model.predict(input_scaled_lr)

        # Display the prediction on the result page
        return render_template('index3.html', prediction_lr=prediction_lr[0])

    except Exception as e:
        # Log the exception for debugging
        app_lr.logger.error(f"Exception: {str(e)}")

        # Display a user-friendly error message
        return render_template('index3.html', error_lr="Error occurred. Please check your input.")

if __name__ == '__main__':
    app_lr.run(debug=True)
