import numpy as np
from flask import Flask, request, render_template
from sklearn.preprocessing import PolynomialFeatures
import pickle

# Initialize the Flask App
app = Flask(__name__)

# Load the model
try:
    model = pickle.load(open('model.pkl', 'rb'))
except Exception as e:
    print(f"Error loading model: {e}")

@app.route('/')
def home():
    return render_template('land.html')

@app.route('/software')
def software():
    return render_template('home.html')

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Fetch data from the client-side form
        int_features = [float(x) for x in request.form.values()]
        final_features = [np.array(int_features)]
        
        # Apply polynomial transformation if necessary
        poly_reg = PolynomialFeatures(degree=2)
        transformed_features = poly_reg.fit_transform(final_features)
        
        # Model prediction
        prediction = model.predict(transformed_features)
        
        # Interpretation of the prediction result
        if prediction > 0.8:
            output = ("DIABETIC - PLEASE TAKE CARE OF YOUR ROUTINE AND MAKE LIFESTYLE CHANGES. "
                      "Probability of being diabetic is HIGH!")
        else:
            output = ("NON-DIABETIC - Congratulations. Your lifestyle is brilliant. Stay healthy, stay safe!")

        return render_template('index.html', prediction_text='PREDICTION => {}'.format(output))
    
    except Exception as e:
        return f"An error occurred: {e}"

if __name__ == "__main__":
    # Use a different port to avoid conflict
    app.run(port=5001, threaded=True)
