import numpy as np
import pickle
import streamlit as st
from sklearn.preprocessing import PolynomialFeatures

# Load the model
try:
    model = pickle.load(open('model.pkl', 'rb'))
except Exception as e:
    st.error(f"Error loading model: {e}")

# Define Streamlit application
def main():
    st.title('Diabetes Prediction App')

    # Input fields
    st.header('Enter your details:')
    age = st.number_input('Age', min_value=0, max_value=120, value=25)
    bmi = st.number_input('BMI', min_value=10.0, max_value=50.0, value=22.0)
    glucose = st.number_input('Glucose Level', min_value=0, max_value=300, value=100)
    insulin = st.number_input('Insulin Level', min_value=0, max_value=300, value=80)

    # Add other input fields as necessary

    if st.button('Predict'):
        try:
            int_features = [age, bmi, glucose, insulin]
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

            st.success(f'PREDICTION => {output}')
        
        except Exception as e:
            st.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
