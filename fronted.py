import requests
import streamlit as st


API_URL = "http://localhost:8000/predict"

st.set_page_config(page_title="Premium Category Predictor", page_icon="📊")
st.title("Insurance Premium Category Predictor")
st.caption("Enter your details to estimate a premium category.")

with st.form("prediction_form"):
    age = st.number_input("Age (years)", min_value=1, max_value=119, value=30, step=1)
    weight = st.number_input("Weight (kg)", min_value=1.0, max_value=500.0, value=65.0, step=0.5)
    height = st.number_input("Height (m)", min_value=0.5, max_value=2.5, value=1.7, step=0.01)
    income_lpa = st.number_input(
        "Annual income (lakh rupees)", min_value=0.1, max_value=100_000.0, value=10.0, step=0.5
    )
    smoker = st.selectbox("Do you smoke?", options=[False, True], format_func=lambda value: "Yes" if value else "No")
    city = st.text_input("City", value="Mumbai").strip()
    occupation = st.selectbox(
        "Occupation",
        [
            "Government Employee", "Doctor", "Business", "Accountant", "Teacher",
            "Other", "Student", "Software Developer", "Manager", "Engineer",
        ],
    )
    submitted = st.form_submit_button("Predict category", type="primary")

if submitted:
    if not city:
        st.error("Enter a city name.")
    else:
        input_data = {
            "age": int(age),
            "weight": float(weight),
            "height": float(height),
            "income_lpa": float(income_lpa),
            "smoker": smoker,
            "city": city,
            "occupation": occupation,
        }

        try:
            response = requests.post(API_URL, json=input_data, timeout=10)
            if response.ok:
                result = response.json()
                st.success(f"Predicted premium category: **{result['predicted_category']}**")
                st.write(
                    f"BMI: {result['bmi']} · Lifestyle risk: {result['lifestyle_risk']} · "
                    f"Age group: {result['age_group']} · City tier: {result['city_tier']}"
                )
                probabilities = result.get("category_probabilities")
                if probabilities:
                    st.subheader("Model scores")
                    for category, probability in sorted(
                        probabilities.items(), key=lambda item: item[1], reverse=True
                    ):
                        st.write(f"{category}: {probability:.1%}")
                        st.progress(float(probability))
            else:
                try:
                    detail = response.json().get("detail", response.text)
                except ValueError:
                    detail = response.text
                st.error(f"API error ({response.status_code}): {detail}")
        except requests.exceptions.ConnectionError:
            st.error("Could not connect to FastAPI. Start the API on port 8000 and try again.")
        except requests.exceptions.Timeout:
            st.error("The API took too long to respond. Please try again.")
        except requests.exceptions.RequestException as error:
            st.error(f"Request failed: {error}") 
