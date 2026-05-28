import streamlit as st
import pandas as pd
import os
from sklearn.neighbors import NearestNeighbors

# Set up page configuration
st.set_page_config(page_title="Bangladeshi Meal Planner ML", layout="wide")

# Load dataset securely
@st.cache_data
def load_data():
    file_name = "Expanded_Bangladeshi_Food_Database_ML.xlsx"
    if os.path.exists(file_name):
        return pd.read_excel(file_name)
    else:
        st.error(f"Dataset file '{file_name}' not found. Please make sure it is in the same directory.")
        return None

df = load_data()

st.title("Machine Learning Meal Planner")
st.write("A hybrid approach using mathematical basal calculations and KNN-based food recommendations.")

if df is not None:
    # Sidebar for user profile inputs
    st.sidebar.header("User Profile Settings")
    gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
    age = st.sidebar.number_input("Age", min_value=1, max_value=120, value=24)
    weight = st.sidebar.number_input("Weight (kg)", min_value=10.0, max_value=200.0, value=70.0)
    height = st.sidebar.number_input("Height (cm)", min_value=50.0, max_value=250.0, value=175.0)
    
    activity = st.sidebar.selectbox(
        "Activity Level", 
        ["Sedentary", "Lightly Active", "Moderately Active", "Very Active"]
    )
    
    goal = st.sidebar.selectbox(
        "Fitness Goal", 
        ["Weight Loss", "Maintenance", "Weight Gain"]
    )

    # 1. Deterministic Logic: Basal Metrics
    height_m = height / 100
    bmi = weight / (height_m ** 2)

    # Mifflin-St Jeor Equation
    if gender == "Male":
        bmr = (10 * weight) + (6.25 * height) - (5 * age) + 5
    else:
        bmr = (10 * weight) + (6.25 * height) - (5 * age) - 161

    activity_multipliers = {
        "Sedentary": 1.2,
        "Lightly Active": 1.375,
        "Moderately Active": 1.55,
        "Very Active": 1.725
    }
    tdee = bmr * activity_multipliers[activity]

    # Target Calorie Constraint Configuration
    if goal == "Weight Loss":
        target_calories = tdee - 500
    elif goal == "Weight Gain":
        target_calories = tdee + 300
    else:
        target_calories = tdee

    # Display Metrics Dashboard
    st.subheader("Your Target Metrics")
    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
    
    metric_col1.metric("BMI", f"{bmi:.1f}")
    metric_col2.metric("BMR", f"{int(bmr)} kcal")
    metric_col3.metric("TDEE", f"{int(tdee)} kcal")
    metric_col4.metric("Daily Target", f"{int(target_calories)} kcal")

    st.markdown("---")

    # 2. Recommendation Engine
    st.subheader("Interactive Food Recommender")
    
    meal_selection = st.selectbox("Choose a Meal Type", ["Breakfast", "Lunch", "Dinner", "Snack"])
    
    # Define macro distribution criteria per meal type
    meal_shares = {"Breakfast": 0.30, "Lunch": 0.40, "Dinner": 0.20, "Snack": 0.10}
    meal_target_cals = target_calories * meal_shares[meal_selection]
    
    # Balanced distribution assumptions for target coordinates
    protein_target = (meal_target_cals * 0.25) / 4
    fat_target = (meal_target_cals * 0.25) / 9
    carb_target = (meal_target_cals * 0.50) / 4

    st.info(f"Target profile for {meal_selection}: **{int(meal_target_cals)} kcal** | Protein: {int(protein_target)}g | Carbs: {int(carb_target)}g | Fat: {int(fat_target)}g")

    # Filter by structural tag column mapping
    tag_mapping = f"is_{meal_selection.lower()}"
    filtered_df = df[df[tag_mapping] == 1].reset_index(drop=True)

    if not filtered_df.empty:
        # Features mapping matrix
        features = ['Energy (kcal)', 'Protein (g)', 'Fat (g)', 'Carbs (g)']
        X = filtered_df[features]
        
        # Fit model dynamically to current subsets
        knn = NearestNeighbors(n_neighbors=min(4, len(filtered_df)), metric='cosine')
        knn.fit(X)
        
        input_vector = pd.DataFrame([[meal_target_cals, protein_target, fat_target, carb_target]], columns=features)
        distances, indices = knn.kneighbors(input_vector)
        
        st.write("### Recommended Local Items")
        
        display_cols = st.columns(len(indices[0]))
        for idx, col_element in enumerate(display_cols):
            row_idx = indices[0][idx]
            food_item = filtered_df.iloc[row_idx]
            
            with col_element:
                st.markdown(f"#### {food_item['English Name']}")
                st.write(f"*{food_item['Bengali Name']}*")
                st.write(f"Category: {food_item['Category']}")
                st.success(f"**Energy:** {food_item['Energy (kcal)']} kcal\n\n"
                           f"**Protein:** {food_item['Protein (g)']}g\n\n"
                           f"**Carbs:** {food_item['Carbs (g)']}g\n\n"
                           f"**Fat:** {food_item['Fat (g)']}g")
    else:
        st.warning(f"No database entries matches the tag constraints for {meal_selection}.")