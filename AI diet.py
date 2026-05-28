import streamlit as st
import pandas as pd
import os
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

# Set up page configuration
st.set_page_config(page_title="Bangladeshi Meal Planner ML", layout="wide")

# Load dataset securely
@st.cache_data
def load_data():
    file_name = "Fully_Cleaned_Food_Database_ML.csv"
    if os.path.exists(file_name):
        return pd.read_csv(file_name)
    else:
        # Fallback to CSV if Excel is missing
        csv_name = "Fully_Cleaned_Bangladeshi_Food_Database_ML.csv"
        if os.path.exists(csv_name):
            return pd.read_csv(csv_name)
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

    meal_selection = st.selectbox("Choose a Meal Type", ["Breakfast", "Lunch", "Dinner", "Snack"])

    # --- Prominent Run Button ---
    run_clicked = st.button("🚀 Generate My Meal Plan", type="primary", use_container_width=True)

    if run_clicked:
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

        # Target Calorie Constraint Configuration with Safety Floors
        if goal == "Weight Loss":
            # Never let calories drop below 1200 for women or 1500 for men
            safe_floor = 1500 if gender == "Male" else 1200
            target_calories = max(safe_floor, tdee - 500)
        elif goal == "Weight Gain":
            target_calories = tdee + 300
        else:
            target_calories = tdee

        # Display Metrics Dashboard
        st.subheader("📊 Your Target Metrics")
        metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
        
        metric_col1.metric("BMI", f"{bmi:.1f}")
        metric_col2.metric("BMR", f"{int(bmr)} kcal")
        metric_col3.metric("TDEE", f"{int(tdee)} kcal")
        metric_col4.metric("Daily Target", f"{int(target_calories)} kcal")

        st.markdown("---")

        # 2. Recommendation Engine
        st.subheader("🍽️ Interactive Food Recommender")
        
        # Define macro distribution criteria per meal type
        meal_shares = {"Breakfast": 0.30, "Lunch": 0.40, "Dinner": 0.20, "Snack": 0.10}
        meal_target_cals = target_calories * meal_shares[meal_selection]
        
        # Goal-aware macro distribution (protein/carb/fat % of calories)
        if goal == "Weight Loss":
            protein_pct, carb_pct, fat_pct = 0.35, 0.40, 0.25
        elif goal == "Weight Gain":
            protein_pct, carb_pct, fat_pct = 0.20, 0.55, 0.25
        else:
            protein_pct, carb_pct, fat_pct = 0.25, 0.50, 0.25

        protein_target = (meal_target_cals * protein_pct) / 4   # 4 kcal/g
        fat_target     = (meal_target_cals * fat_pct)     / 9   # 9 kcal/g
        carb_target    = (meal_target_cals * carb_pct)    / 4   # 4 kcal/g

        st.info(f"Target for {meal_selection}: **{int(meal_target_cals)} kcal** | "
                f"Protein: {int(protein_target)}g ({int(protein_pct*100)}%) | "
                f"Carbs: {int(carb_target)}g ({int(carb_pct*100)}%) | "
                f"Fat: {int(fat_target)}g ({int(fat_pct*100)}%)")

        # Filter by meal tag
        tag_mapping = f"is_{meal_selection.lower()}"
        filtered_df = df[df[tag_mapping] == 1].copy().reset_index(drop=True)

        if not filtered_df.empty:
            features = ['Energy (kcal)', 'Protein (g)', 'Fat (g)', 'Carbs (g)']
            target_vector = np.array([meal_target_cals, protein_target, fat_target, carb_target])

            # --- Score each food by macro RATIO fit ---
            def macro_ratios(row):
                total_cal = max(row['Energy (kcal)'], 1)
                p_cal = row['Protein (g)'] * 4
                c_cal = row['Carbs (g)'] * 4
                f_cal = row['Fat (g)'] * 9
                return np.array([p_cal / total_cal, c_cal / total_cal, f_cal / total_cal])

            goal_ratios = np.array([protein_pct, carb_pct, fat_pct])

            scores = []
            for _, row in filtered_df.iterrows():
                food_ratios = macro_ratios(row)
                dot = np.dot(food_ratios, goal_ratios)
                norm_a = np.linalg.norm(food_ratios)
                norm_b = np.linalg.norm(goal_ratios)
                similarity = dot / (norm_a * norm_b + 1e-9)
                scores.append(similarity)
            
            filtered_df['goal_score'] = scores

            # --- KNN on normalized features for diversity ---
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(filtered_df[features])
            target_scaled = scaler.transform([target_vector])

            knn = NearestNeighbors(n_neighbors=min(20, len(filtered_df)), metric='euclidean')
            knn.fit(X_scaled)
            distances, indices = knn.kneighbors(target_scaled)

            # Combine KNN rank with goal_score to pick the best items
            knn_candidates = filtered_df.iloc[indices[0]].copy()
            knn_candidates['knn_rank'] = range(len(knn_candidates))
            knn_candidates['knn_rank_norm'] = 1 - (knn_candidates['knn_rank'] / max(len(knn_candidates) - 1, 1))
            knn_candidates['combined_score'] = (
                0.6 * knn_candidates['goal_score'] +
                0.4 * knn_candidates['knn_rank_norm']
            )
            knn_candidates = knn_candidates.sort_values('combined_score', ascending=False)

            # --- Dynamic plate sizing to prevent unrealistic serving sizes ---
            if meal_selection == "Snack":
                n_items = 2 if meal_target_cals < 250 else 3
            else:
                if meal_target_cals < 400:
                    n_items = 3   # Small meals get fewer items
                elif meal_target_cals > 1000:
                    n_items = 6   # Huge meals get high variety
                elif meal_target_cals > 700:
                    n_items = 5   # Large meals get moderate variety
                else:
                    n_items = 4   # Standard meals

            # --- Enforce Category Diversity ---
            selected_indices = []
            seen_categories = set()

            # Pass 1: Try to pick items from unique categories
            for idx, row in knn_candidates.iterrows():
                cat = row['Category']
                if cat not in seen_categories:
                    selected_indices.append(idx)
                    seen_categories.add(cat)
                
                if len(selected_indices) == n_items:
                    break
            
            # Pass 2: If we ran out of unique categories, fill the rest with the next best items
            if len(selected_indices) < n_items:
                for idx, row in knn_candidates.iterrows():
                    if idx not in selected_indices:
                        selected_indices.append(idx)
                    if len(selected_indices) == n_items:
                        break

            selected = knn_candidates.loc[selected_indices].copy()

            # --- Compute serving sizes so the combo reaches the calorie target ---
            total_score = selected['combined_score'].sum()
            selected['cal_share'] = (selected['combined_score'] / total_score) * meal_target_cals
            selected['servings'] = selected['cal_share'] / selected['Energy (kcal)'].clip(lower=1)

            # --- Helper: convert grams to household measurements ---
            HOUSEHOLD_UNITS = {
                'Rice, white, polished, boiled': ('cup cooked rice', 160),
                'Rice, brown, boiled': ('cup cooked rice', 160),
                'Rice, puffed': ('cup muri', 15),
                'Plain Khichuri': ('cup khichuri', 200),
                'Vermicelli, boiled': ('cup semai', 140),
                'Puffed rice mix': ('cup jhalmuri', 30),
                'Roti / Flatbread': ('piece roti', 40),
                'Paratha': ('piece paratha', 80),
                'Naan bread': ('piece naan', 90),
                'Lentil, boiled': ('cup dal', 200),
                'Green gram, boiled': ('cup mung dal', 200),
                'Black gram, boiled': ('cup mashkalai dal', 200),
                'Bengal gram, boiled': ('cup chola', 160),
                'Yellow pigeon peas': ('cup arhar dal', 200),
                'Brinjal, boiled': ('cup begun', 150),
                'Pumpkin, boiled': ('cup kumra', 150),
                'Gourd, pointed, boiled': ('cup potol', 150),
                'Gourd, bitter, boiled': ('cup korola', 120),
                'Okra, boiled': ('cup bhindi', 100),
                'Tomato, boiled': ('medium tomato', 120),
                'Potato Mash': ('cup mashed alu', 210),
                'Cauliflower, boiled': ('cup fulkopi', 130),
                'Cabbage, boiled': ('cup badhakopi', 150),
                'Bottle gourd, boiled': ('cup lau', 150),
                'Green papaya, boiled': ('cup kancha pepe', 150),
                'Snake gourd': ('cup chichinga', 150),
                'Ridge gourd': ('cup jhinga', 150),
                'Amaranth leaves, red': ('cup lal shak', 60),
                'Spinach, boiled': ('cup palong shak', 60),
                'Indian spinach': ('cup pui shak', 60),
                'Water amaranth': ('cup kolmi shak', 60),
                'Moringa leaves, raw': ('cup moringa pata', 40),
                'Jute leaves': ('cup pat shak', 60),
                'Rohu fish, cooked': ('piece rui mach', 85),
                'Pangas fish, cooked': ('piece pangas', 85),
                'Tilapia fish, cooked': ('piece tilapia', 85),
                'Hilsa fish, cooked': ('piece ilish', 85),
                'Climbing perch': ('piece koi mach', 60),
                'Mola carplet': ('cup mola mach', 50),
                'Walking catfish': ('piece magur', 80),
                'Chicken breast, cooked': ('piece chicken breast', 100),
                'Beef, curry': ('cup beef curry', 150),
                'Mutton, curry': ('cup mutton curry', 150),
                'Egg, chicken, boiled': ('boiled egg', 50),
                'Egg, duck, boiled': ('duck egg', 60),
                'Samosa': ('piece samosa', 50),
                'Pitha, Bhapa': ('piece bhapa pitha', 80),
                'Fuchka / Panipuri': ('piece fuchka', 30),
                'Chotpoti': ('cup chotpoti', 200),
                'Beef Halim': ('cup halim', 250),
                'Milk, cow': ('glass milk', 250),
                'Yogurt, plain': ('cup doi', 150),
                'Yogurt, sweet': ('cup mishti doi', 150),
                'Banana': ('medium banana', 120),
                'Mango, ripe': ('cup mango slices', 165),
                'Jackfruit, ripe': ('cup kathal', 150),
                'Guava': ('medium guava', 100),
                'Papaya, ripe': ('cup papaya', 145),
                'Pineapple': ('cup pineapple', 165),
                'Litchi': ('piece litchi', 10),
            }

            def grams_to_household(food_name, grams):
                if food_name in HOUSEHOLD_UNITS:
                    unit_name, unit_grams = HOUSEHOLD_UNITS[food_name]
                    qty = grams / unit_grams
                    if qty < 0.3:
                        return f"{int(grams)}g"
                    elif qty < 1:
                        if qty <= 0.34:
                            frac = "⅓"
                        elif qty <= 0.55:
                            frac = "½"
                        elif qty <= 0.8:
                            frac = "¾"
                        else:
                            frac = "1"
                        return f"{frac} {unit_name}"
                    elif qty < 1.3:
                        return f"1 {unit_name}"
                    elif qty < 1.7:
                        return f"1½ {unit_name}"
                    elif qty < 2.3:
                        return f"2 {unit_name}"
                    elif qty < 2.7:
                        return f"2½ {unit_name}"
                    elif qty < 3.3:
                        return f"3 {unit_name}"
                    else:
                        return f"{qty:.0f} {unit_name}"
                else:
                    return f"{int(grams)}g"

            # Display the recommended meal combo
            st.write("### ✅ Recommended Meal Combo")
            st.caption(f"*{n_items} items combined to reach ~{int(meal_target_cals)} kcal for your **{goal.lower()}** goal*")

            display_cols = st.columns(len(selected))
            for idx, (col_element, (_, food_item)) in enumerate(zip(display_cols, selected.iterrows())):
                servings = food_item['servings']
                grams = servings * 100  # dataset is per 100g
                amount_str = grams_to_household(food_item['English Name'], grams)
                with col_element:
                    st.markdown(f"#### {food_item['English Name']}")
                    st.write(f"*{food_item['Bengali Name']}*")
                    st.markdown(f"### 🍽️ {amount_str}")
                    st.caption(f"({int(grams)}g)")
                    actual_cal = food_item['Energy (kcal)'] * servings
                    actual_protein = food_item['Protein (g)'] * servings
                    actual_carbs = food_item['Carbs (g)'] * servings
                    actual_fat = food_item['Fat (g)'] * servings
                    st.success(
                        f"**{int(actual_cal)} kcal**\n\n"
                        f"Protein: {actual_protein:.1f}g\n\n"
                        f"Carbs: {actual_carbs:.1f}g\n\n"
                        f"Fat: {actual_fat:.1f}g"
                    )
                    match_pct = int(food_item['goal_score'] * 100)
                    if match_pct >= 90:
                        st.markdown(f"🟢 **{match_pct}%** match")
                    elif match_pct >= 70:
                        st.markdown(f"🟡 **{match_pct}%** match")
                    else:
                        st.markdown(f"🔴 **{match_pct}%** match")

            # --- Summary table ---
            st.markdown("---")
            st.write("### 📋 Meal Summary")
            summary_data = []
            for _, item in selected.iterrows():
                s = item['servings']
                g = s * 100
                summary_data.append({
                    'Food': item['English Name'],
                    'Amount': grams_to_household(item['English Name'], g),
                    'Grams': f"{int(g)}g",
                    'Calories': int(item['Energy (kcal)'] * s),
                    'Protein (g)': round(item['Protein (g)'] * s, 1),
                    'Carbs (g)': round(item['Carbs (g)'] * s, 1),
                    'Fat (g)': round(item['Fat (g)'] * s, 1),
                })
            summary_df = pd.DataFrame(summary_data)
            
            totals = {
                'Food': '📊 TOTAL',
                'Amount': '',
                'Grams': '',
                'Calories': summary_df['Calories'].sum(),
                'Protein (g)': summary_df['Protein (g)'].sum(),
                'Carbs (g)': summary_df['Carbs (g)'].sum(),
                'Fat (g)': summary_df['Fat (g)'].sum(),
            }
            summary_df = pd.concat([summary_df, pd.DataFrame([totals])], ignore_index=True)
            st.dataframe(summary_df, use_container_width=True, hide_index=True)

            actual_total_cal = totals['Calories']
            st.write(f"**Target:** {int(meal_target_cals)} kcal → **Actual:** {int(actual_total_cal)} kcal "
                     f"({'✅ on track' if abs(actual_total_cal - meal_target_cals) < 50 else '⚠️ slight deviation'})")

        else:
            st.warning(f"No database entries match the tag constraints for {meal_selection}.")
else:
    st.markdown("---")
    st.markdown(
        "<div style='text-align:center; padding:40px 20px; color:#888;'>"
        "<h3>👆 Fill in your details in the sidebar, pick a meal type, then click the button above!</h3>"
        "</div>",
        unsafe_allow_html=True,
    )
