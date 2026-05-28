# Hybrid Diet Recommendation System: Integrating Food Engineering Constraints with K-Nearest Neighbors

This repository contains a hybrid machine learning and deterministic application designed to generate localized, macronutrient-optimized meal plans. By combining established nutritional engineering formulas with a vector-based recommendation engine, the system processes user basal metrics and filters regional food profiles to deliver precise, target-aligned dietary frameworks.

The dataset is curated and feature-engineered specifically using regional nutritional profiles, providing an authentic application of data science to localized food systems.

---

## System Architecture

The application bypasses the limitations of purely statistical models by utilizing a two-phase hybrid pipeline:

### 1. Deterministic Rule Engine (Basal Metrics)
Before executing predictive or filtering algorithms, the system establishes individual biological targets using exact physiological formulas.

* **Body Mass Index (BMI):** Calculated to baseline user health metrics.
* **Basal Metabolic Rate (BMR):** Determined via the modern Mifflin-St Jeor equation, chosen for its validated accuracy in clinical and field settings.
* **Total Daily Energy Expenditure (TDEE):** Quantified by applying physical activity multipliers to the calculated BMR baseline.
* **Caloric Targeting:** Configured programmatically based on user objectives (e.g., a continuous deficit constraint of 500 kcal for mass reduction or a 300 kcal surplus for mass gain).

### 2. Machine Learning Recommendation Engine
Once target macronutrient constraints are set, the system passes these vectors to a K-Nearest Neighbors (KNN) algorithm. The model evaluates the mathematical distance between the target meal profile and the available food matrix to output the closest structural matches.

---

## Data Engineering & Localization

A core feature of this project is the integration of localized food profiles, derived from the official *Food Composition Table for Bangladesh*. Standard open-source nutrition datasets often fail to reflect regional eating habits accurately, rendering general models impractical for local deployment.

### Feature Engineering Details
To optimize the recommendation matrix, the raw food composition data was engineered with custom binary constraint tags:
* **Meal Classification Flags:** `is_breakfast`, `is_lunch`, `is_dinner`, and `is_snack`. These prevent the model from making culturally incompatible recommendations (e.g., suggesting heavy carbohydrate main courses for breakfast or traditional street snacks for lunch).
* **Dietary Profiling Flags:** `is_vegetarian`, `is_diabetic_friendly` (low glycemic index tracking), and `is_high_protein`. These act as hard constraints to filter the dataset before calculating vector distances.

---

## Mathematical Formulation

The system utilizes the following formal equations outside the machine learning pipeline to establish the target constraints:

### Body Mass Index
$$\text{BMI} = \frac{\text{Weight (kg)}}{\text{Height (m)}^2}$$

### Basal Metabolic Rate (Mifflin-St Jeor)
$$\text{BMR}_{\text{Male}} = (10 \times \text{Weight}_{\text{kg}}) + (6.25 \times \text{Height}_{\text{cm}}) - (5 \times \text{Age}_{\text{years}}) + 5$$

$$\text{BMR}_{\text{Female}} = (10 \times \text{Weight}_{\text{kg}}) + (6.25 \times \text{Height}_{\text{cm}}) - (5 \times \text{Age}_{\text{years}}) - 161$$

### Vector Distance Metric
The KNN model utilizes **Cosine Similarity** to isolate food recommendations. This metric evaluates the angular distance between vectors rather than absolute magnitudes, ensuring the *ratio* of protein, fat, and carbohydrates matches the nutritional target, regardless of portion sizes.

$$\text{Cosine Similarity} = \frac{\mathbf{A} \cdot \mathbf{B}}{\|\mathbf{A}\| \|\mathbf{B}\|}$$

---

## Technology Stack

* **Language:** Python
* **Framework:** Streamlit (Deployment and Interactive Interface)
* **Data Manipulation:** Pandas, NumPy
* **Machine Learning:** Scikit-learn (NearestNeighbors unsupervised learner)

---

## Installation & Deployment

To run the pipeline and interactive web interface locally, ensure you have Python installed, then follow these steps:

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/hybrid-diet-ml-planner.git
   cd hybrid-diet-ml-planner
   ```

2. **Install dependencies:**
   ```bash
   pip install streamlit pandas scikit-learn openpyxl numpy
   ```

3. **Run the application:**
   ```bash
   streamlit run "AI diet.py"
   ```
   Or on Windows, simply double-click `run.bat`.

4. The application will launch in your default browser at `http://localhost:8501`.

---

## Project Structure

```
├── AI diet.py                                  # Main application script
├── Expanded_Bangladeshi_Food_Database_ML.xlsx  # Curated regional food dataset
├── FCT_10_2_14_final_version.pdf               # Reference: Food Composition Table for Bangladesh
├── run.bat                                     # Windows quick-launch script
├── .gitignore                                  # Git ignore configuration
└── README.md                                   # Project documentation
```

---

## License

This project is developed for academic and research purposes.
