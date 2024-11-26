import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt

# Title and Description
st.title("Cricket Player Performance Analysis and Prediction")
st.markdown("""
Analyze and predict cricket player performance in ODIs using advanced visualizations and multiple machine learning algorithms.
""")

# File Upload
uploaded_file = st.file_uploader("Upload a CSV file", type="csv")

if uploaded_file:
    # Load the dataset
    data = pd.read_csv(uploaded_file)
    st.write("### Dataset Preview")
    st.dataframe(data.head())

    # Data Cleaning
    st.write("### Data Cleaning")
    if st.checkbox("Fill missing values (numerical columns with mean)"):
        numeric_cols = data.select_dtypes(include=["float64", "int64"]).columns
        data[numeric_cols] = data[numeric_cols].fillna(data[numeric_cols].mean())
        non_numeric_cols = data.select_dtypes(exclude=["float64", "int64"]).columns
        data[non_numeric_cols] = data[non_numeric_cols].fillna("Unknown")
        st.success("Missing values filled!")

    # Dropping rows with missing values (Approach 2)
    if st.checkbox("Drop rows with missing values"):
        data = data.dropna()  # This removes any row with missing values in any column
        st.success("Rows with missing values dropped!")

    # Exploratory Data Analysis (EDA)
    st.write("### Exploratory Data Analysis")
    if st.checkbox("Show summary statistics"):
        st.write(data.describe())

    if st.checkbox("Show histograms for numerical columns"):
        numeric_cols = data.select_dtypes(include=['float64', 'int64']).columns
        for col in numeric_cols:
            st.write(f"Histogram for {col}")
            fig = px.histogram(data, x=col, title=f"Distribution of {col}")
            st.plotly_chart(fig)

    if st.checkbox("Show scatter plot (Bivariate Analysis)"):
        x_col = st.selectbox("Select X-axis column:", data.columns)
        y_col = st.selectbox("Select Y-axis column:", data.columns)
        if x_col and y_col:
            fig = px.scatter(data, x=x_col, y=y_col, title=f"Scatter Plot: {x_col} vs {y_col}")
            st.plotly_chart(fig)

    if st.checkbox("Show correlation heatmap"):
        corr_matrix = data.corr()
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', ax=ax)
        st.pyplot(fig)

    # Machine Learning
    st.write("### Machine Learning")
    target_column = st.selectbox("Select the target column for prediction:", data.columns)

    if target_column:
        features = data.drop(columns=[target_column]).select_dtypes(include=['float64', 'int64'])
        target = data[target_column]

        # Train-Test Split
        test_size = st.slider("Test Set Size (as a percentage):", 10, 50, 20)
        X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=test_size/100, random_state=42)

        # Algorithm Selection
        model_choice = st.selectbox("Select an Algorithm:", [
            "Logistic Regression", "k-Nearest Neighbors", "Naive Bayes", 
            "Decision Tree", "Random Forest", "Support Vector Machine (SVM)"
        ])

        # Hyperparameter Tuning
        model = None
        if model_choice == "Logistic Regression":
            st.write("### Logistic Regression")
            C = st.slider("Regularization Strength (C):", 0.01, 10.0, 1.0)
            model = LogisticRegression(C=C, random_state=42)
        elif model_choice == "k-Nearest Neighbors":
            st.write("### k-Nearest Neighbors")
            n_neighbors = st.slider("Number of Neighbors (k):", 1, 15, 5)
            model = KNeighborsClassifier(n_neighbors=n_neighbors)
        elif model_choice == "Naive Bayes":
            st.write("### Naive Bayes")
            model = GaussianNB()
        elif model_choice == "Decision Tree":
            st.write("### Decision Tree")
            max_depth = st.slider("Maximum Depth:", 1, 20, 5)
            model = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
        elif model_choice == "Random Forest":
            st.write("### Random Forest")
            n_estimators = st.slider("Number of Trees:", 10, 200, 100)
            model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
        elif model_choice == "Support Vector Machine (SVM)":
            st.write("### Support Vector Machine (SVM)")
            kernel = st.selectbox("Kernel Type:", ["linear", "rbf", "poly"])
            C = st.slider("Regularization Strength (C):", 0.01, 10.0, 1.0)
            model = SVC(kernel=kernel, C=C, random_state=42)

        if st.button("Train Model"):
            # Initialize and fit scaler here
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            # Train the model
            model.fit(X_train_scaled, y_train)
            predictions = model.predict(X_test_scaled)

            st.write(f"### Accuracy: {accuracy_score(y_test, predictions):.2%}")
            st.text("Classification Report:")
            st.text(classification_report(y_test, predictions))

            # Confusion Matrix
            cm = confusion_matrix(y_test, predictions)
            fig, ax = plt.subplots()
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
            st.pyplot(fig)

        # Make Predictions
        st.write("### Make Predictions")
        user_input = {}
        for col in features.columns:
            user_input[col] = st.number_input(f"Enter value for {col}:", value=float(features[col].mean()))

        if st.button("Predict"):
            # Ensure input is scaled using the same scaler
            input_df = pd.DataFrame([user_input])
            input_scaled = scaler.transform(input_df)
            prediction = model.predict(input_scaled)
            st.success(f"Prediction: {prediction[0]}")

else:
    st.info("Please upload a CSV file to proceed.")
