🏥 Diabetes Prediction Model

📌 About the Project
This project uses machine learning to predict whether a person has diabetes based on their medical data. By analyzing key health indicators such as glucose levels, blood pressure, BMI, age, and insulin levels, the model provides an accurate diagnosis, assisting in early detection and preventive healthcare.

📑 Table of Contents


Features
Technologies Used
Installation Guide
Usage
Model Performance
Dataset
Project Structure
Contributing


🚀 Features

✅ Diabetes Prediction – Determines the likelihood of diabetes based on medical attributes.
✅ Data Preprocessing & Cleaning – Handles missing values, outliers, and feature scaling.
✅ Machine Learning Models – Implements Logistic Regression, Random Forest, Gradient Boosting, and more.
✅ Performance Evaluation – Uses accuracy, precision, recall, and F1-score to assess model effectiveness.
✅ Visualization – Includes graphs, correlation heatmaps, and prediction plots for better insights.
✅ User-Friendly Interface – Deployed using Streamlit for interactive predictions.


🛠 Technologies Used
Python 🐍

Scikit-learn 🤖

Pandas & NumPy 📊

Matplotlib & Seaborn 📉

Streamlit 🌐 (for deployment)


⚙ Installation Guide
🔹 Prerequisites
Ensure you have Python 3.7+ installed. You can download it from Python’s official website.

🔹 Clone the Repository
bash
Copy
Edit
git clone https://github.com/pranjalraturi/Diabetes-Prediction
cd diabetes-prediction-model
🔹 Install Required Packages
bash
Copy
Edit
pip install -r requirements.txt


🖥 Usage
Running the Model
To train the model and evaluate performance, run:

bash
Copy
Edit
python train.py
Running the Web App
To launch the Streamlit-based UI, run:

bash
Copy
Edit
streamlit run app.py


📈 Model Performance
Among multiple tested models, the Support Vector Machine achieved the highest accuracy of 91.45%, making it the most reliable for predicting diabetes.

Model	Accuracy (%)

Support Vector Machine (SVM)	90.79
Gradient Boosting Classifier	91.45
Random Forest Classifier	90.13
Logistic Regression	89.47

📊 Dataset
The dataset used in this project is the Pima Indians Diabetes Dataset, obtained from the UCI Machine Learning Repository. It contains 768 samples with 8 medical features and a binary target variable (0 = No Diabetes, 1 = Diabetes).

Features:


Glucose Level
Blood Pressure
BMI (Body Mass Index)
Insulin Level
Age
Skin Thickness
Pregnancies
Diabetes Pedigree Function


📂 Project Structure
bash
Copy
Edit
📦 diabetes-prediction-model
├── 📂 data
│   ├── diabetes.csv
├── 📂 models
│   ├── trained_model.pkl
├── app.py              # Streamlit Web App
├── train.py            # Model Training Script
├── requirements.txt    # Dependencies
├── README.md           # Project Documentation
└── LICENSE             # License File


🤝 Contributing
Contributions are welcome! To contribute:

Fork the repository.

Clone it to your local machine.

Create a new branch (feature-branch).

Commit your changes.

Push to your fork and submit a Pull Request.
