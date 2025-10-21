Problem statement

Hospital readmission is a major concern in healthcare systems, especially for chronic conditions like diabetes.
The goal of this project is to predict whether a patient will be readmitted to the hospital and, if so, within what timeframe based on their demographic and medical data.

The model classifies each patient encounter into three categories:

-No Readmission

-Readmission within 30 days

-Readmission after 30 days

By identifying high-risk patients early, healthcare teams can design better follow-up plans and reduce avoidable readmissions.

⚙️ Project Workflow

Data Understanding

-Used the UCI Diabetes dataset containing 100,000+ hospital encounters (1999–2008).

-Explored features such as age, admission type, diagnoses, medications, and lab results.

Data Cleaning & Preprocessing

-Removed duplicate and irrelevant features.

-Handled missing and inconsistent data.

-Encoded categorical variables and normalized numerical values.

-Feature Engineering

-Grouped patients into age brackets.

-Derived new features for number of prior visits and medications changed.

Model Training

Compared multiple classifiers:

-Logistic Regression

-Random Forest

-Gradient Boosting

Selected the best-performing model based on accuracy and recall.

Evaluation

-Used confusion matrix and classification report for evaluation.

-Balanced the dataset to reduce class bias.

-Deployment

-Integrated the trained model into a Streamlit web application.

Built a user-friendly interface where patient attributes can be entered to get predictions instantly.

🧠 Tech Stack

Python 3

pandas, numpy, scikit-learn

Streamlit

Matplotlib / Seaborn (for exploratory analysis)

Joblib / Pickle (for model serialization)
