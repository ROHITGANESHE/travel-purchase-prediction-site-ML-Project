<h1 align="center">🌍 Travel Package Purchase Prediction Site</h1>

<p align="center">
A production-ready <b>Machine Learning web application</b> that predicts whether a customer will purchase a travel package based on demographic, behavioral, and sales interaction features.
</p>

<p align="center">
Built using <b>Scikit-learn + Streamlit</b>, the application provides real-time predictions with probability scores to help businesses improve marketing conversion strategies.
</p>

<hr>

<h2>📌 Project Overview</h2>

<p>
Travel companies often struggle to identify which customers are most likely to convert after a sales pitch.
</p>

<p>This project applies <b>Supervised Machine Learning (Classification)</b> to predict customer purchase behavior and help:</p>

<ul>
<li>🎯 Improve targeted marketing</li>
<li>📈 Increase conversion rates</li>
<li>💰 Optimize sales efforts</li>
</ul>

<p>
The application allows users to enter customer details and instantly get a prediction.
</p>

<hr>

<h2>❓ Problem Statement</h2>

<p>
Given customer demographic, behavioral, and pitch-related features:
</p>

<p><b>👉 Predict whether a customer will purchase a travel package (Yes / No).</b></p>

<hr>

<h2>🛠 Solution Approach</h2>

<ol>
<li>Data Cleaning & Preprocessing</li>
<li>Feature Engineering</li>
<li>Handling categorical & numerical data using ColumnTransformer</li>
<li>Model Training & Evaluation</li>
<li>Saving trained model & preprocessor</li>
<li>Building interactive Streamlit Web App</li>
</ol>

<hr>

<h2>📊 Features Used</h2>

<h3>🔢 Numerical Features</h3>
<ul>
<li>Age</li>
<li>Monthly Income</li>
<li>Duration of Pitch</li>
<li>Number of Followups</li>
<li>Number of Trips</li>
<li>Preferred Property Star</li>
<li>Pitch Satisfaction Score</li>
<li>Number of Persons Visiting</li>
<li>Number of Children Visiting</li>
</ul>

<h3>🧾 Categorical Features</h3>
<ul>
<li>Gender</li>
<li>Marital Status</li>
<li>Occupation</li>
<li>Type of Contact</li>
<li>Product Pitched</li>
<li>Designation</li>
<li>City Tier</li>
<li>Passport</li>
<li>Own Car</li>
</ul>

<hr>

<h2>🧠 Model & Preprocessing</h2>

<ul>
<li>Classification model trained using <b>Scikit-learn</b></li>
<li><b>ColumnTransformer</b> used for:
    <ul>
        <li>Scaling numerical features</li>
        <li>Encoding categorical features</li>
    </ul>
</li>
<li>Saved artifacts:
    <ul>
        <li>tourism_model.pkl</li>
        <li>preprocessor.pkl</li>
        <li>.joblib versions for production use</li>
    </ul>
</li>
</ul>

<p>
The same preprocessing pipeline is reused during inference to ensure consistency.
</p>

<hr>

<h2>🌐 Web Application (Streamlit)</h2>

<ul>
<li>✔ Clean dark-themed UI</li>
<li>✔ 3-column structured input layout</li>
<li>✔ Controlled inputs (dropdowns & numeric fields)</li>
<li>✔ Centered prediction button</li>
<li>✔ Displays prediction + probability score</li>
<li>✔ Celebration animation for positive predictions 🎉</li>
</ul>

<hr>

<h2>🧰 Tech Stack</h2>

<ul>
<li><b>Language:</b> Python</li>
<li><b>Libraries:</b> Pandas, NumPy, Scikit-learn</li>
<li><b>Web Framework:</b> Streamlit</li>
<li><b>Model Persistence:</b> Pickle / Joblib</li>
<li><b>Version Control:</b> Git & GitHub</li>
</ul>

<hr>

<h2>📁 Project Structure</h2>

<pre>
Travel-Package-Prediction/
│
├── app
│   ├── EDA.py
│   └── ml.py
│
├── data
│   └── traveling_data.csv
│
├── model_building
│   └── model_building.py
│
├── notebooks
│   ├── EDA.ipynb
│   ├── model_building.ipynb
│   └── domain_knowledge.ipynb
│
├── pkl
│   ├── tourism_model.pkl
│   └── preprocessor.pkl
│
├── README.md
└── requirements.txt
</pre>

<hr>

<h2>▶️ How to Run the Project</h2>

<ol>
<li><b>Clone the Repository</b></li>
</ol>

<pre>
git clone https://github.com/YOUR-USERNAME/Travel-Package-Prediction.git
cd Travel-Package-Prediction
</pre>

<ol start="2">
<li><b>Create Virtual Environment (Optional)</b></li>
</ol>

<pre>
python -m venv myenv
myenv\Scripts\activate   # Windows
</pre>

<ol start="3">
<li><b>Install Dependencies</b></li>
</ol>

<pre>
pip install -r requirements.txt
</pre>

<ol start="4">
<li><b>Run the Application</b></li>
</ol>

<pre>
streamlit run app/ml.py
</pre>

<hr>

<h2>📈 Results</h2>

<ul>
<li>Predicts customer purchase behavior</li>
<li>Displays probability score</li>
<li>Helps sales teams prioritize high-potential leads</li>
<li>Demonstrates end-to-end ML deployment</li>
</ul>

<hr>

<h2>🚀 Business Impact</h2>

<ul>
<li>✔ Reduces marketing cost</li>
<li>✔ Improves sales conversion rate</li>
<li>✔ Enables data-driven decision making</li>
</ul>

<hr>

<h2>👨‍💻 Author</h2>

<p>
<b>Rohit Ganeshe</b><br>
Data Science | Machine Learning | GenAI Enthusiast
</p>

<p>
🔗 GitHub:(https://github.com/ROHITGANESHE/travel-purchase-prediction-site-ML-Project.git)<br>
🔗 LinkedIn:(https://www.linkedin.com/in/rohit-ganeshe-rsg030/)
</p>
