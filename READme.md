
**Student Performance Prediction**
This project predicts students’ math scores based on demographic and academic factors (gender, race/ethnicity, parental education, lunch type, test preparation course, reading and writing scores). It demonstrates an end‑to‑end machine learning pipeline with training, preprocessing, and deployment using Flask.

**Project Structure**

```
muthupandian02-student_performance_prediction/
├── app.py                # Flask app entry point
├── Dockerfile            # Containerization setup
├── requirements.txt      # Python dependencies
├── setup.py              # Package setup
├── artifact/             # Saved artifacts (model, preprocessor, train/test data)
├── data/                 # Raw dataset (StudentsPerformance.csv)
├── notebook/             # Jupyter notebooks (EDA, experiments)
├── src/                  # Source code
│   ├── exception.py      # Custom exception handling
│   ├── logger.py         # Logging utilities
│   ├── utils.py          # Helper functions
│   ├── components/       # Data ingestion, transformation, training modules
│   └── pipeline/         # Training and prediction pipelines
├── templates/            # HTML templates for Flask UI
│   ├── home.html
│   └── index.html
└── .github/              # CI/CD workflows
```

**Workflow**

1. **Data Ingestion**  
   - Reads raw dataset (`data/StudentsPerformance.csv`).  
   - Splits into `train.csv` and `test.csv` (saved in `artifact/`).  

2. **Data Transformation**  
   - Encodes categorical features (gender, lunch, parental education, etc.).  
   - Scales numeric features (reading_score, writing_score).  
   - Saves preprocessor as `preprocessor.pkl`.  

3. **Model Training**  
   - Trains ML models (Random Forest, ANN, etc.).  
   - Evaluates with accuracy, precision, recall, F1 score.  
   - Saves best model as `model.pkl`.  

4. **Pipeline Orchestration**  
   - `train_pipeline.py` runs ingestion → transformation → training sequentially.  
   - Produces final artifacts (`train.csv`, `test.csv`, `preprocessor.pkl`, `model.pkl`).  

5. **Prediction Pipeline**  
   - `predict_pipeline.py` loads artifacts.  
   - `UserInputData` converts form inputs into a DataFrame.  
   - `PredictPipeline` transforms inputs and predicts math score.  

6. **Flask Web App**  
   - `app.py` serves routes:  
     - `/home` → input form (`index.html`).  
     - `/predictdata` → runs prediction, displays result (`home.html`).  
   - User fills form → pipeline predicts → result shown in browser.  
 Open your browser at **http://127.0.0.1:5000/home**.

 Dataset
- **Source:** `data/StudentsPerformance.csv`  
- Contains student demographics and exam scores.  
- Used for training and evaluation.

**Notebooks**
- `notebook/EDA.ipynb` — Exploratory Data Analysis, visualizations, and initial experiments.
- 
**Future Improvements**
- Add more models and compare performance.  
- Improve UI with CSS/Bootstrap.  
- Deploy to cloud (Heroku, AWS, Azure).  
- Add API endpoints for programmatic predictions.  

**Contact**
**Author:** Muthupandian S  
- 📧 Email: muthupandiansuresh2003@gmail.com  
- 🔗 [LinkedIn](https://www.linkedin.com/in/muthupandian-s-aa3b48239)  
- 💻 [GitHub](https://github.com/Muthupandian02)
