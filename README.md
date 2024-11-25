# Bank Marketing Decision Tree Classifier 

## Project Overview
This data science project implements a decision tree classifier to predict whether a client will subscribe to a term deposit based on demographic and behavioral data from a bank marketing campaign.

## Dataset
The project uses the Bank Marketing dataset from the UCI Machine Learning Repository, which includes:
- Client demographic data (age, job, marital status, education)
- Campaign information (contact type, month, day)
- Economic indicators
- Previous campaign outcomes
- Target variable: whether the client subscribed to a term deposit (yes/no)

## Project Structure
```
PRODIGY_DS_03/
├── data/               # Dataset directory
│   └── bank-marketing.csv
├── notebooks/         # Jupyter notebooks
│   └── bank_marketing_decision_tree.ipynb
├── src/              # Source code
│   └── utils.py      # Utility functions
├── requirements.txt   # Project dependencies
└── README.md         # Project documentation
```

## Features
1. Data Preprocessing
   - Handling categorical variables
   - Feature scaling
   - Train-test splitting

2. Exploratory Data Analysis
   - Feature distributions
   - Target variable analysis
   - Correlation studies

3. Decision Tree Implementation
   - Model training and optimization
   - Feature importance analysis
   - Tree visualization

4. Model Evaluation
   - Accuracy metrics
   - Confusion matrix
   - Classification report

## Technical Stack
- Python 3.12
- Key Libraries:
  * pandas for data manipulation
  * scikit-learn for decision tree implementation
  * matplotlib and seaborn for visualization
  * numpy for numerical operations

## Getting Started

1. Clone the repository:
```bash
git clone [repository-url]
cd PRODIGY_DS_03
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run Jupyter Notebook:
```bash
jupyter notebook
```

4. Open `notebooks/bank_marketing_decision_tree.ipynb` to see the analysis

## Model Features
The decision tree classifier uses various features including:
- Age
- Job type
- Education level
- Marital status
- Previous campaign outcome
- Economic indicators
- Contact type and timing

## Expected Outcomes
- Accurate prediction of term deposit subscriptions
- Identification of key factors influencing customer decisions
- Actionable insights for marketing strategy
- Visual representation of decision paths

## Contributing
Feel free to fork this repository and submit pull requests. For major changes, please open an issue first to discuss what you would like to change.

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments
- Dataset source: [UCI Machine Learning Repository - Bank Marketing Dataset](https://archive.ics.uci.edu/ml/datasets/Bank+Marketing)
- Part of the Prodigy InfoTech Data Science Internship Program
