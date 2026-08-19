import pandas as pd
import numpy as np
import os
import re
import time
import sys

# Classifier Imports
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

# ==========================================
# ENVIRONMENT CONFIGURATION
# ==========================================
# N_JOBS=1 for sequential visualization of combinations and to avoid errors in VS Code
N_JOBS = 1 

# ==========================================
# SUPPORT FUNCTIONS
# ==========================================
def extract_sample(filename):
    numbers = re.findall(r'\d+', str(filename))
    return int(numbers[0]) if numbers else 0

# ==========================================
# FILE CONFIGURATION
# ==========================================
INPUT_FILE = os.path.join('DataExtracted', '1second50overlapSeg.csv')
OUTPUT_CSV = 'all_combinations_grid_search.csv'
OUTPUT_TXT = 'top3_models_report.txt'

# 1. LOADING AND PREPARATION
if not os.path.exists(INPUT_FILE):
    print(f"Error: File {INPUT_FILE} not found.")
    sys.exit()

df = pd.read_csv(INPUT_FILE)
df['sample_number'] = df['filename'].apply(extract_sample)

# Cleaning and Filtering
cols_to_drop = ['quat_mean_w', 'quat_mean_x', 'quat_mean_y', 'quat_mean_z', 'quat_var', 'quat_norm_mean', 
                'euler_roll', 'euler_pitch', 'euler_yaw', 'Volunteer', 'filename', 'win_idx', 
                'win_start_sec', 'activation_start_idx', 'activation_seg', 'duration', 'rep_idx']
df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])
df = df.dropna()

valid_terms = ['pu']
df = df[df['class'].apply(lambda x: any(t in str(x).lower().split('_') for t in valid_terms))]

X = df.drop(columns=['class', 'sample_number'], errors='ignore')
y = df['class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ==========================================
# 2. ROBUST GRID DEFINITION
# ==========================================
param_grids = {
    "Random Forest": {
        "model": RandomForestClassifier(random_state=42),
        "params": {
            "n_estimators": [50, 75, 100, 150],
            "max_depth": [None, 10, 20, 30],
            "min_samples_split": [2, 5, 10],
            "max_features": ['sqrt', 'log2']
        }
    },
    "SVM": {
        "model": SVC(random_state=42),
        "params": {
            "C": [0.1, 1, 10, 100],
            "gamma": [0.1, 0.01, 'scale'],
            "kernel": ['rbf', 'poly']
        }
    },
    "K-Neighbors": {
        "model": KNeighborsClassifier(),
        "params": {
            "n_neighbors": [3, 5, 7, 11],
            "weights": ['uniform', 'distance'],
            "metric": ['euclidean', 'manhattan']
        }
    },
    "Decision Tree": {
        "model": DecisionTreeClassifier(random_state=42),
        "params": {
            "criterion": ["gini", "entropy"],
            "max_depth": [None, 5, 10, 20],
            "min_samples_split": [2, 5, 10]
        }
    },
    "Logistic Regression": {
        "model": LogisticRegression(max_iter=2000, random_state=42),
        "params": {
            "C": [0.01, 0.1, 1, 10, 100],
            "solver": ['lbfgs', 'liblinear']
        }
    },
    "AdaBoost": {
        "model": AdaBoostClassifier(random_state=42),
        "params": {
            "n_estimators": [50, 100, 200, 500],
            "learning_rate": [0.01, 0.1, 1.0]
        }
    },
    "MLP Neural Net": {
        "model": MLPClassifier(max_iter=1000, random_state=42),
        "params": {
            "hidden_layer_sizes": [(50,), (100,), (100, 50)],
            "activation": ['tanh', 'relu'],
            "alpha": [0.0001, 0.05]
        }
    },
    "SGD Classifier": {
        "model": SGDClassifier(max_iter=2000, random_state=42),
        "params": {
            "loss": ['hinge', 'log_loss'],
            "penalty": ['l2', 'elasticnet'],
            "alpha": [0.0001, 0.01]
        }
    },
    "LDA": {
        "model": LinearDiscriminantAnalysis(),
        "params": {
            "solver": ['svd', 'lsqr']
        }
    },
    "Naive Bayes": {
        "model": GaussianNB(),
        "params": {
            "var_smoothing": np.logspace(0,-9, num=5)
        }
    }
}

# ==========================================
# 3. EXECUTION WITH INCREMENTAL SAVING
# ==========================================
accumulated_cv_list = []
total_models = len(param_grids)

print("\n" + "="*80)
print(f"{'STARTING GRID SEARCH WITH STEP-BY-STEP SAVING':^80}")
print("="*80)

# Initialize clean TXT
with open(OUTPUT_TXT, 'w') as f_txt:
    f_txt.write("PERFORMANCE REPORT: TOP 3 COMBINATIONS PER MODEL\n")
    f_txt.write("="*60 + "\n\n")

start_total = time.time()

for i, (name, config) in enumerate(param_grids.items(), 1):
    print(f"\n>>> [{i}/{total_models}] PROCESSING: {name}")
    start_model = time.time()
    
    try:
        # Grid Search Execution
        grid = GridSearchCV(
            config["model"], config["params"], cv=5, 
            n_jobs=N_JOBS, scoring='accuracy', verbose=2
        )
        grid.fit(X_train_scaled, y_train)

        # --- RESULTS PROCESSING ---
        
        # 1. Create DataFrame with all tests for this model
        current_cv_df = pd.DataFrame(grid.cv_results_)
        current_cv_df['Model_Name'] = name
        accumulated_cv_list.append(current_cv_df)

        # 2. Save/Update CSV (Incrementally)
        # Concatenate everything done so far to ensure column alignment
        csv_progress_df = pd.concat(accumulated_cv_list, ignore_index=True)
        csv_progress_df.to_csv(OUTPUT_CSV, index=False)
        
        # 3. Extract Top 3 to TXT (Appending to file)
        df_top3 = current_cv_df.sort_values(by='rank_test_score').head(3)
        duration = time.time() - start_model
        
        with open(OUTPUT_TXT, 'a') as f_txt:
            f_txt.write(f"CLASSIFIER: {name}\n")
            f_txt.write(f"Execution time: {duration:.2f}s\n")
            for rank, (_, row) in enumerate(df_top3.iterrows(), 1):
                f_txt.write(f"  {rank}th Place (Mean CV Accuracy: {row['mean_test_score']:.4f}):\n")
                f_txt.write(f"     - Parameters: {row['params']}\n")
            f_txt.write("-" * 40 + "\n\n")

        print(f"    [SAVED] Results for '{name}' added to files.")
        print(f"    [OK] Best CV Accuracy: {grid.best_score_:.4f}")

    except Exception as e:
        print(f"    [ERROR] Failure in {name}: {e}")
        with open(OUTPUT_TXT, 'a') as f_txt:
            f_txt.write(f"CLASSIFIER: {name} - PROCESSING ERROR\n\n")

total_time_min = (time.time() - start_total) / 60
print("\n" + "="*80)
print(f"PROCESS FINISHED IN {total_time_min:.2f} MINUTES")
print(f"CSV updated with all combinations: {OUTPUT_CSV}")
print(f"TXT report updated with Top 3: {OUTPUT_TXT}")
print("="*80)