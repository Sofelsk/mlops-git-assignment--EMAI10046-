# Cell 1: Imports
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, roc_auc_score, accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
#1. Load dataset (GitHub CSV)
#Use a GitHub CSV such as penguins_size.csv from a Palmer Penguins repo. Example:



# Cell 2: Load data from GitHub or local
# Option A: direct GitHub raw link (replace with your chosen repo/raw link)
url = "url = "https://raw.githubusercontent.com/allisonhorst/palmerpenguins/master/inst/extdata/penguins.csv"

df = pd.read_csv(url)
df.head()


# Alternative if file is local:
# df = pd.read_csv("penguins_size.csv")
#2. Dataset understanding and cleaning
#python
# Cell 3: Understand data
print("Shape:", df.shape)
print(df.info())
print(df.describe(include="all"))
print("Missing values:\n", df.isna().sum())
#python
# Cell 4: Basic cleaning - drop rows with NA
df_clean = df.dropna().copy()

# For typical GitHub penguin CSVs:
# species (target), culmen_length_mm/bill_length_mm, culmen_depth_mm/bill_depth_mm,
# flipper_length_mm, body_mass_g, island, sex etc. [web:8][web:3]

# Try common column names safely
possible_length = [c for c in df_clean.columns if "culmen_length" in c or "bill_length" in c][0]
possible_depth = [c for c in df_clean.columns if "culmen_depth" in c or "bill_depth" in c][0]

length_col = possible_length
depth_col = possible_depth

print("Using length column:", length_col)
print("Using depth column:", depth_col)
#python
# Cell 5: Define features and target
target_col = "species"

numeric_features = [
    length_col,         # culmen/bill length
    depth_col,          # culmen/bill depth
    "flipper_length_mm",
    "body_mass_g"
]

X_all = df_clean[numeric_features]
y = df_clean[target_col]

# Encode target labels
label_enc = LabelEncoder()
y_enc = label_enc.fit_transform(y)

print("Classes:", label_enc.classes_)
X_all.head()
#3. Helper function for experiments

# Cell 6: Generic experiment runner
def run_experiment(
    model,
    X,
    y,
    test_size=0.2,
    random_state=42,
    scale=False,
    description=""
):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    scaler = None
    if scale:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)
        auc = roc_auc_score(y_test, y_proba, multi_class="ovr")
    else:
        auc = np.nan

    precision = precision_score(y_test, y_pred, average="macro")
    accuracy = accuracy_score(y_test, y_pred)

    print("=== Experiment:", description, "===")
    print("Precision (macro):", precision)
    print("Accuracy:", accuracy)
    print("AUC (ovr):", auc)
    print()

    return {
        "description": description,
        "precision": precision,
        "accuracy": accuracy,
        "auc": auc,
        "model": model,
        "scaler": scaler,
    }
#4. Experiments: Decision Tree (EXP‑01, EXP‑02)

# Cell 7: EXP-01 – Decision Tree, default, all numeric, no scaling
dt_default = DecisionTreeClassifier(random_state=0)

result_exp01 = run_experiment(
    model=dt_default,
    X=X_all,
    y=y_enc,
    test_size=0.2,
    random_state=0,
    scale=False,
    description="EXP-01: Decision Tree, default, all numeric, no scaling, 80/20"
)

# Cell 8: EXP-02 – Decision Tree, tuned hyperparameters
dt_tuned = DecisionTreeClassifier(
    max_depth=3,
    min_samples_split=4,
    random_state=1
)

result_exp02 = run_experiment(
    model=dt_tuned,
    X=X_all,
    y=y_enc,
    test_size=0.2,
    random_state=1,
    scale=False,
    description="EXP-02: Decision Tree, max_depth=3, min_samples_split=4, all numeric, no scaling, 80/20"
)
#5. Experiments: kNN (EXP‑03, EXP‑04)

# Cell 9: Define subset features for feature selection experiment
subset_features = [length_col, depth_col]  # only bill/culmen length + depth
X_subset = df_clean[subset_features]

# Cell 10: EXP-03 – kNN, k=5, all numeric, scaled
knn_5 = KNeighborsClassifier(n_neighbors=5)

result_exp03 = run_experiment(
    model=knn_5,
    X=X_all,
    y=y_enc,
    test_size=0.2,
    random_state=2,
    scale=True,
    description="EXP-03: kNN (k=5), all numeric features, scaled, 80/20"
)

# Cell 11: EXP-04 – kNN, k=7, subset (bill/culmen only), scaled
knn_7 = KNeighborsClassifier(n_neighbors=7)

result_exp04 = run_experiment(
    model=knn_7,
    X=X_subset,
    y=y_enc,
    test_size=0.2,
    random_state=3,
    scale=True,
    description="EXP-04: kNN (k=7), bill/culmen length + depth only, scaled, 80/20"
)
#6. Collect results for manual table

# Cell 12: Build results table for your Word document
results = pd.DataFrame([
    {
        "Experiment ID": "EXP-01",
        "Model Type": "Decision Tree",
        "Hyperparameters": "default, random_state=0",
        "Preprocessing": "Drop NA, no scaling",
        "Feature Selection": "All numeric features",
        "Train/Test Split": "80/20, stratified, random_state=0",
        "Precision": result_exp01["precision"],
        "Accuracy": result_exp01["accuracy"],
        "AUC": result_exp01["auc"],
    },
    {
        "Experiment ID": "EXP-02",
        "Model Type": "Decision Tree",
        "Hyperparameters": "max_depth=3, min_samples_split=4, random_state=1",
        "Preprocessing": "Drop NA, no scaling",
        "Feature Selection": "All numeric features",
        "Train/Test Split": "80/20, stratified, random_state=1",
        "Precision": result_exp02["precision"],
        "Accuracy": result_exp02["accuracy"],
        "AUC": result_exp02["auc"],
    },
    {
        "Experiment ID": "EXP-03",
        "Model Type": "kNN",
        "Hyperparameters": "n_neighbors=5",
        "Preprocessing": "Drop NA, StandardScaler on numeric",
        "Feature Selection": "All numeric features",
        "Train/Test Split": "80/20, stratified, random_state=2",
        "Precision": result_exp03["precision"],
        "Accuracy": result_exp03["accuracy"],
        "AUC": result_exp03["auc"],
    },
    {
        "Experiment ID": "EXP-04",
        "Model Type": "kNN",
        "Hyperparameters": "n_neighbors=7",
        "Preprocessing": "Drop NA, StandardScaler on numeric",
        "Feature Selection": "bill/culmen length + depth only",
        "Train/Test Split": "80/20, stratified, random_state=3",
        "Precision": result_exp04["precision"],
        "Accuracy": result_exp04["accuracy"],
        "AUC": result_exp04["auc"],
    },
])

#results
#You can copy the numbers from results into your Word tracking table.
#7. Quick helpers for your analysis text

# Cell 13: Identify best by accuracy and AUC
best_acc = results.sort_values("Accuracy", ascending=False).iloc[0]
best_auc = results.sort_values("AUC", ascending=False).iloc[0]

print("Best Accuracy experiment:\n", best_acc, "\n")
print("Best AUC experiment:\n", best_auc)


