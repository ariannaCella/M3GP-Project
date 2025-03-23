import os
from re import VERBOSE
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold
import warnings

warnings.filterwarnings("ignore", category=FutureWarning,
                        message="From version 0.21, test_size will always complement",
                        module="sklearn")

DATASETS_DIR = "tabelle_hyperfeatures/nose_WAF_hyper"  
K_FOLDS = 3
df = r'..\Python-M3GP\datasets\nose.csv'
num_runs = 30
CSV_OUTPUT = "risultati_metriche_nose.csv"

# Liste per salvare le metriche per ogni run
accuracy_listRF, accuracy_listDT, accuracy_listXGB = [], [], []
accuracyRFo, accuracyDTo, accuracyXGBo = [], [], []
results = []

def openAndSplitDatasetsKFold(which, seed, n_splits=K_FOLDS):
    """
    Legge il dataset utilizzando la prima riga come intestazione e lo suddivide in n fold
    utilizzando GroupKFold, in modo che i gruppi di 3 righe non vengano separati.
    """
    if VERBOSE:
        print("> Opening:", which)

    ds = pd.read_csv(which, header=0)
    column_names = ds.columns.tolist()
    groups = np.arange(len(ds)) // 3
    
    # Crea un generatore di numeri casuali con il seed passato
    if (seed !=0):
        # Ottieni gli id unici dei gruppi
        unique_groups = np.unique(groups)
        rng = np.random.RandomState(seed)
        rng.shuffle(unique_groups)
        new_order = np.concatenate([np.where(groups == grp)[0] for grp in unique_groups])
        ds = ds.iloc[new_order].reset_index(drop=True)
    
    class_header = ds.columns[4]
    X = ds.drop(columns=[ds.columns[4], ds.columns[5], ds.columns[6], ds.columns[7]])
    y = ds[class_header]
    
    # Suddivide in n_splits mantenendo interi i gruppi
    gkf = GroupKFold(n_splits=n_splits)
    splits = list(gkf.split(X, y, groups=groups))

    return X, y, splits, column_names


def process_dataset(dataset_path, modelName, i=0):
    # dati e fold per cross validation
    X, y, splits, column_names = openAndSplitDatasetsKFold(dataset_path, i, n_splits=K_FOLDS)
    accuracy_listFold = []

    if modelName == "xgboost":
        for col in X.select_dtypes(include=['object']).columns:
            X[col] = X[col].astype('category').cat.codes
        le = LabelEncoder()
        y = pd.Series(le.fit_transform(y))

    model_dict = {
        "xgboost": xgb.XGBClassifier(random_state=42, eval_metric='mlogloss'),
        "RF": RandomForestClassifier(n_estimators=50, random_state=42),
        "DecisionTree": DecisionTreeClassifier(random_state=42)
    }

    model = model_dict[modelName]

    # Esegui il training su ogni fold
    for fold, (train_idx, test_idx) in enumerate(splits):
        if VERBOSE:
            print(f"  > Fold {fold+1}/{K_FOLDS}")
        Tr_X, Te_X = X.iloc[train_idx], X.iloc[test_idx]
        Tr_Y, Te_Y = y.iloc[train_idx], y.iloc[test_idx]

        model.fit(Tr_X, Tr_Y)
        y_pred = model.predict(Te_X)
        report = classification_report(Te_Y, y_pred, output_dict=True)
        accuracy_listFold.append(report['accuracy'])
    
    print("Avg accuracy run: ", np.mean(accuracy_listFold))
    return np.mean(accuracy_listFold)

# Funzione per aggiungere risultati
def add_model_results(model_name, avg_acc_orig, avg_acc_m3gp, acc_orig_list, acc_m3gp_list):
    results.append([model_name, f"Media accuracy dataset originale: {avg_acc_orig:.4f}", f"Media accuracy dataset M3GP: {avg_acc_m3gp:.4f}"])
    for i in range(num_runs):
        results.append([
            f"Iterazione {i + 1}",
            f"{acc_orig_list[i]:.4f}",
            f"{acc_m3gp_list[i]:.4f}"
        ])


# Ciclo per ogni dataset
for i in range(num_runs):
    dataset_name = f"noseWAF_run_{i}.csv"
    dataset_path = os.path.join(DATASETS_DIR, dataset_name)
    reportRF = process_dataset(dataset_path, "RF", i)
    reportDT = process_dataset(dataset_path, "DecisionTree", i)
    reportXGB = process_dataset(dataset_path, "xgboost", i)
    
    # Salva le metriche
    accuracy_listRF.append(reportRF)
    accuracy_listDT.append(reportDT)
    accuracy_listXGB.append(reportXGB)

avg_accuracyRF= np.mean(accuracy_listRF)
avg_accuracyDT= np.mean(accuracy_listDT)
avg_accuracyXGB= np.mean(accuracy_listXGB)

print("\nMedia accuracy:")
print(f"Accuracy RF: {avg_accuracyRF:.4f}")
print(f"Accuracy DT: {avg_accuracyDT:.4f}")
print(f"Accuracy XGB: {avg_accuracyXGB:.4f}")


#calcolo originali
for i in range(num_runs):
    reportRFo = process_dataset(df, "RF", i)
    reportDTo = process_dataset(df, "DecisionTree",i)
    reportXGBo = process_dataset(df, "xgboost",i)
    # Salva le metriche 
    accuracyRFo.append(reportRFo)
    accuracyDTo.append(reportDTo)
    accuracyXGBo.append(reportXGBo)

avg_accuracyRFo= np.mean(accuracyRFo)
avg_accuracyDTo= np.mean(accuracyDTo)
avg_accuracyXGBo= np.mean(accuracyXGBo)

print("\nOriginal media accuracy:")
print(f"Accuracy RF: {avg_accuracyRFo:.4f}")
print(f"Accuracy DT: {avg_accuracyDTo:.4f}")
print(f"Accuracy XGB: {avg_accuracyXGBo:.4f}")

# Preparazione risultati csv
add_model_results("RANDOM FOREST", avg_accuracyRFo, avg_accuracyRF, accuracyRFo, accuracy_listRF)
add_model_results("DECISION TREE", avg_accuracyDTo, avg_accuracyDT, accuracyDTo, accuracy_listDT)
add_model_results("XGBOOST", avg_accuracyXGBo, avg_accuracyXGB, accuracyXGBo, accuracy_listXGB)

# Salvataggio risultati csv
df_results = pd.DataFrame(results, columns=['Descrizione', 'Accuracy dataset originale', 'Accuracy dataset M3GP'])
df_results.to_csv(CSV_OUTPUT, index=False)
print(f"Risultati salvati correttamente in {CSV_OUTPUT}")

metrics = {
    'AccuracyRF original': accuracyRFo,
    'AccuracyRF': accuracy_listRF,
    'AccuracyDT original': accuracyDTo,
    'AccuracyDT': accuracy_listDT,
    'AccuracyXGB original': accuracyXGBo,
    'AccuracyXGB': accuracy_listXGB
}

# Crea i boxplot
plt.figure(figsize=(12, 6))
plt.boxplot(metrics.values(), labels=metrics.keys(), patch_artist=True)
plt.title("Distribuzione delle metriche sui dataset")
plt.ylabel("Valori delle metriche")
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

