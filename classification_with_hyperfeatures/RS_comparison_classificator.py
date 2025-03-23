import os
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import ttest_1samp

# Percorso della cartella contenente i dataset
dataset = "mcd3"
#dataset = "brazil2"

if (dataset=="mcd3"):
    data_folder = r'tabelle_hyperfeatures\mcd3_WAF_hyper'
    df = r'..\Python-M3GP\datasets\mcd3.csv'
elif (dataset=="brazil2"):
    data_folder = r'tabelle_hyperfeatures\brazil2_WAF_hyper'
    df = r'..\Python-M3GP\datasets\brazil2.csv'
else: 
    print("dataset non supportato")
    exit()

num_runs = 30  # dataset da 0 a 29

# Liste per salvare le metriche per ogni run
accuracy_listRF = []
accuracy_listDT = []
accuracy_listXGB =[]

accuracyRFo = []
accuracyDTo=[]
accuracyXGBo=[]

def process_dataset(dataset_path, modelName, i=0):
    try:
        df = pd.read_csv(dataset_path)
    except Exception as e:
        print(f"Errore nel caricamento di {dataset_path}: {e}")
        return None

    if 'Class' not in df.columns:
        print(f"Il dataset {os.path.basename(dataset_path)} non contiene la colonna 'Class'.")
        return None

    # Separazione delle features e del target
    X = df.drop('Class', axis=1)
    y = df['Class']
    
    # Se il modello è xgboost, converti le colonne di tipo object in numerico
    if modelName == "xgboost":
        for col in X.select_dtypes(include=['object']).columns:
            X[col] = X[col].astype('category').cat.codes

    # Applica il Label Encoding per il target, se necessario
    if modelName == "xgboost":
        
        le = LabelEncoder()
        y = le.fit_transform(y)
    
    # Divisione in training e test set (70%-30%)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42+i)

    # Crea il modello
    if modelName == "xgboost":
        model = xgb.XGBClassifier(random_state=42, eval_metric='mlogloss')
    elif modelName == "RF":
        model = RandomForestClassifier(n_estimators=50, random_state=42)
    elif modelName == "DecisionTree":
        model = DecisionTreeClassifier(random_state=42)

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    return classification_report(y_test, y_pred, output_dict=True)


# Ciclo per ogni dataset
for i in range(num_runs):

    if (dataset == "brazil2"):
        dataset_name = f"brazil2WAF_run_{i}.csv"
    elif (dataset == "mcd3"):
        dataset_name = f"mcd3WAF_run_{i}.csv"

    dataset_path = os.path.join(data_folder, dataset_name)
    reportRF = process_dataset(dataset_path, "RF")
    reportDT = process_dataset(dataset_path, "DecisionTree")
    reportXGB = process_dataset(dataset_path, "xgboost")
    
    # Salva le metriche di interesse
    accuracy_listRF.append(reportRF['accuracy'])
    accuracy_listDT.append(reportDT['accuracy'])
    accuracy_listXGB.append(reportXGB['accuracy'])

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
    # Salva le metriche di interesse
    accuracyRFo.append(reportRFo['accuracy'])
    accuracyDTo.append(reportDTo['accuracy'])
    accuracyXGBo.append(reportXGBo['accuracy'])

avg_accuracyRFo= np.mean(accuracyRFo)
avg_accuracyDTo= np.mean(accuracyDTo)
avg_accuracyXGBo= np.mean(accuracyXGBo)

print("\nOriginal media accuracy:")
print(f"Accuracy RF: {avg_accuracyRFo:.4f}")
print(f"Accuracy DT: {avg_accuracyDTo:.4f}")
print(f"Accuracy XGB: {avg_accuracyXGBo:.4f}")
# Organizza le liste in un dizionario per facilitare la creazione dei boxplot
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


