import pandas as pd
import numpy as np
import os
import re

def clean_expression(expr):
    if not isinstance(expr, str):
        return expr
    # Rimuovi spazi all'inizio, alla fine e all'interno
    return expr.strip().replace(" ", "")

def evaluate_expression(expr, row):
    if not isinstance(expr, str) or expr.strip().lower() == "nan":
        return np.nan
    try:
        cleaned_expr = clean_expression(expr)
        # Valuta l'espressione usando il dizionario della riga (es. {"X0":66, "X1":56, ...})
        return eval(cleaned_expr, {}, row.to_dict())
    except ZeroDivisionError:
        return np.nan
    except Exception as e:
        print(f"Errore nella valutazione dell'espressione '{expr}' per la riga {row.name}: {e}")
        return np.nan

# Percorsi dei file
dataset_file = r"..\Python-M3GP\datasets\nose.csv"
#dataset_file = r"..\Python-M3GP\datasets\mcd3.csv"
#dataset_file = r"..\Python-M3GP\datasets\brazil2.csv"

#hyperfeatures_file = r"..\Python-M3GP\results\m3gpWAF_mcd3.csv"
#hyperfeatures_file = r"..\Python-M3GP\results\m3gpWAF_brazil2.csv"
hyperfeatures_file = r"..\Python-M3GP\results\m3gpWAF_nose.csv"

# Cartella di output per i file CSV generati
output_folder = r"tabelle_hyperfeatures\nose_WAF_hyper"
#output_folder = r"tabelle_hyperfeatures\mcd3_WAF_hyper"
#output_folder = r"tabelle_hyperfeatures\brazil2_WAF_hyper"
os.makedirs(output_folder, exist_ok=True)

# Leggi il dataset originale
df_original = pd.read_csv(dataset_file)

# Leggi il file delle hyperfeatures (senza header)
hf_df = pd.read_csv(hyperfeatures_file, header=None)

# Filtra solo le righe che iniziano con "Final_Model" nella prima colonna
hf_df = hf_df[hf_df[0].astype(str).str.startswith("Final_Model")]

# Per ogni run (ogni riga filtrata) creiamo un file CSV con le hyperfeatures calcolate
for idx, hf_row in hf_df.iterrows():
    # Recupera l'identificativo della run dalla seconda colonna (ad es. 9)
    run_id = hf_row[1]
    
    # Seleziona le colonne a partire dalla terza e filtra le espressioni valide (non NaN e non "nan")
    raw_expressions = hf_row.iloc[2:]
    valid_expressions = [expr for expr in raw_expressions if isinstance(expr, str) and expr.strip().lower() != "nan"]
    
    # Crea una copia del dataset originale
    df_run = df_original.copy()
    
    # Per ogni espressione valida, valuta il risultato per ogni riga del dataset
    for i, expr in enumerate(valid_expressions, start=1):
        col_name = f"Hyperfeature_{i}"
        df_run[col_name] = df_run.apply(lambda r: evaluate_expression(expr, r), axis=1)
    
    # Salva il dataset arricchito in un nuovo file CSV
    output_file = os.path.join(output_folder, f"noseWAF_run_{run_id}.csv")
    #output_file = os.path.join(output_folder, f"mcd3WAF_run_{run_id}.csv")
    #output_file = os.path.join(output_folder, f"brazil2WAF_run_{run_id}.csv")
    df_run.to_csv(output_file, index=False)
    print(f"File salvato: {output_file}")
