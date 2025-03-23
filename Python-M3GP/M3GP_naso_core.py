import pandas as pd
import numpy as np
from m3gp.M3GP import M3GP
from sys import argv
from Arguments import *
import logging
import os
from sklearn.model_selection import GroupKFold
from concurrent.futures import ProcessPoolExecutor
import warnings

warnings.filterwarnings("ignore", category=FutureWarning,
                        message="From version 0.21, test_size will always complement",
                        module="sklearn")

# Costanti globali
DATASETS_DIR = "Python-M3GP/datasets/"  # Assicurati di impostare il percorso corretto
K_FOLDS = 3  # Numero di fold per la cross validation

def is_iterable(obj):
        # Considera iterabile se possiede __iter__ ma non è una stringa
        try:
            iter(obj)
            return not isinstance(obj, str)
        except TypeError:
            return False


def openAndSplitDatasetsKFold(which, seed, n_splits=K_FOLDS):
    if VERBOSE:
        print("> Opening:", which)

    ds = pd.read_csv(DATASETS_DIR+which, header=0)
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

def run(r, dataset):
    """
    Per il run 'r' su un determinato dataset, esegue la K-Fold Cross Validation.
    Per ogni fold viene addestrato il modello M3GP, ne vengono estratte le metriche e
    successivamente vengono mediate le prestazioni su tutti i fold.
    """
    if VERBOSE:
        print("> Starting run:")
        print("  > ID:", r)
        print("  > Dataset:", dataset)
        print()

    # Ottieni i dati e i fold di cross validation
    X, y, splits, column_names = openAndSplitDatasetsKFold(dataset, r, n_splits=K_FOLDS)
    
    # Liste per salvare le metriche di ciascun fold
    all_tr_acc = []
    all_te_acc = []
    all_tr_waf = []
    all_te_waf = []
    all_tr_kappa = []
    all_te_kappa = []
    all_tr_mse = []
    all_te_mse = []
    all_sizes = []
    all_dimensions = []
    all_times = []
    best_models = []  # per salvare il modello migliore per ogni fold

    # Esegui il training su ogni fold
    for fold, (train_idx, test_idx) in enumerate(splits):
        if VERBOSE:
            print(f"  > Fold {fold+1}/{K_FOLDS}")
        Tr_X, Te_X = X.iloc[train_idx], X.iloc[test_idx]
        Tr_Y, Te_Y = y.iloc[train_idx], y.iloc[test_idx]

        # Inizializza ed addestra il modello
        m3gp = M3GP(OPERATORS, MAX_DEPTH, POPULATION_SIZE, MAX_GENERATION, TOURNAMENT_SIZE,
                    ELITISM_SIZE, LIMIT_DEPTH, DIM_MIN, DIM_MAX, THREADS, r, VERBOSE, MODEL_NAME, FITNESS_TYPE)
        m3gp.fit(Tr_X, Tr_Y, Te_X, Te_Y)

        # Estrai le metriche per questo fold
        accuracy  = m3gp.getAccuracyOverTime()
        waf       = m3gp.getWaFOverTime()
        kappa     = m3gp.getKappaOverTime()
        mse       = m3gp.getMSEOverTime()
        sizes     = m3gp.getSizesOverTime()
        model_str = str(m3gp.getBestIndividual())
        times     = m3gp.getGenerationTimes()

        all_tr_acc.append(accuracy[0])
        all_te_acc.append(accuracy[1])
        all_tr_waf.append(waf[0])
        all_te_waf.append(waf[1])
        all_tr_kappa.append(kappa[0])
        all_te_kappa.append(kappa[1])
        all_tr_mse.append(mse[0])
        all_te_mse.append(mse[1])
        all_sizes.append(sizes[0])
        all_dimensions.append(sizes[1])
        all_times.append(times)
        best_models.append(model_str)

    # Aggrega i risultati mediando le metriche su tutti i fold
    avg_tr_acc = np.mean(all_tr_acc, axis=0).tolist()
    avg_te_acc = np.mean(all_te_acc, axis=0).tolist()
    avg_tr_waf = np.mean(all_tr_waf, axis=0).tolist()
    avg_te_waf = np.mean(all_te_waf, axis=0).tolist()
    avg_tr_kappa = np.mean(all_tr_kappa, axis=0).tolist()
    avg_te_kappa = np.mean(all_te_kappa, axis=0).tolist()
    avg_tr_mse = np.mean(all_tr_mse, axis=0).tolist()
    avg_te_mse = np.mean(all_te_mse, axis=0).tolist()
    avg_size = np.mean(all_sizes)
    avg_dimensions = np.mean(all_dimensions)
    # Per il tempo, se ogni fold restituisce una lista per ogni generazione:
    avg_times = np.mean(all_times, axis=0).tolist() if isinstance(all_times[0], (list, np.ndarray)) else np.mean(all_times)
    # Seleziona il modello migliore in base alla migliore test accuracy nell'ultima generazione
    best_model = best_models[np.argmax([acc[-1] for acc in all_te_acc])]
    final_model_str = f"{best_model} | HyperFeatures: {', '.join(column_names)}"

    if VERBOSE:
        print("> Ending run:")
        print("  > ID:", r)
        print("  > Dataset:", dataset)
        print("  > Best model:", final_model_str)
        print("  > Average Training accuracy:", avg_tr_acc[-1])
        print("  > Average Test accuracy:", avg_te_acc[-1])
        print()

    return (avg_tr_acc, avg_te_acc,
            avg_tr_waf, avg_te_waf,
            avg_tr_kappa, avg_te_kappa,
            avg_tr_mse, avg_te_mse,
            avg_size, avg_dimensions,
            avg_times,
            best_model)

def callm3gp():
    try:
        os.makedirs(OUTPUT_DIR)
    except Exception as e:
        if VERBOSE:
            print("Output directory exists or cannot be created:", e)

    for dataset in DATASETS:
        outputFilename = os.path.join(OUTPUT_DIR, "m3gpWAF_" + dataset)
        if not os.path.exists(outputFilename):
            results = []
            # Utilizza ProcessPoolExecutor con 15 workers per eseguire 30 run in parallelo
            with ProcessPoolExecutor(max_workers=15) as executor:
                futures = [executor.submit(run, r + RANDOM_STATE, dataset) for r in range(RUNS)]
                for future in futures:
                    results.append(future.result())

            
            # Scrittura dei risultati sul file
            with open(outputFilename, "w") as file:
                file.write("Attribute,Run,")
                for i in range(MAX_GENERATION):
                    file.write(str(i) + ",")
                file.write("\n")

                attributes = ["Training-Accuracy", "Test-Accuracy",
                              "Training-WaF", "Test-WaF",
                              "Training-Kappa", "Test-Kappa",
                              "Training-MSE", "Test-MSE",
                              "Size", "Dimensions",
                              "Time",
                              "Final_Model"]

                for ai in range(len(attributes) - 1):
                    for i in range(len(results)):
                        file.write("\n" + attributes[ai] + "," + str(i + RANDOM_STATE) + ",")

                        res_val = results[i][ai]
                        # Se l'elemento è iterabile, scrivi un join dei suoi valori, altrimenti scrivi direttamente la stringa
                        if is_iterable(res_val):
                            file.write(",".join([str(val) for val in res_val]))
                        else:
                            file.write(str(res_val))
                    file.write("\n")

                for i in range(len(results)):
                    file.write("\n" + attributes[-1] + "," + str(i + RANDOM_STATE) + ",")
                    file.write(results[i][-1])
                file.write("\n")

                file.write("\n\nParameters")
                file.write("\nOperators," + str(OPERATORS))
                file.write("\nMax Initial Depth," + str(MAX_DEPTH))
                file.write("\nPopulation Size," + str(POPULATION_SIZE))
                file.write("\nMax Generation," + str(MAX_GENERATION))
                file.write("\nTournament Size," + str(TOURNAMENT_SIZE))
                file.write("\nElitism Size," + str(ELITISM_SIZE))
                file.write("\nDepth Limit," + str(LIMIT_DEPTH))
                file.write("\nMinimum Dimensions," + str(DIM_MIN))
                file.write("\nMaximum Dimensions," + str(DIM_MAX))
                file.write("\nWrapped Model," + MODEL_NAME)
                file.write("\nFitness Type," + FITNESS_TYPE)
                file.write("\nThreads," + str(THREADS))
                file.write("\nRandom State," + str(list(range(RUNS))))
                file.write("\nDataset," + dataset)
        else:
            print("Filename: " + outputFilename + " already exists.")

if __name__ == '__main__':
    callm3gp()
