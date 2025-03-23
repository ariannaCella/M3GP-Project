import pandas
from m3gp.M3GP import M3GP
from sys import argv
from Arguments import *
import logging
import os

from sklearn.model_selection import train_test_split
from concurrent.futures import ProcessPoolExecutor

import numpy as np

import warnings

warnings.filterwarnings("ignore", category=FutureWarning,
                        message="From version 0.21, test_size will always complement",
                        module="sklearn")


def openAndSplitDatasets(which, seed):
    if VERBOSE:
        print("> Opening: ", which)

    # Open dataset
    ds = pandas.read_csv(DATASETS_DIR + which)

    # Read header
    class_header = ds.columns[-1]

    return train_test_split(ds.drop(columns=[class_header]), ds[class_header],
                            train_size=TRAIN_FRACTION, random_state=seed,
                            stratify=ds[class_header])

def run(r, dataset):
    if VERBOSE:
        print("> Starting run:")
        print("  > ID:", r)
        print("  > Dataset: " + dataset)
        print()

    Tr_X, Te_X, Tr_Y, Te_Y = openAndSplitDatasets(dataset, r)

    # Train a model
    m3gp = M3GP(OPERATORS, MAX_DEPTH, POPULATION_SIZE, MAX_GENERATION, TOURNAMENT_SIZE,
                ELITISM_SIZE, LIMIT_DEPTH, DIM_MIN, DIM_MAX, THREADS, r, VERBOSE, MODEL_NAME, FITNESS_TYPE)
    m3gp.fit(Tr_X, Tr_Y, Te_X, Te_Y)

    # Obtain training results
    accuracy  = m3gp.getAccuracyOverTime()
    waf       = m3gp.getWaFOverTime()
    kappa     = m3gp.getKappaOverTime()
    mse       = m3gp.getMSEOverTime()
    sizes     = m3gp.getSizesOverTime()
    model_str = str(m3gp.getBestIndividual())
    times     = m3gp.getGenerationTimes()

    tr_acc     = accuracy[0]
    te_acc     = accuracy[1]
    tr_waf     = waf[0]
    te_waf     = waf[1]
    tr_kappa   = kappa[0]
    te_kappa   = kappa[1]
    tr_mse     = mse[0]
    te_mse     = mse[1]
    size       = sizes[0]
    dimensions = sizes[1]

    if VERBOSE:
        print("> Ending run:")
        print("  > ID:", r)
        print("  > Dataset:", dataset)
        print("  > Final model:", model_str)
        print("  > Training accuracy:", tr_acc[-1])
        print("  > Test accuracy:", te_acc[-1])
        print()

    return (tr_acc, te_acc,
            tr_waf, te_waf,
            tr_kappa, te_kappa,
            tr_mse, te_mse,
            size, dimensions,
            times,
            model_str)

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
            # Utilizza ProcessPoolExecutor con 10 workers per eseguire 10 run
            with ProcessPoolExecutor(max_workers=15) as executor:
                # Crea 10 futuri, uno per ogni run
                futures = [executor.submit(run, r + RANDOM_STATE, dataset) for r in range(RUNS)]
                # Raccogli i risultati man mano che sono completati
                for future in futures:
                    results.append(future.result())

            # Ora procedi con la scrittura dei risultati sul file come già facevi
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
                        file.write(",".join([str(val) for val in results[i][ai]]))
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
