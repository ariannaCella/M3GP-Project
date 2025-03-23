import pandas as pd
import re
from collections import defaultdict
import csv

#model_name = "brazil2"
#model_name = "mcd3"
model_name = "nose"

# Caricare il file CSV
if (model_name == "brazil2"):
    file_path = r"results\m3gpWAF_brazil2.csv"
elif (model_name == "mcd3"):
    file_path = r"results\m3gpWAF_mcd3.csv"
elif (model_name == "nose"):
    file_path = r"results\m3gpWAF_nose.csv"
else:
    print("Modello non riconosciuto")
    exit()

df = pd.read_csv(file_path)

############ TRAINING ACCURACY ##############

# Filtrare solo le righe relative a Training-Accuracy
df_training = df[df["Attribute"].str.startswith("Training-Accuracy", na=False)]

# Selezionare solo le colonne numeriche (da 0 a 49)
df_training_values = df_training.loc[:, "0":"49"].astype(float)

# Calcolare la media per ogni colonna
training_accuracy_average = df_training_values.mean().tolist()

# Estrarre la media dell'ultima colonna (49)
training_accuracy_average_final = training_accuracy_average[-1]

# Stampare i risultati
#print("Media Training Accuracy per colonna:", training_accuracy_average)
print("Media Training Accuracy ultima colonna (49):", training_accuracy_average_final)


########## TEST ACCURACY ###############

# Filtrare solo le righe relative a Test-Accuracy
df_test = df[df["Attribute"].str.startswith("Test-Accuracy", na=False)]

# Selezionare solo le colonne numeriche (da 0 a 49)
df_test_values = df_test.loc[:, "0":"49"].astype(float)

# Calcolare la media per ogni colonna
test_accuracy_average = df_test_values.mean().tolist()

# Estrarre la media dell'ultima colonna (49)
test_accuracy_average_final = test_accuracy_average[-1]

# Stampare i risultati
#print("Media Test Accuracy per colonna:", test_accuracy_average)
print("\nMedia Test Accuracy ultima colonna (49):", test_accuracy_average_final)


if (model_name != "nose"):
    #### DIMENSION #########

    # Filtrare solo le righe relative a "Dimensions"
    df_dimension = df[df["Attribute"].str.startswith("Dimensions", na=False)]

    # Selezionare solo le colonne numeriche (da 0 a 49)
    df_dimension_values = df_dimension.loc[:, "0":"49"].astype(float)
    dimension_average = df_dimension_values.mean().tolist()

    # Estrarre la media dell'ultima colonna (49)
    dimension_average_final = dimension_average[-1]

    dimension_min_final = df_dimension_values.iloc[:, -1].min()
    dimension_max_final = df_dimension_values.iloc[:, -1].max()

    # Stampare i risultati
    #print("Media Dimension per colonna:", dimension_average)
    print("\nMedia Dimension ultima colonna (49):", dimension_average_final)
    print("Valore minimo dell'ultima colonna (49) per Dimension:", dimension_min_final)
    print("Valore massimo dell'ultima colonna (49) per Dimension:", dimension_max_final)


    #### SIZE ######
    # Filtrare solo le righe relative a "Size"
    df_size = df[df["Attribute"].str.startswith("Size", na=False)]

    # Selezionare solo le colonne numeriche (da 0 a 49)
    df_size_values = df_size.loc[:, "0":"49"].astype(float)

    # Divisione elemento per elemento per tutte le colonne di Size e Dimension
    df_avg_size_values = df_size_values.div(df_dimension_values.values)

    # Calcolare la media per ogni colonna risultante
    size_average = df_avg_size_values.mean().tolist()

    size_min_final = df_avg_size_values.iloc[:, -1].min()
    size_max_final = df_avg_size_values.iloc[:, -1].max()


    # Stampare i risultati
    print("\nMedia Size ultima colonna:", size_average[-1])
    print("Valore minimo dell'ultima colonna (49) per Size:", size_min_final)
    print("Valore massimo dell'ultima colonna (49) per Size:", size_max_final)


    ####################################################################
    # 
    #   CALCOLO GRAFICO BANDE
    #
    ###################################################################


    def analizza_csv(file_path):
        risultati = []

        with open(file_path, mode='r', newline='') as file:
            reader = csv.reader(file)

            for riga in reader:
                # Controlla se la riga inizia con "Final_Model"
                if riga and riga[0] == 'Final_Model':
                    # Unisci tutti gli elementi successivi ai primi due in una stringa unica
                    dati_grezzi = ','.join(riga[2:])
                    # Dividi gli elementi usando la virgola come separatore
                    dati_utili = [elemento.strip() for elemento in dati_grezzi.split(',')]
                    for d in dati_utili:
                        # Aggiungi i dati utili all'array dei risultati
                        risultati.append(d)

        return risultati

    # Funzione per calcolare le percentuali di presenza dei termini B1-B7 per Brazil2 e X0-X5 per MCD3
    def calcola_percentuali(risultati):
        percentuali = []
        if(model_name == "brazil2"):
            termini = [f'B{i}' for i in range(1, 8)]
        elif (model_name == "mcd3"):
            termini = [f'X{i}' for i in range(0, 6)]
        

        for elemento in risultati:
            conteggi = {termine: len(re.findall(termine, elemento)) for termine in termini}
            totale = sum(conteggi.values())
            if totale > 0:
                percentuali_elemento = {termine: (conteggio / totale) * 100 for termine, conteggio in conteggi.items()}
            else:
                percentuali_elemento = {termine: 0 for termine in termini}
            percentuali.append(percentuali_elemento)

        return percentuali

    # Funzione per calcolare la media delle percentuali di ciascun termine B1-B7 per Brazil2 e X0-X5 per MCD3
    def calcola_media_percentuali(percentuali):
        if (model_name == "brazil2"):
            termini = [f'B{i}' for i in range(1, 8)]
        elif (model_name == "mcd3"):
            termini = [f'X{i}' for i in range(0, 6)]

        somme = {termine: 0 for termine in termini}
        conteggio = len(percentuali)

        for elemento in percentuali:
            for termine, valore in elemento.items():
                somme[termine] += valore

        medie = {termine: (somme[termine] / conteggio) for termine in termini}
        medie_ordinate = dict(sorted(medie.items(), key=lambda item: item[1], reverse=True))
        return medie_ordinate

    # Funzione per calcolare la percentuale di presenza di ciascun termine nei vari elementi
    def calcola_percentuale_presenza(risultati):
        if (model_name == "brazil2"):
            termini = [f'B{i}' for i in range(1, 8)]
        elif (model_name == "mcd3"):
            termini = [f'X{i}' for i in range(0, 6)]

        presenza = {termine: 0 for termine in termini}
        totale_elementi = len(risultati)

        for elemento in risultati:
            for termine in termini:
                if re.search(termine, elemento):
                    presenza[termine] += 1

        percentuali_presenza = {termine: (presenza[termine] / totale_elementi) * 100 for termine in termini}
        percentuali_ordinate = dict(sorted(percentuali_presenza.items(), key=lambda item: item[1], reverse=True))
        return percentuali_ordinate



    risultati = analizza_csv(file_path)
    percentuali = calcola_percentuali(risultati)
    medie = calcola_media_percentuali(percentuali)
    presenza = calcola_percentuale_presenza(risultati)
    #print("Dati estratti:", risultati)
    #print("Percentuali di presenza:", percentuali)
    #print("\n\nMedie delle percentuali:", medie)
    print("\nPercentuale di presenza di ciascun termine:", presenza)


