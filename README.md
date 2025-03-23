# 🐍 M3GP for ML project 
Questo progetto è stato realizzato da **[Arianna Cella](https://github.com/ariannaCella)** ed **[Elena Azzi](https://github.com/azzielena)** per il corso di Machine Learning erogato dall'Università di Parma.

## 📖 Guida pratica di utilizzo del codice

Questa guida descrive passo-passo come impostare ed eseguire il codice M3GP per effettuare esperimenti sui dataset desiderati.

---

## 🚀 Installazione progetto

Scaricare e installare il nostro progetto dal repository GitHub:

```bash
git clone https://github.com/jespb/Python-M3GP.git
cd Python-M3GP
pip install -r requirements.txt
```

---

## 📂 Organizzazione Cartelle

Il progetto è suddiviso in due cartelle principali:

- **Python-M3GP**: esegue M3GP sui dataset di interesse, generando un file CSV con le metriche calcolate e le migliori hyperfeatures.
- **Classification_with_hyperfeatures**: per ciascun dataset, crea nuovi dataset arricchiti con le hyperfeatures e confronta i risultati della classificazione tra il dataset originale e quello aumentato, utilizzando tre modelli di Machine Learning (Random Forest, Decision Tree e XGBoost).

### 🏗️ Struttura delle cartelle principali

#### 📁 Python-M3GP/
```
│
├── datasets/
│   ├── nose.csv
│   ├── mcd3.csv
│   └── brazil2.csv
│
├── m3GP/
│   └── [vari file per eseguire m3GP]
│
├── results/
│   ├── m3gpWAF_brazil2.csv
│   ├── m3gpWAF_mcd3.csv
│   └── m3gpWAF_nose.csv
│
├── M3GP_naso_core.py
├── M3GP_RS_core.py
├── visual_results.py
└── Arguments.py
```

#### 📁 Classification_with_hyperfeatures/
```
│
├── tabelle_hyperfeatures/
│   ├── brazil2_WAF_hyper/
│   │   └── 30 dataset brazil2 aumentati con HF
│   ├── mcd3_WAF_hyper/
│   │   └── 30 dataset im3 aumentati con HF
│   └── nose_WAF_hyper/
│       └── 30 dataset nose aumentati con HF
│
├── creazioneDataset.py
├── nose_comparison_classificator.py
├── RS_comparison_classificator.py
└── risultati_metriche_nose.csv
```

---

## 🌍 Esecuzione Remote Sensing

### 1️⃣ Entriamo nella cartella di interesse:
```bash
cd Python-M3GP
```

### 2️⃣ Configurazione dei parametri

Modificare il file `Arguments.py` per impostare i parametri desiderati:

- **DATASETS**: specificare il nome del dataset (`mcd3` o `brazil2`).
- **OUTPUT_DIR**: indicare la cartella dove salvare i risultati.
- **MODEL_NAME**: impostare il modello (default: Mahalanobis Distance Classifier).
- **RUNS**: numero di esecuzioni (default: 30).
- **MAX_GENERATION**: numero di generazioni (default: 50).
- **FITNESS_TYPE**: scegliere il tipo di fitness (default: Accuracy, consigliato: WAF per migliori performance).
- **RANDOM_STATE**: 0 per ottenere i nostri risultati (default: 42).

### 3️⃣ Esecuzione del codice
```bash
python M3GP_RS_core.py
```

### 4️⃣ Risultati

Al termine delle 30 esecuzioni, verrà prodotto un file CSV contenente tutte le metriche calcolate e salvato nella cartella `results`. I file generati saranno:

- `m3gpWAF_brazil2.csv` per il dataset brazil2.
- `m3gpWAF_mcd3.csv` per il dataset mcd3.

### 5️⃣ Visualizzazione dei risultati

Utilizzare `visual_results.py` per analizzare le metriche riportate nei file CSV di output, modificando le righe 6,7,8 con il dataset di interesse.

### 6️⃣ Creazione dataset aumentati
```bash
cd classification_with_hyperfeatures
python creazioneDataset.py
```

I nuovi dataset saranno salvati nella cartella `tabelle_hyperfeatures`.

### 7️⃣ Classificazione con DT, RF e XGB
```bash
python RS_comparison_classificator.py
```

I risultati ottenuti da ciascun modello ML saranno confrontati tra dataset originale e aumentato, visualizzando un boxplot.

---

## 👃 Esecuzione Nose

### 1️⃣ Entriamo nella cartella di interesse:
```bash
cd Python-M3GP
```

### 2️⃣ Configurazione dei parametri

Modificare il file `Arguments.py` per impostare i parametri desiderati:

- **DATASETS**: specificare il nome del dataset `nose.csv`.
- **OUTPUT_DIR**: indicare la cartella dove salvare i risultati.
- **MODEL_NAME**: impostare il modello (default: Mahalanobis Distance Classifier).
- **RUNS**: numero di esecuzioni (default: 30).
- **MAX_GENERATION**: numero di generazioni (default: 50).
- **FITNESS_TYPE**: scegliere il tipo di fitness (default: Accuracy, consigliato: WAF per migliori performance).
- **RANDOM_STATE**: 0 per ottenere i nostri risultati (default: 42).

### 3️⃣ Esecuzione del codice
```bash
python M3GP_naso_core.py
```

### 4️⃣ Risultati

Al termine delle 30 esecuzioni, verrà prodotto un file CSV contenente tutte le metriche calcolate e salvato nella cartella `results`. Il file generato sarà:

- `m3gpWAF_nose.csv`

### 5️⃣ Visualizzazione dei risultati

Utilizzare `visual_results.py` per analizzare le metriche, modificando le righe 6,7,8 con il dataset di interesse.

### 6️⃣ Creazione dataset aumentati
```bash
cd classification_with_hyperfeatures
python creazioneDataset.py
```

I nuovi dataset saranno salvati nella cartella `tabelle_hyperfeatures`.

### 7️⃣ Classificazione con DT, RF e XGB
```bash
python nose_comparison_classificator.py
```

I risultati ottenuti saranno visualizzati con un boxplot e salvati nel file `risultati_metriche_nose.csv`.

---

## 🏁 Conclusione

Seguendo questi passaggi, sarà possibile eseguire M3GP sui dataset desiderati, confrontare le prestazioni tra dataset originali e aumentati e analizzare i risultati con metodi di Machine Learning avanzati.

