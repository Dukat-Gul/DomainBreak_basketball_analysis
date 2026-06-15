abdullahtarek:
# 🏀 Basketball Video Analysis

Analyze basketball footage with automated detection of players, ball, team assignment, and more. This repository integrates object tracking, zero-shot classification, and custom keypoint detection for a fully annotated basketball game experience.

Leveraging the convenience of Roboflow for dataset management and Ultralytics' YOLO models for both training and inference, this project provides a robust framework for basketball video analysis.

Training notebooks are included to help you customize and fine-tune models to suit your specific needs, ensuring a seamless and efficient workflow.

## 📁 Table of Contents

1.  [Features](#-features)
2.  [Prerequisites](#-prerequisites)
3.  [Demo Video](#-demo-video)
4.  [Installation](#-installation)
5.  [Training the Models](#-training-the-models)
6.  [Usage](#-usage)
7.  [Project Structure](#-project-structure)
8.  [Future Work](#-future-work)
9.  [Contributing](#-contributing)
10. [License](#-license)

---

## ✨ Features

- Player and ball detection/tracking using pretrained models.
- Court keypoint detection for visualizing important zones.
- Team assignment with jersey color classification.
- Ball possession detection, pass detection, and interception detection.
- Easy stubbing to skip repeated computation for fast iteration.
- Various “drawers” to overlay detected elements onto frames.

---

## 🎮 Demo Video

Below is the final annotated output video.

[![BasketBall Analysis Demo Video](https://img.youtube.com/vi/xWpP0LjEUng/0.jpg)](https://youtu.be/xWpP0LjEUng)

## 🔧 Prerequisites

- Python 3.8+
- (Optional) Docker

---

## ⚙️ Installation

Setup your environment locally or via Docker.

### Python Environment

1. Create a virtual environment (e.g., venv/conda).
2. Install the required packages:

```bash
pip install -r requirements.txt
```

### Docker

#### Build the Docker image:

```bash
docker build -t basketball-analysis .
```

#### Verify the image:

```bash
docker images
```

## 🎓 Training the Models

Harnessing the powerful tools offered by Roboflow and Ultralytics makes it straightforward to manage datasets, handle annotations, and train advanced object detection models. Roboflow provides an intuitive platform for dataset preprocessing and augmentation, while Ultralytics’ YOLO architectures (v5, v8, and beyond) deliver state-of-the-art detection performance.

This repository relies on trained models for detecting basketballs, players, and court keypoints. You have two options to get these models:

1. Download the Pretrained Weights

   - ball_detector_model.pt  
     (https://drive.google.com/file/d/1KejdrcEnto2AKjdgdo1U1syr5gODp6EL/view?usp=sharing)
   - court_keypoint_detector.pt  
     (https://drive.google.com/file/d/1nGoG-pUkSg4bWAUIeQ8aN6n7O1fOkXU0/view?usp=sharing)
   - player_detector.pt  
     (https://drive.google.com/file/d/1fVBLZtPy9Yu6Tf186oS4siotkioHBLHy/view?usp=sharing)

   Simply download these files and place them into the `models/` folder in your project. This allows you to run the pipelines without manually retraining.

2. Train Your Own Models  
   The training scripts are provided in the `training_notebooks/` folder. These Jupyter notebooks use Roboflow datasets and the Ultralytics YOLO frameworks to train various detection tasks:

   - `basketball_ball_training.ipynb`: Trains a basketball ball detector (using YOLOv5). Incorporates motion blur augmentations to improve ball detection accuracy on fast-moving game footage.
   - `basketball_court_keypoint_training.ipynb`: Uses YOLOv8 to detect keypoints on the court (e.g., lines, corners, key zones).
   - `basketball_player_detection_training.ipynb`: Trains a player detection model (using YOLO v11) to identify players in each frame.

   You can easily run these notebooks in Google Colab or another environment with GPU access. After training, download the newly generated `.pt` files and place them in the `models/` folder.

## Once you have your models in place, you may proceed with the usage steps described above. If you want to retrain or fine-tune for your specific dataset, remember to adjust the paths in the notebooks and in `main.py` to point to the newly generated models.

## 🚀 Usage

You can run this repository’s core functionality (analysis pipeline) with Python or Docker.

### 1) Using Python Directly

Run the main entry point with your chosen video file:

```bash
python main.py path_to_input_video.mp4 --output_video output_videos/output_result.avi
```

- By default, intermediate “stubs” (pickled detection results) are used if found, allowing you to skip repeated detection/tracking.
- Use the `--stub_path` flag to specify a custom stub folder, or disable stubs if you want to run everything fresh.

### 2) Using Docker

#### Build the container if not built already:

```bash
docker build -t basketball-analysis .
```

#### Run the container, mounting your local input video folder:

```bash
docker run \
  -v $(pwd)/videos:/app/videos \
  -v $(pwd)/output_videos:/app/output_videos \
  basketball-analysis \
  python main.py videos/input_video.mp4 --output_video output_videos/output_result.avi
```

---

## 🏰 Project Structure

- `main.py`  
  – Orchestrates the entire pipeline: reading video frames, running detection/tracking, team assignment, drawing results, and saving the output video.

- `trackers/`  
  – Houses `PlayerTracker` and `BallTracker`, which use detection models to generate bounding boxes and track objects across frames.

- `utils/`  
  – Contains helper functions like `bbox_utils.py` for geometric calculations, `stubs_utils.py` for reading and saving intermediate results, and `video_utils.py` for reading/saving videos.

- `drawers/`  
  – Contains classes that overlay bounding boxes, court lines, passes, etc., onto frames.

- `ball_aquisition/`  
  – Logic for identifying which player is in possession of the ball.

- `pass_and_interception_detector/`  
  – Identifies passing events and interceptions.

- `court_keypoint_detector/`  
  – Detects lines and keypoints on the court using the specified model.

- `team_assigner/`  
  – Uses zero-shot classification (Hugging Face or similar) to assign players to teams based on jersey color.

- `configs/`  
  – Holds default paths for models, stubs, and output video.

---

## 🔮 Future Work

As we continue to enhance the capabilities of this basketball video analysis tool, several areas for future development have been identified:

1. **Integrating a Pose Model for Advanced Rule Detection**  
   Incorporating a pose detection model could enable the identification of complex basketball rules such as double dribbling and traveling. By analyzing player movements and positions, the system could automatically flag these infractions, adding another layer of analysis to the video footage.

These enhancements will further refine the analysis capabilities and provide users with more comprehensive insights into basketball games.

## 🤝 Contributing

Contributions are welcome!

1. Fork the repository.
2. Create a new branch for your feature or bug fix.
3. Submit a pull request with a clear explanation of your changes.

---

## 🐜 License

This project is licensed under the MIT License.  
See `LICENSE` for details.

---

## 💬 Questions or Feedback?

Feel free to open an issue or reach out via email if you have questions about the project, suggestions for improvements, or just want to say hi!

Enjoy analyzing basketball footage with automatic detection and tracking!








DomainBreak:
# Analisi Tattica di Partite di Basket Tramite Computer Vision

## Panoramica

Questo progetto utilizza la computer vision e il deep learning per analizzare video di partite di basket, estraendo automaticamente dati e insight tattici. Il sistema è in grado di rilevare e tracciare giocatori e palla, assegnare i giocatori alle rispettive squadre, mappare i loro movimenti su una vista tattica 2D e identificare eventi di gioco chiave.

L'obiettivo è fornire uno strumento potente per allenatori, analisti e appassionati per studiare le performance dei giocatori e le strategie di squadra in modo oggettivo e data-driven.

## Demo Visiva

![Demo dell'analisi video](docs/images/demo.gif)

---

## Funzionalità Principali

- **Rilevamento e Tracciamento di Giocatori**: Identifica ogni giocatore sul campo, gli assegna un ID univoco e ne traccia i movimenti per tutta la durata del video.
- **Rilevamento e Tracciamento Palla**: Localizza e segue la palla, gestendo anche situazioni di alta velocità e occlusioni.
- **Assegnazione Automatica delle Squadre**: Analizza il colore delle maglie per dividere i giocatori nelle due squadre avversarie.
- **Mappatura del Campo e Vista Tattica**: Riconosce i punti chiave del campo da basket per calcolare una matrice di omografia. Questa permette di proiettare le posizioni dei giocatori da una vista 2D del video a una mappa tattica 2D standard.
- **Analisi del Possesso Palla**: Determina quale giocatore ha il controllo della palla in ogni istante.
- **Rilevamento di Passaggi e Intercettazioni**: Identifica con successo i passaggi tra compagni di squadra e le palle rubate dagli avversari.
- **Calcolo di Velocità e Distanze**: Misura la distanza percorsa e la velocità istantanea di ogni giocatore.
- **Rilevamento e Classificazione Tiri**: Isola i momenti in cui un giocatore effettua un tiro e classifica il tentativo.
- **Visualizzazione Dati**: Sovrappone tutte le informazioni estratte direttamente sul video di output, includendo bounding box, tracce, ID giocatore, statistiche e una mini-mappa tattica.

- **Output Dati Strutturato**: Oltre al video, l'analisi viene salvata in un file JSON per ulteriori elaborazioni.

---

## Come Funziona

Il sistema processa il video frame per frame attraverso una pipeline di analisi modulare:

1.  **Caricamento**: Il video di input viene letto da `main.py`.
2.  **Rilevamento**: Per ogni frame, i modelli di deep learning (YOLO) vengono usati per rilevare le posizioni di giocatori e palla.
3.  **Tracciamento**: I `Tracker` (basati su filtri di Kalman) associano le rilevazioni correnti a quelle dei frame precedenti per mantenere un tracciamento coerente.
4.  **Mappatura Campo**: Il `CourtKeypointDetector` identifica i punti di riferimento del campo per calcolare la trasformazione omografica.
5.  **Assegnazione Squadre**: Il `TeamAssigner` analizza i colori delle maglie e assegna ogni giocatore a una squadra.
6.  **Analisi di Gioco**: Moduli di logica superiore analizzano i dati di tracciamento per determinare il possesso palla, rilevare passaggi, intercettazioni e tiri.
7.  **Calcolo Metriche**: Vengono calcolate velocità e distanze per ogni giocatore.
8.  **Visualizzazione**: I moduli `Drawers` disegnano sul frame tutte le informazioni elaborate (box, tracce, vista tattica, etc.).
9.  **Output**: Il frame finale arricchito di dati viene salvato nella cartella `output_videos/`.

---

## Configurazione e Utilizzo

L'esecuzione dello script avviene tramite riga di comando, permettendo di specificare le risorse di input e output.

### Argomenti da Riga di Comando

- `--input_video` (Obbligatorio): Specifica il percorso del video da analizzare.
- `--output_video` (Obbligatorio): Specifica il percorso dove salvare il video con le annotazioni.
- `--no_stubs` (Opzionale): Se presente, forza una nuova analisi di tutti i dati, ignorando i file temporanei (`stubs`) salvati da esecuzioni precedenti. Utile per rieseguire l'analisi dopo aver modificato qualche parametro.

### Esempio di Esecuzione

```bash
python main.py --input_video "input_videos/video_1.mp4" --output_video "output_videos/video_1_analyzed.mp4"
```

Per forzare una nuova analisi:

```bash
python main.py --input_video "input_videos/video_1.mp4" --output_video "output_videos/video_1_analyzed.mp4" --no_stubs
```

---

## Formato dei Dati di Output (JSON)

Per ogni video analizzato, il sistema produce un file `.json` contenente tutti i dati estratti, frame per frame. Questo permette analisi quantitative e lo sviluppo di ulteriori visualizzazioni.

Il file JSON ha la seguente struttura:

```json
{
    "0": {
        "frame_number": 0,
        "player_with_ball": null,
        "players": [
            {
                "id": 1,
                "team_id": 0,
                "position_2d": [x1, y1, x2, y2]
            }
        ],
        "ball": [
            {
                "id": 1,
                "position_2d": [x1, y1, x2, y2]
            }
        ],
        "events": []
    },
    "150": {
        "frame_number": 150,
        "player_with_ball": 5,
        "players": [...],
        "ball": [...],
        "events": [
            {
                "event_type": "shot_started",
                "player_id": 5,
                "frame": 150
            }
        ]
    }
}
```

---

## Dettagli Tecnici e Strategie Implementative

Questa sezione approfondisce le tecniche utilizzate nei moduli chiave del progetto.

### Rilevamento Oggetti (Giocatori e Palla)
- **Tecnologia**: Il rilevamento si basa su modelli **YOLO (You Only Look Once)**, forniti dalla libreria `ultralytics`.
- **Modelli**: Vengono utilizzati modelli pre-addestrati su dataset specifici per il basket (contenuti in `roboflow_dataset/` e `basketball-player-detection.../`) per garantire un'alta accuratezza nell'identificare giocatori e la palla in contesti di gioco affollati.

### Tracciamento (Tracking-by-Detection)
- **Strategia**: Il sistema adotta un approccio "Tracking-by-Detection". Rileva oggetti in ogni frame e poi li collega nel tempo.
- **Filtro di Kalman**: Per ogni oggetto tracciato, viene istanziato un **Filtro di Kalman** (`utils/kalman_filter.py`) che predice la posizione dell'oggetto nel frame successivo. 
- **Associazione**: Le nuove rilevazioni vengono associate alle tracce esistenti calcolando la metrica di **Intersection over Union (IoU)** tra le bounding box rilevate e quelle predette. Un'associazione riuscita aggiorna il filtro con la nuova posizione, mentre una mancata associazione può indicare un'occlusione o un oggetto che esce dalla scena.
### Assegnazione Squadre
- **Strategia**: L'assegnazione si basa sull'analisi del colore dominante delle maglie, usando un algoritmo di clustering **K-Means**.
- **Processo**:
    1.  Per ogni giocatore, viene isolata una porzione centrale del suo bounding box (l'area del torso).
    2.  Viene applicato un algoritmo di clustering **K-Means** ai pixel di quest'area per trovare il colore dominante.
    3.  Una volta ottenuti i colori dominanti di tutti i giocatori, un secondo K-Means (con K=2) raggruppa questi colori nei due cluster principali, che rappresentano le due squadre.

### Omografia e Vista Tattica
- **Strategia**: La conversione delle coordinate dalla vista della telecamera a una vista 2D top-down è realizzata tramite una **trasformazione omografica**, calcolata usando `cv2.findHomography` sui punti chiave del campo.
- **Processo**:
    1.  Il `CourtKeypointDetector` rileva punti noti e fissi del campo (es. angoli dell'area dei tre secondi, centro campo).
    2.  Queste coordinate in pixel vengono messe in corrispondenza con le loro coordinate "reali" su un diagramma standard del campo da basket.
    3.  La funzione `cv2.findHomography` di OpenCV calcola la matrice di trasformazione (3x3) che mappa i punti tra i due piani.
    4.  Questa matrice viene poi usata per convertire la posizione di qualsiasi giocatore dal frame del video alla mappa tattica.

### Rilevamento Eventi (Passaggi e Tiri)
- **Logica a Stati**: Il rilevamento di eventi si basa sulla gestione dello stato del gioco, in particolare sul **possesso palla**, che viene assegnato al giocatore più vicino alla palla.
- **Possesso Palla**: Il sistema determina il possesso assegnando la palla al giocatore più vicino, a condizione che la distanza sia inferiore a una certa soglia.
- **Passaggio**: Un evento "passaggio" viene registrato quando lo stato di possesso cambia da `Giocatore A` a `Giocatore B`, dove A e B appartengono alla stessa squadra. Se B è un avversario, l'evento è classificato come **intercettazione**.
- **Tiro**: Un "tiro" è identificato da una sequenza di segnali cinematici: un'improvvisa accelerazione verticale della palla, il superamento di un'altezza relativa alla testa del giocatore e la successiva perdita di possesso.

---

## Struttura del Progetto

Il codice è organizzato in moduli con responsabilità specifiche:

```
.
├── main.py                           # Entry point principale dell'applicazione
├── docs/images/demo.gif              # Demo visiva del progetto
├── requirements.txt                  # Dipendenze Python
├── Dockerfile                        # Per la containerizzazione
├── configs/                          # File di configurazione
├── input_videos/                     # Video da analizzare
├── output_videos/                    # Video e dati JSON analizzati
├── models/                           # Pesi dei modelli pre-addestrati (.pt)
├── training_notebooks/               # Notebooks per il training dei modelli
├── basketball-player-detection.../   # Dataset per il training
├── trackers/                         # Logica per il tracciamento di giocatori e palla
├── team_assigner/                    # Logica per l'assegnazione delle squadre
├── court_keypoint_detector/          # Rilevamento punti chiave del campo
├── tactical_view_converter/          # Calcolo omografia e vista tattica
├── pass_and_interception_detector/   # Rilevamento passaggi e intercettazioni
├── shot_detector/                    # Rilevamento tentativi di tiro
├── speed_and_distance_calculator/    # Calcolo velocità e distanze
├── drawers/                          # Moduli per la visualizzazione dei dati sul video
└── utils/                            # Funzioni di utilità (es. filtro di Kalman)
```

---

## Installazione

1.  **Clonare il repository e navigare nella cartella**
    ```bash
    git clone <URL_DEL_REPOSITORY>
    cd basketball_analysis
    ```
2.  **Creare un ambiente virtuale:**
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # Su Windows: .venv\Scripts\activate
    ```
3.  **Installare le dipendenze:**
    ```bash
    pip install -r requirements.txt
    ```

## Utilizzo

1.  Assicurarsi che i percorsi dei video di input e dei modelli siano configurati correttamente nel file `configs/configs.py`.
2.  Eseguire lo script principale:
    ```bash
    python main.py
    ```
3.  Il video processato verrà salvato nella cartella `output_videos/`.
---

## Possibili Miglioramenti Futuri

Il progetto ha una solida base che può essere ulteriormente estesa.

### 1. Rilevamento e Analisi del Tiro (Shot Detection & Analysis)
**Obiettivo**: Insegnare al sistema a riconoscere la "cinematica" di un tiro per identificarne automaticamente l'esito (canestro/errore) e la tipologia (da 2 o 3 punti).
- **Implementazione**:
    1.  **Definire la Zona del Canestro**: Mappare le coordinate 3D del canestro sull'immagine 2D tramite l'omografia.
    2.  **Identificare il "Trigger" del Tiro**: Riconoscere l'istante del tiro analizzando il movimento della palla (che supera la testa del giocatore) e la perdita di possesso.
    3.  **Tracciare la Traiettoria**: Seguire la traiettoria parabolica della palla dopo il trigger per determinare se interseca la zona del canestro (canestro) o la manca (errore).
    4.  **Classificare il Tipo di Tiro**: Usare la posizione del giocatore al momento del tiro per determinare se è stato effettuato da dentro o fuori l'arco da 3 punti.


### 2. Rilevamento di Tattiche di Squadra (Es: Pick-and-Roll)
**Obiettivo**: Identificare sequenze di movimenti complesse tra due o più compagni di squadra.
- **Implementazione**:
    1.  **Identificare il Blocco (Pick)**: Rilevare quando un giocatore (bloccante) si avvicina a un compagno in possesso palla (palleggiatore) e diventa quasi stazionario.
    2.  **Rilevare l'Uso del Blocco**: Verificare che il palleggiatore cambi direzione muovendosi attorno alla posizione del bloccante.
    3.  **Tracciare il Movimento del Bloccante (Roll)**: Verificare che il bloccante, dopo il blocco, si muova decisamente verso il canestro.
    4.  **Confermare l'Evento**: Se i tre passi avvengono in sequenza, registrare un evento "Pick-and-Roll".

### 3. Analisi Avanzata del Palleggio e del Possesso

**Obiettivo**: Estrarre dati più profondi sulla gestione della palla per capire lo stile di gioco e il processo decisionale dei giocatori.

- **Implementazione**:
    1.  **Contare i Palleggi**: Rilevare ogni ciclo di movimento verticale della palla vicino a un giocatore in possesso come un singolo palleggio.
    2.  **Calcolare il Tempo di Possesso Effettivo**: Cronometrare la durata di ogni possesso per ogni giocatore.
    3.  **Creare Metriche Avanzate**: Calcolare statistiche come "palleggi per possesso" o "tempo decisionale medio" per distinguere tra playmaker e finalizzatori.
