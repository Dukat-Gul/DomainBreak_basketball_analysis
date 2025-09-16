**Ruolo & Obiettivo**
Agisci come redattore accademico. Scrivi il **Capitolo 5 – Conclusioni e Sviluppi Futuri** della tesi *“Sviluppo di un Sistema di Visione Artificiale per l’Analisi di Video di Pallacanestro”* in **italiano**, stile **accademico ma leggibile**. **Lunghezza**: **1–2 pagine** (≈ 400–800 parole), con **1 figura o 1 box** massimo.

**Vincoli**

* Voce impersonale, concisa, nessun codice.
* Richiama **obiettivi/metriche chiave** e i **risultati** del Cap. 4 (mAP, MAE/RMSE, ADE/FDE, FPS/latenza).
* Evidenzia **limiti pratici indoor** (occlusioni, illuminazione, riflessi parquet, camera shake, palla piccola/veloce, maglie simili).
* “Sviluppi Futuri” organizzati per **impatto** e **fattibilità** (breve motivazione + sforzo atteso).

**Struttura**

* **5.1 Sintesi del Lavoro Svolto** (150–250 parole)
* **5.2 Limiti del Progetto** (100–200 parole, elenco puntato ragionato)
* **5.3 Sviluppi Futuri** (150–300 parole, elenco strutturato **Impatto/Fattibilità**)

**Output opzionali (uno solo):**

* **Figura 5.1 “Roadmap di miglioramento”** *oppure* **Box 5.A “Contributi della tesi in 5 punti”**.

---

# Scaletta espansa (contenuti suggeriti)

## 5.1. Sintesi del Lavoro Svolto

* **Richiamo degli obiettivi** (O1–O5): revisione, dati/annotazioni, detector giocatori, stima traiettoria palla, valutazione quantitativa.
* **Metodologia in breve**: architettura pipeline; dataset pubblico + video propri; protocolli (split, IC 95%).
* **Risultati chiave** (placeholder numerici da riempire):

  * *Detection*: mAP@\[.5:.95], AP50/AP75.
  * *Conteggio*: MAE/RMSE/MAPE per frame/clip.
  * *Traiettoria palla*: RMSE (px/cm), ADE/FDE, Success\@r.
  * *Efficienza*: FPS, latenza mediana/p95, device.
* **Contributo**: prototipo funzionante + linee guida di valutazione riproducibili.

## 5.2. Limiti del Progetto

* **Dominio & generalizzazione**: inquadratura e prospettiva specifiche; performance degradano con angoli estremi o zoom dinamico.
* **Illuminazione & riflessi**: sensibilità a luci puntiformi e glare sul parquet; FP su highlights.
* **Occlusioni & crowding**: undercount in raggruppamenti stretti; recovery della palla ritardato.
* **Piccoli oggetti & motion blur**: palla poco risolta ad alte velocità; errori durante shutter ridotto.
* **Dipendenza dall’hardware**: throughput reale legato a GPU; latenza non garantita su CPU-only.
* **Scope**: niente identificazione individuale, tattiche complesse o multi-camera (se non trattati).

## 5.3. Sviluppi Futuri (con priorità)

* **Tracking individuale dei giocatori (ID persistente)**

  * *Impatto*: analisi line-up/spacing; *Fattibilità*: media (Re-ID + tracker MOT, tuning su occlusioni).
* **Event detection complessi (canestro, fallo, rimbalzo, passaggio)**

  * *Impatto*: insight tattici; *Fattibilità*: media-alta (fusioni regole + modelli sequenziali/temporal CNN).
* **Real-time stabile**

  * *Impatto*: usi a bordo campo; *Fattibilità*: media (quantizzazione, TensorRT/ONNX, batching dinamico, ROI adattive).
* **Dataset ampliato & benchmark pubblico**

  * *Impatto*: generalizzazione; *Fattibilità*: media (protocollo di annotazione, licenze, release split).
* **Rettifica del campo & coordinate metriche**

  * *Impatto*: traiettorie in cm, heatmap per zona; *Fattibilità*: bassa-media (omografia robusta, corner detection).
* **Robustezza a condizioni avverse**

  * *Impatto*: stabilità; *Fattibilità*: alta (augmentation mirata, stabilizzazione video, controllo esposizione).
* **Multi-camera (se rilevante)**

  * *Impatto*: tracking occlusioni zero; *Fattibilità*: bassa (sincronizzazione, calibrazione, fusione).

---

# Idee per elementi grafici (scegline uno)

* **Figura 5.1 – Roadmap di miglioramento (12 mesi)**
  Timeline in 3 fasi: *Stabilità & Real-time* → *Eventi & Tracking ID* → *Multi-camera & Metriche in cm*. Ogni blocco con 2–3 deliverable e metriche target (es. +5% IDF1, latenza < 40 ms, RMSE palla < X cm).

* **Box 5.A – Contributi della tesi (in 5 punti)**
  Elenco conciso: (1) pipeline riproducibile, (2) dataset/GT con protocollo, (3) detector/ball-tracker ottimizzati per basket indoor, (4) valutazione completa con IC 95%, (5) linee guida operative per deployment.

---

# Stima pagine

* Testo (senza grafica): \~1,2–1,6 pagine.
* Con **1** figura/box: \~1,5–2,0 pagine.
  → Il range **1–2 pagine** è realistico.

---

# Checklist finale

* [ ] Richiamati **obiettivi** e **metriche** principali con 2–3 numeri chiave dal Cap. 4.
* [ ] **Limiti** esplicitati e collegati alle condizioni indoor reali.
* [ ] **Sviluppi futuri** ordinati per impatto/fattibilità con micro-motivazione.
* [ ] Inserita **una** figura *o* un box (non entrambi se vuoi restare nelle 2 pagine).
* [ ] Rimando chiaro a come i miglioramenti verrebbero **misurati** (stesse metriche di Cap. 4).

Se vuoi, posso generarti un **mini-template LaTeX** del Cap. 5 con segnaposto per *Figura 5.1* o *Box 5.A* e struttura già etichettata (`\label{fig:roadmap}`, `\label{box:contributi}`).
