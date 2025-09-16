# Scopo e Obiettivo
Guidare la progettazione e la stesura di una tesi in Visione Artificiale applicata al basket, utilizzando LaTeX su VS Code, con particolare attenzione alla struttura generale fornita e garantendo completezza, coerenza e sufficiente materiale per ogni capitolo, inclusi immagini e idee dettagliate dove opportuno.

# Best Practice Checklist
- Begin with a concise checklist (3-7 bullets) of sub-tasks da seguire per l'analisi della scaletta: (1) Analisi sezione per sezione, (2) Verifica copertura argomenti, (3) Stima materiale/pagine, (4) Identificazione di gap o carenze, (5) Proposta di estensioni e figure, (6) Suggerimento titoli alternativi, (7) Produzione di sola conferma finale.

# Istruzioni Principali
- Analizza la scaletta proposta della tesi e valuta la completezza degli argomenti, suggerendo aggiunte dove necessario e stimando la possibilità di rispettare il numero di pagine previsto (utilizzando immagini e descrizioni suggerite).
- Se utile, proponi estensioni e dettagli sotto forma di elenchi puntati per ciascuna sezione.
- Fornisci idee per titoli alternativi o modifiche a quello suggerito.
- Segui questo ordine per la stesura effettiva dei capitoli: 2, 3, 4, 1, 5.
- Quando l’analisi strutturale è terminata, comunica esclusivamente che è stata completata e attendi istruzioni successive.

# Sotto-categorie
- Per ogni sezione suggerisci immagini, articoli o figure che possono arricchire l’elaborato.
- Evidenzia eventuali carenze e suggerisci temi aggiuntivi dove opportuno.

# Contesto
- Lo studente sta lavorando su una tesi in Visione Artificiale per il basket focalizzata sull’analisi video, tramite LaTeX Workshop in VS Code.
- È richiesta una analisi critica della scaletta proposta con suggerimenti concreti di miglioramento e arricchimento, inclusa l’individuazione di immagini descrittive.

# Reasoning Effort
- Set reasoning_effort = medium: sufficient depth per analisi critica e suggerimenti puntuali, comunicando però solo la conferma a lavoro finito.

# Planning e Verifica
- Elenca e verifica i capitoli previsti.
- Stima se il materiale trattato è sufficiente per il numero di pagine raccomandato per ciascun capitolo.
- Suggerisci argomenti supplementari in caso di materiale insufficiente.
- Proponi titoli migliorati dove appropriato.

# Output Format
- Al termine restituisci solo una breve conferma: nessun dettaglio di analisi, solo la conferma dell’avvenuto completamento.

# Verbosità
- Sintetica e puntuale: manifestare esclusivamente la conferma di analisi conclusa.

# Condizioni di Stop
- Termina quando è stata condotta e comunicata l’analisi preliminare della struttura e attendi istruzioni ulteriori.

# Possibile Scaletta
Ecco una bozza della possibile scaletta della tesi, ragiona e prova a comprendere se copre tutti gli argomenti di cui si potrebbe trattare e soprattutto se ci sarebbe il materiale per scrivere più o meno il numero di pagine che è indicato per ciascun capitolo (considera che in una tesi del genere si dovrebbero inserire molte immagini - guidami anche su quelle, magari descrivendomi, quando necessaria, con una bella descrizione articolata e idee generali). 
Se necessario, prova ad espandere ciascuna sezione con un elenco puntato degli argomenti che potrebbe essere opportuno inserire. 
Possiamo produrre tutte le modifiche che vogliamo alla scaletta.

la struttura generale, che mi è stata fornita modi scaletta dal mio professore:

\documentclass[12pt,a4paper,openright,twoside]{book}

% input e lingua
\usepackage[utf8]{inputenc}
\usepackage[english,italian]{babel} % italiano come lingua principale

% Template e pacchetti del tamplate
\usepackage{disi-thesis}
\usepackage{code-lstlistings}
\usepackage{notes}
\usepackage{shortcuts}

% Tabelle
\usepackage{booktabs}

% Acronomi: usa 'acro' (NON 'acronym')
\usepackage{acro}

% Bibliografia: usa biblatex+biber per \parencite
\usepackage{csquotes}
\usepackage[
  backend=biber,
  style=alphabetic,   % cambia se preferisci
  sorting=ynt
]{biblatex}
\addbibresource{bibliography.bib}

\school{\unibo}
\programme{Corso di Laurea in Ingegneria e Scienze Informatiche}
\title{Sviluppo di un Sistema di Visione Artificiale per l'Analisi di Video di Pallacanestro}
\author{Nome Cognome}
\date{\today}
\subject{Visione Artificiale}
\supervisor{Prof. Nome Cognome}
\session{II}
\academicyear{2024-2025}

% Definition of acronyms
% Acronyms (acro)
\DeclareAcronym{iot}{
    short = IoT ,
    long  = Internet of Things
}
\DeclareAcronym{vm}{
  short = VM ,
  long  = Virtual Machine ,
}
\DeclareAcronym{vms}{
  short = VMs ,
  long  = Virtual Machines
}
\DeclareAcronym{od}{
    short = OD ,
    long  = Object Detection
}
\DeclareAcronym{ot}{
    short = OT ,
    long  = Object Tracking
}
\DeclareAcronym{cv}{
    short = CV ,
    long  = Computer Vision
}
\DeclareAcronym{rgb}{
    short = RGB ,
    long = Red Green Blue
}
\DeclareAcronym{yuv}{
    short = YUV ,
    long = YUV Color Space
}
\DeclareAcronym{fps}{
    short = fps ,
    long = frames per second
}
\DeclareAcronym{id}{
    short = ID ,
    long = identity
}
\DeclareAcronym{kpi}{
    short = KPI ,
    long = Key Performance Indicator
}



\mainlinespacing{1.241} % line spacing in mainmatter, comment to default (1)

\begin{document}

\frontmatter\frontispiece

\begin{abstract}	
Max 2000 characters, strict.
\end{abstract}

\begin{dedication} % this is optional
Optional. Max a few lines.
\end{dedication}

%----------------------------------------------------------------------------------------
\tableofcontents   
\listoffigures     % (optional) comment if empty
\lstlistoflistings % (optional) comment if empty
%----------------------------------------------------------------------------------------

\mainmatter

% THIS WAS INITIALLY INSIDE THE INTRODUCTION CHAPTER - IT MAY BE USEFUL DO NOT DELETE IT - |FOR NOW|
% You can use acronyms that your defined previously,
% such as \ac{iot}.
% %
% If you use acronyms twice,
% they will be written in full only once
% (indeed, you can mention the \ac{iot} now without it being fully explained).
% %
% In some cases, you may need a plural form of the acronym.
% %
% For instance,
% that you are discussing \ac{vms},
% you may need both \ac{vm} and \ac{vms}.

% \paragraph{Structure of the Thesis}

% \note{At the end, describe the structure of the paper}
%----------------------------------------------------------------------------------------
\chapter{Introduzione (2-3 pagine)}
\label{chap:intro}
% \sidenote{Add sidenotes in this way. They are named after the author of the thesis}
%----------------------------------------------------------------------------------------
\section{Contesto e Motivazioni:}
\label{sec:contesto-motivazioni}
Iniziare descrivendo la crescente importanza dell'analisi dati e della tecnologia nel mondo dello sport, 
con un focus sul basket. Spiegare come l'analisi automatizzata possa fornire insight preziosi a coach, 
atleti e analisti, superando i limiti di tempo e costo dell'analisi manuale.
\section{Il Problema:}
\label{sec:problema}
Definire in modo chiaro e formale i problemi che la tesi affronta:
\begin{enumerate}
  \item La necessità di un sistema automatico per ricostruire la traiettoria della palla durante un tiro 
  a canestro a partire da un video.
  \item L'esigenza di contare in modo affidabile il numero di giocatori presenti in una determinata area 
  del campo o in un frame video.
  \item ?Altri?
\end{enumerate}

\section{Obiettivi della Tesi}
\label{sec:obiettivi-tesi}
Elencare gli scopi concreti del progetto:
\begin{itemize}
  \item Studiare lo stato dell'arte delle tecniche di computer vision per l'analisi del basket.
  \item Creare e/o adattare un dataset di video, etichettandolo per creare un ground truth affidabile.
  \item Sviluppare e addestrare un modello per il rilevamento dei giocatori.
  \item Implementare un algoritmo per il tracciamento della traiettoria della palla.
  \item Valutare le performance dei modelli sviluppati tramite metriche quantitative.
\end{itemize}


\section{Struttura della Tesi}
\label{sec:struttura-tesi}
Fornire una breve panoramica dei capitoli successivi, guidando il lettore attraverso il documento.

%----------------------------------------------------------------------------------------
\chapter{Stato dell'Arte (10-15 pagine)}
\label{chap:stato-dell-arte}
%----------------------------------------------------------------------------------------

\section{Elementi di Visione Artificiale}
\label{sec:elementi-vision-artificiale}
Fornire una breve introduzione ai concetti fondamentali necessari per comprendere il progetto: elaborazione
di immagini, object detection, object tracking.

\section{Algoritmi di Rilevamento e Tracciamento di Oggetti:}
\label{sec:algoritmi-rilevamento-tracciamento}

\subsection{Approcci Tradizionali (cenni):}
\label{sec:approcci-tradizionali}
Menzionare brevemente tecniche classiche (es. background subtraction) e i loro limiti in contesti dinamici 
come lo sport.

\subsection{Approcci basati su Deep Learning:}
\label{sec:approcci-deep-learning}
Concentrarsi sulle Reti Neurali Convoluzionali (CNN). Descrivere architetture moderne per l'object detection
come YOLO, SSD o Faster R-CNN, spiegando perché sono particolarmente adatte al contesto sportivo. 
Per il tracking, introdurre concetti come i filtri di Kalman o tracker più moderni come DeepSORT.

\section{La Visione Artificiale applicata al Basket:}
\label{sec:cv-applicata-basket}
Analizzare la letteratura scientifica esistente. Esaminare cosa è già stato fatto in questo campo, cercando
studi su:
\begin{itemize}
  \item Tracciamento dei giocatori (player tracking).
  \item Rilevamento di eventi (tiro, palleggio, passaggio).
  \item Tracciamento della palla.
  \item Analisi tattica automatizzata.
\end{itemize}

\section{Dataset Pubblici per l'Analisi Sportiva:}
\label{sec:dataset-pubblici}
Descrivere brevemente i dataset già e spiegare perché, nonostante la loro esistenza, si è reso necessario 
registrare e/o etichettare video specifici per questo progetto.



%----------------------------------------------------------------------------------------
\chapter{Metodologia e Sviluppo del Progetto (15-25 pagine)}
\label{chap:metodologia-sviluppo}
%----------------------------------------------------------------------------------------
\section{Architettura del Sistema:}
\label{sec:architettura-sistema}
Descrivere il flusso di lavoro complessivo del progetto, possibilmente con un diagramma. 
Esempio: Input Video $\rightarrow$ Pre-elaborazione Frame $\rightarrow$ Modello di Rilevamento 
Giocatori $\rightarrow$ Modello di Tracciamento Palla $\rightarrow$ Output Dati Analitici.

\section{Raccolta e Preparazione dei Dati:}
\label{sec:raccolta-dati}

\subsection{Dataset Utilizzati:}
\label{sec:dataset-utilizzati}
Descrivere nel dettaglio i dati usati. Spiegare quali dataset sono stati trovati online e quali 
caratteristiche avevano. Dettagliare il processo di registrazione dei video custom (tipo di telecamera,
prospettiva, condizioni di luce).

\subsection{Annotazione e Creazione del Ground Truth:}
\label{sec:annotazione-ground-truth}
Spiegare come sono stati etichettati i video per ottenere i dati di riferimento. Descrivere il software
utilizzato, il formato delle etichette e il processo seguito per garantire la qualità delle annotazioni.

\section{Task 1: Rilevamento e Conteggio dei Giocatori:}
\label{sec:task-1}

\subsection{Scelta del Modello:}
\label{sec:scelta-modello}
Giustificare la scelta del modello utilizzato (es. "È stato scelto il modello YOLOv8 per il suo 
ottimo compromesso tra velocità di inferenza e accuratezza...").

\subsection{Addestramento del Modello:}
\label{sec:addestramento-modello}
Descrivere il processo di training: iperparametri utilizzati (learning rate, numero di epoche, batch size),
l'hardware su cui è stato effettuato l'addestramento, e come è stato suddiviso il dataset (training set,
validation set, test set).

\section{Task 2: Rilevamento della Traiettoria della Palla:}
\label{sec:task-2}

\subsection{Approccio Implementato:}
Descrivere il metodo implementato.

\subsection{Dettagli Implementativi:}
Discutere le sfide tecniche affrontate e le relative soluzioni: la velocità della palla, il motion blur,
le occlusioni da parte dei giocatori, il cambio di illuminazione, …


%----------------------------------------------------------------------------------------
\chapter{Risultati Sperimentali e Valutazione (10-20 pagine)}
\label{chap:risultati-sperimentali}
 %----------------------------------------------------------------------------------------
\section{Metriche di Valutazione:}
\label{sec:metriche-valutazione}
\begin{itemize}
  \item \textbf{Per il Rilevamento Giocatori:} Errore medio percentuale rispetto al ground truth?.
  \item \textbf{Per la Traiettoria della Palla:} Errore Quadratico Medio (MSE) tra le coordinate
  predette e quelle del ground truth?
\end{itemize}

\section{Risultati del Conteggio Giocatori:}
\label{sec:risultati-conteggio-giocatori}
Presentare i risultati in modo chiaro usando tabelle e grafici. Confrontare le performance ottenute sul
dataset pubblico e sui video registrati. Discutere eventuali differenze. Includere esempi visivi 
(frame con i bounding box disegnati dal modello).

\section{Risultati del Tracciamento della Traiettoria:}
\label{sec:risultati-tracciamento-traiettoria}
Mostrare l'efficacia dell'algoritmo di tracciamento. Utilizzare grafici che sovrappongono la traiettoria 
reale (ground truth) e quella predetta dal modello. Quantificare l'errore usando le metriche definite e 
fornire esempi visivi.

\section{Analisi e Discussione dei Risultati:}
\label{sec:analisi-risultati}
Interpretare i dati ottenuti. Analizzare perché i modelli si sono comportati in un certo modo, quali sono
 state le principali cause di errore e come potrebbero essere migliorati i risultati.


 %----------------------------------------------------------------------------------------
\chapter{Conclusioni e Sviluppi Futuri (1-2 pagine)}
\label{chap:conclusioni-sviluppi-futuri}
 %----------------------------------------------------------------------------------------

 \section{Sintesi del Lavoro Svolto:}
 \label{sec:sintesi-lavoro-svolto}
 Riassumere brevemente gli obiettivi, le attività svolte e i principali risultati raggiunti, ribadendo il 
 contributo della tesi.

 \section{Limiti del Progetto:}
 \label{sec:limiti-progetto}
 Essere onesti riguardo ai limiti del lavoro. Ad esempio, il sistema funziona solo con una certa 
 inquadratura? È sensibile a cambi di luce?

\section{Sviluppi Futuri:}
\label{sec:sviluppati-futuri}
Proporre possibili miglioramenti ed estensioni del progetto. Idee potrebbero essere:
\begin{itemize}
  \item Passare dal conteggio al tracciamento individuale dei giocatori.
  \item Rilevare automaticamente eventi di gioco complessi (canestro segnato, fallo, passaggio).
  \item Ottimizzare il sistema per un'analisi in tempo reale (real-time).
  \item Ampliare il dataset per coprire una maggiore varietà di situazioni di gioco.
  \item ?Altri?
\end{itemize}


 %----------------------------------------------------------------------------------------
 \chapter{Contribution}
  %----------------------------------------------------------------------------------------

\section{Some cool topic}

You may also put some code snippet (which is NOT float by default), eg: \cref{lst:random-code}.

\lstinputlisting[
  float,
  language=Java,
  caption={Hello World in Java},
  label={lst:random-code}
]{listings/HelloWorld.java}

\section{Fancy formulas here}

%----------------------------------------------------------------------------------------
% BIBLIOGRAPHY
%----------------------------------------------------------------------------------------

\backmatter

\nocite{*} % rimuovi quando non più necessario
\printbibliography

\begin{acknowledgements} % this is optional
Optional. Max 1 page.
\end{acknowledgements}

\end{document}



Quando hai analizzato l'intera struttura non fare niente, dimmelo e basta.
Dopo ti fornisco ulteriori istruzioni. 