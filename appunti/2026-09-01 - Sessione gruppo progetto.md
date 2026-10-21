## [SealQA](https://huggingface.co/datasets/vtllms/sealqa)

È un dataset e framework di benchmark. Le opzioni suggerite sono:
- Quella di allargare il dataset facendo collezione automatica di dati.
- Quella di aggiungere RAG.

Si potrebbe progettare un altro **agente search-augmented (con RAG)** per ottenere questi dati (per poi farli valutare da SealQA). Però onestamente non la vedo come una grande aggiunta perché i sistemi moderni mi sembrano già molto buoni, tipo il paper ne cita alcuni come [FreshPrompt](https://arxiv.org/abs/2310.03214) ma risale al 2023. Sarebbe un gran lavoro di scraping/retrieval e prompt engineering, il che ha poco a che fare con NLP.

Tipo bisognerebbe forse fare una roba tipo filtrare i dati garbage (e.g. oggi che quasi tutti i siti sono AI slop), gestire *quanto cercare* e riconciliare meglio le fonti (e.g. non che prendi i primi *n* risultati).

IMO 6/10.

## [Healthbench](https://openai.com/index/healthbench/)

È che è pesante perché richiede di essere esperti nel campo medico (che non sono). È un framework per valutare l'accuratezza dei modelli nel fornire risposte in situazioni realistiche nel campo medico. Però io non sono per nulla esperto a riguardo dunque non saprei bene come aggiungere del mio.

IMO 4/10.

---

# SealQA

Uno dei problemi principali è la noise, perché in mezzo allo slop e a training data precedente dell'LLM che istruiscono un'informazione diversa (o anche i dati correnti, come *quale azienda ha superato da poco $1 tr worth of value*, il training set darà i risultati statici fino alla cutoff date mentre i risultati di ricerca forniranno i risultati più rilevanti che possono essere, ad esempio, aziende rilevanti nel contesto americano, che però differiscono da quelle effettive (Apple vs compagnia cinese che invece ha raggiunto il valore prima).

