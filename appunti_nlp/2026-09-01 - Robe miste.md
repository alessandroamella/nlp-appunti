# TF-IDF e normalizzazione (pag. 24–28, "02 - Neural network primer")

## Pag. 24 — Matrice TF-IDF

Matrice con:
- **righe** = documenti
- **colonne** = parole del vocabolario

Ogni cella:

$$
\text{TF-IDF}(d, w) = tf(d, w) \cdot idf(w)
$$

- $tf(d,w)$: frequenza della parola $w$ nel documento $d$ (è il bag-of-words)
- $idf(w)$: quanto è rara/informativa $w$ nell'intero corpus (calcolata una sola volta su tutto il corpus)

> Moltiplicare per $idf(w)$ equivale a riscalare **l'intera colonna** $w$ per una costante → l'IDF è già di fatto una normalizzazione di colonna (torna utile a pag. 28).

![[assets/2026-09-01/nn-primer-024.png]]

## Pag. 25–27 — Normalizzazione per riga (row normalization)

**Problema**: documenti lunghi hanno più entrate non-nulle → vettori-riga più "lunghi" nello spazio. Questo sbilancia algoritmi basati su distanza (cosine similarity, k-NN, clustering): due documenti sullo stesso argomento ma di lunghezza diversa finiscono lontani solo per la lunghezza del testo, non per il contenuto.

**Soluzione**: normalizzare ogni riga (ogni documento).

![[assets/2026-09-01/nn-primer-025.png]]

### L2-normalization

Dividi il vettore per la sua norma euclidea:

$$
v_{\text{norm}} = \frac{v}{\lVert v \rVert_2}, \qquad \lVert v \rVert_2 = \sqrt{x_1^2 + x_2^2 + \dots + x_n^2}
$$

Ogni riga ottiene distanza euclidea **= 1** dall'origine → tutti i punti finiscono su una circonferenza (ipersfera) unitaria: $x_1^2 + x_2^2 = 1$.

### L1-normalization

Dividi per la somma dei valori assoluti:

$$
v_{\text{norm}} = \frac{v}{|x_1| + |x_2| + \dots + |x_n|}
$$

La somma delle componenti della riga fa **1** → i punti finiscono su un "rombo unitario": $|x_1| + |x_2| = 1$. Se i valori sono non-negativi (tipico di tf-idf), la riga normalizzata si può interpretare come una **distribuzione di probabilità** sulle parole.

Esempio dal grafico:
- L2: $(-1.2,\ 0.9) \to (-0.8,\ 0.6)$ — stessa direzione, lunghezza 1
- L1: $(-1.2,\ 0.4) \to (-0.75,\ 0.25)$ — stessa direzione, $|-0.75|+|0.25|=1$

![[assets/2026-09-01/nn-primer-026-crop.png]]

**Domanda aperta (pag. 27)**: quale usare? Non c'è una risposta universale — dipende dall'algoritmo a valle. Se serve interpretare le righe come probabilità → L1 è più naturale. Se si userà cosine similarity → L2 è lo standard (è anche il default di `TfidfVectorizer` in scikit-learn).

![[assets/2026-09-01/nn-primer-027.png]]

## Pag. 28 — Normalizzazione per colonna (column normalization)

Prospettiva opposta: si normalizzano le **parole** (colonne), non i documenti.

- L'**IDF stesso è già normalizzazione di colonna** (vedi nota a pag. 24): riduce il peso di parole troppo comuni (es. "the", "is") che compaiono ovunque.
- In più si possono applicare tecniche standard di scaling per colonna:

**Min-max scaling**:
$$
x' = \frac{x - \min(x)}{\max(x) - \min(x)}
$$

**Standardizzazione (z-score)**: sottrai la media della colonna, dividi per la deviazione standard $\sigma$:
$$
x' = \frac{x - \mu}{\sigma}
$$
(risultato: media 0, varianza 1 per ogni colonna)

**Domanda aperta**: anche qui, quale (se una) usare va valutato empiricamente per il task.

![[assets/2026-09-01/nn-primer-028.png]]

---

## Esempio completo passo-passo

Corpus di 3 documenti, vocabolario di 3 parole (`gatto`, `cane`, `corre`):

| doc | testo |
|---|---|
| $d_1$ | "il gatto corre" |
| $d_2$ | "il gatto il gatto" |
| $d_3$ | "il cane corre corre" |

### 1. Term frequency (tf)

| doc | gatto | cane | corre |
|---|---|---|---|
| $d_1$ | 1 | 0 | 1 |
| $d_2$ | 2 | 0 | 0 |
| $d_3$ | 0 | 1 | 2 |

### 2. Inverse document frequency (idf)

$N = 3$ documenti totali. $df(w)$ = in quanti documenti compare $w$ almeno una volta.

$$
idf(w) = \ln\!\left(\frac{N}{df(w)}\right)
$$

- `gatto`: $df=2$ (in $d_1, d_2$) $\to idf = \ln(3/2) = 0.405$
- `cane`: $df=1$ (solo in $d_3$) $\to idf = \ln(3/1) = 1.099$
- `corre`: $df=2$ (in $d_1, d_3$) $\to idf = \ln(3/2) = 0.405$

Nota: `cane` è la parola più rara → il suo idf è il più alto → peserà di più quando compare.

### 3. Matrice TF-IDF ($tf \cdot idf$)

| doc | gatto | cane | corre |
|---|---|---|---|
| $d_1$ | $1 \times 0.405 = 0.405$ | $0$ | $1 \times 0.405 = 0.405$ |
| $d_2$ | $2 \times 0.405 = 0.811$ | $0$ | $0$ |
| $d_3$ | $0$ | $1 \times 1.099 = 1.099$ | $2 \times 0.405 = 0.811$ |

Si vede già l'effetto: $d_2$ ha solo "gatto" ripetuto, ma il suo valore ($0.811$) non è il doppio "informativo" di $d_1$ perché "gatto" è comune (idf basso). $d_3$ ottiene un peso alto su `cane` proprio perché è una parola rara.

### 4. Normalizzazione di riga

Prendiamo il vettore-riga di ciascun documento e lo normalizziamo.

**L2** — $v_{\text{norm}} = v / \lVert v \rVert_2$:

- $d_1 = (0.405,\ 0,\ 0.405)$, $\lVert d_1 \rVert_2 = \sqrt{0.405^2+0.405^2} = 0.573$
  $\to (0.707,\ 0,\ 0.707)$ — infatti $0.707^2+0.707^2 = 1$ ✓
- $d_2 = (0.811,\ 0,\ 0)$, $\lVert d_2 \rVert_2 = 0.811$
  $\to (1,\ 0,\ 0)$
- $d_3 = (0,\ 1.099,\ 0.811)$, $\lVert d_3 \rVert_2 = 1.366$
  $\to (0,\ 0.805,\ 0.594)$

**L1** — $v_{\text{norm}} = v / \sum_i |x_i|$:

- $d_1$: somma $=0.81 \to (0.5,\ 0,\ 0.5)$ — le due componenti sono uguali, e ora sommano a 1: si può leggere come "50% peso su gatto, 50% su corre"
- $d_2$: somma $=0.811 \to (1,\ 0,\ 0)$
- $d_3$: somma $=1.91 \to (0,\ 0.575,\ 0.425)$

Confronta $d_1$ nei due casi: L2 dà $(0.707, 0, 0.707)$ (la sfera unitaria), L1 dà $(0.5, 0, 0.5)$ (il rombo unitario) — stessa **direzione** (proporzione 50/50 tra le due parole), ma la lunghezza del vettore è normalizzata in modo diverso. Con L1, le componenti si possono leggere come una distribuzione di probabilità sulle parole del documento.

### 5. Normalizzazione di colonna

Prendiamo ora la colonna `gatto` da sola: $(0.405,\ 0.811,\ 0)$ per $d_1, d_2, d_3$.

**Min-max**: $x' = \dfrac{x - \min}{\max - \min}$, con $\min=0$, $\max=0.811$

- $d_1$: $0.405/0.811 = 0.5$
- $d_2$: $0.811/0.811 = 1$
- $d_3$: $0/0.811 = 0$

**Standardizzazione (z-score)**: media $\mu = 0.405$, deviazione standard $\sigma = 0.331$

- $d_1$: $(0.405-0.405)/0.331 = 0$
- $d_2$: $(0.811-0.405)/0.331 = 1.23$
- $d_3$: $(0-0.405)/0.331 = -1.22$

Qui, a differenza della normalizzazione di riga, ogni colonna viene riscalata **indipendentemente dalle altre colonne**, usando solo i valori che quella parola assume nei vari documenti — è un'operazione "verticale" invece che "orizzontale".

---

### Riassunto concettuale

- **Normalizzazione di riga** → corregge per la **lunghezza del documento**.
- **Normalizzazione di colonna** (IDF, min-max, z-score) → corregge per quanto è **comune/rara/scalata** una parola nel corpus.

Sono due assi ortogonali del problema, spesso combinati: TF-IDF (IDF = normalizzazione di colonna) + L2-norm per riga è esattamente ciò che fa di default `TfidfVectorizer` di scikit-learn.
