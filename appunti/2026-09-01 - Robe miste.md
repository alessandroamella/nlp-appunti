# TF-IDF e normalizzazione — esempio svolto

La teoria (pag. 24–28) è in [[2026-09-01 - Neural Networks and Word Embeddings#1. Recap of postprocessing the BoW/TF-IDF matrix (from Lecture 1, with extra detail)]]; qui resta solo l'esempio numerico.

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
