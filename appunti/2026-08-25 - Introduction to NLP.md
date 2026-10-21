## 1. What is Natural Language Processing?

> "…the processing of **natural language** information by a **computer**. The study of NLP, a subfield of computer science, is generally associated with **artificial intelligence**. NLP is related to **information retrieval**, **knowledge representation**, **computational linguistics**, and more broadly with linguistics." — Wikipedia

Major processing tasks in an NLP system include:

- **Speech recognition**
- **Text classification**
- **Natural language understanding (NLU)**
- **Natural language generation (NLG)** — currently the most prominent/hyped sub-area (e.g., LLM chat systems), largely because most other tasks (classification, translation, etc.) can now be reframed as generation problems.

### Historical framing

- NLP grew out of **computational linguistics** (the scientific study of language).
- It sits at the intersection of linguistics, information retrieval, and machine learning.
- "Artificial intelligence" as an umbrella term has grown/shrunk in popularity over the decades: statistical ML → deep learning → today, everything gets folded under "AI" again, largely driven by the rise of LLMs.

### Three pillars / purposes of NLP

|Pillar|Description|Examples|
|---|---|---|
|**Enabling human–machine collaboration**|Systems that let humans interact with/access machine knowledge|Search engines (Google, Bing), dialogue agents (Siri), text generation (ChatGPT)|
|**Accelerating human–human communication**|Systems that help humans communicate across language/other barriers|Machine translation (Google Translate, DeepL), text summarization, information extraction|
|**Mining human insights**|Extracting latent signals about people from text|Sentiment analysis, motivation analysis, emotion detection (e.g., Luminoso)|

### Example applications discussed

- **Machine translation**: quality (BLEU score) has improved drastically since ~2019–2020, especially into English; still weaker between two _low-resource_ language pairs.
    - **Low- vs. high-resource language**: defined by amount of available data/speakers/researchers/Wikipedia articles for that language. Translating low→low resource is harder than low→high→low (pivoting through English/high-resource languages), though pivoting isn't ideal since it loses cultural/contextual nuance.
- **Conversational systems / voice assistants**: Siri, Alexa, Google Home — an older but foundational NLP application category.
- **Question answering**: search engines evolved from "10 blue links" → direct AI-generated answer overviews (e.g., Google's AI Overview).
- **Text generation & agents**: modern LLMs (ChatGPT, Claude) can generate full documents and plan/orchestrate multi-step tasks (agentic workflows). The live demo was Claude Code designing a whole course: the lecturer described it as "an orchestrator or an agentic harness on top of the models", with sub-agents working on different parts of the problem in parallel.
    - Caution highlighted: over-relying on LLMs for tasks like curriculum design risks **loss of diversity**. If everyone asks the same thing, everyone gets very similar answers, so if the answer misses some fine points, everyone misses them. The lecturer's image: it's like repeatedly sampling from a skewed distribution, where you keep getting the same region and miss the other parts of the distribution.

### Why language models got so good

- Model sizes have grown roughly **~5000× from 2018 to 2022** (ELMo ~94M params → Megatron-Turing NLG 530B params).
- For comparison, the human cerebral cortex has ~1000× more neurons than a mouse's.
- This scale-up (plus algorithmic/hardware advances) is the key driver behind modern disruptive NLP applications.

---

## 2. Classical (Pre-Neural) NLP: Core Tasks

All four tasks below reduce, in classical NLP, to **representing documents as feature vectors** and applying standard ML.

### Task 1 — Document retrieval

- **Given**: a document collection (corpus) + a query document (can be a short string).
- **Task**: rank all documents in the collection by similarity to the query.
- Core task behind web search engines ("10 blue links").
- **Approach**: nearest-neighbor search (like kNN) — define a distance function between documents, return the $k$ closest to the query.
- **Hard part**: designing/learning a good distance function, and scaling it to the size of the Web.

### Task 2 — Document classification

- **Given**: a document $d$ and a set of classes (e.g., news, sports, tech, music, romance).
- **Task**: decide which class $d$ belongs to.
- Example: finding gangster movies on IMDb.
- **Approach**: supervised learning — label documents, represent as feature vectors, train a classifier (kNN, logistic regression, decision tree, random forest, boosted trees, neural net, …).

> [!example] Lab — "Is this page from Frankenstein?"
> `ta_sessions/Exercises/01 - Intro to Tools/2. Handling Text - Part 1.ipynb`, Task 2. Four novels are cut into 50-sentence chunks, turned into a raw **BoW** matrix (`CountVectorizer`), and a **regularized logistic regression** predicts Frankenstein vs. the rest.
> - **581 chunks × 23,116 features**: far more features than samples. Without regularization the model could fit the training set perfectly with meaningless weights. The strength $C$ is picked by 10-fold cross-validation *on the training set only*, and the test accuracy is 1.0.
> - **BoW is interpretable**: each weight belongs to a word, so you can read off what characterises the class. Among the top-20 Frankenstein features are *misery, justine, ice, father*, but also *my, me, which, and, of*. Those are **function words**, the ones a stopword list would delete. Frankenstein is told in the first person, which would explain the pronouns (my reading, the notebook doesn't say why). This is the author-identification point from the stopword section below, seen in real data.

> [!example] Lab — 78% accuracy that means nothing (majority baseline)
> `ta_sessions/Exercises/01 - Intro to Tools/3. Handling Text - Part 2.ipynb`, Task D (Big Bang Theory scripts). Sheldon claims he can tell whether a line is his with >70% accuracy. The solution builds the words that *only* Sheldon says in seasons 1–8, labels a test line (seasons 9–10) "Sheldon" if it contains one of them, and gets **78.3%**. It presents that as confirming the claim.
> ⚠️ It doesn't. Only 2,237 of the 10,938 test lines are Sheldon's, so the trivial classifier "never Sheldon" scores **79.5%**, better than the rule. With **imbalanced classes**, compare accuracy against the **majority-class baseline** before calling it good. (I computed the baseline from `data/all_scripts.txt`; the notebook doesn't mention it.)

### Task 3 — Sentiment analysis

- **Given**: a document $d$ (e.g., a product review).
- **Task**: produce a "sentiment" score capturing how positive/negative $d$ is.
- Use cases: inferring opinion without explicit ratings; tracking historical opinion shifts (e.g., toward politicians over time).
- **Approach**: same supervised setup as classification, but can be framed as regression (continuous score) or classification (positive/negative/neutral).

> [!example] Lab — lexicon-based sentiment with VADER
> `ta_sessions/Exercises/01 - Intro to Tools/2. Handling Text - Part 1.ipynb`, Task 1. The lab doesn't train anything here: **VADER** is a hand-built sentiment lexicon plus rules, meant for short social-media text. For each sentence it gives `neg`, `neu`, `pos` (they sum to 1) and a `compound` score in $[-1, 1]$. The label comes from thresholding compound at $\pm 0.05$.
> On Pride and Prejudice, most sentences come out **neutral** (3,595 neutral, 3,169 positive, 2,286 negative). The notebook's point is that novels are mostly neutral, while social-media opinions tend to be **polarised** (bimodal). So the same tool behaves differently depending on the domain.

### Task 4 — Topic detection

- **Given**: an _unlabeled_ document collection.
- **Task**: discover a set of prevalent topics and assign documents to them.
- Use cases: trending topics on social media, distinct viewpoints on a political issue, exploratory analysis of a large corpus.
- **Approach**: clustering — represent docs as feature vectors, run hierarchical (agglomerative/divisive) or point-assignment (k-means, DBSCAN) clustering. Alternative: matrix factorization.

> [!example] Lab — topic detection with LDA on four novels
> `ta_sessions/Exercises/01 - Intro to Tools/2. Handling Text - Part 1.ipynb`, Task 3. The model is **LDA** (Latent Dirichlet Allocation), a probabilistic topic model. It is *not* in the lecture, which names clustering and matrix factorization (for the latter, LSA, see [[2026-09-01 - Neural Networks and Word Embeddings]]). LDA gives each document a **probability distribution over topics**, whereas LSA gives it real-valued coordinates in a latent space.
> - The preprocessing uses nearly the whole toolbox of §3: lemmatize, keep alphabetic tokens only, drop stopwords and tokens of ≤2 characters, add multi-token named entities and frequent bigrams (`Phrases`, the "n-grams as features" idea), then drop words in >50% of chunks or in <5 chunks (the document-frequency heuristic from the stopword section).
> - With 4 topics for 4 books, **Moby-Dick** (*whale, sea, ship, Queequeg*) and **Pride and Prejudice** (*Elizabeth, Darcy, Bingley*) are recovered at 100%. Dracula and Frankenstein only reach ~55%. The topics mostly pick up **character names**. Topics found without supervision are not necessarily the ones you had in mind.

---

## 3. From Text to Feature Vectors

Nearly all ML methods need fixed-length numeric feature vectors — but raw text is variable-length, and even fixed-length text (e.g. tweets) doesn't have positions that correspond to meaningful features (unlike, say, image pixels).

Two broad strategies:

1. **Traditional/vetted**: Bag of Words (BoW).
2. **More recent**: _learn_ a mapping from strings → vectors ("text embeddings").

### Bag of Words (BoW)

- "Bag" = **multiset**: keep word multiplicities, discard word order.
- Example: "what you see is what you get" → `{get:1, is:1, see:1, what:2, you:2}`
- To get a fixed-length representation: build a vector with one entry per unique vocabulary word.
    - Example vector: `[0…0 1 0…0 1 0…0 1 0…0 2 0…0 2 0…0]`
- These vectors are **very high-dimensional** (typically $10^5$–$10^6$) and **very sparse**.

**Zipf's Law** (extra reason for sparsity): word frequency is inversely proportional to its frequency rank:

$$p(w_i) \propto \frac{1}{i}$$

where $w_i$ is the $i$-th most frequent word. A small number of words (e.g., "and", "the") dominate frequency counts, while most vocabulary words are rare.

> [!example] Lab — plotting Zipf
> `ta_sessions/Exercises/01 - Intro to Tools/3. Handling Text - Part 2.ipynb`, Q4. A histogram of how often each word occurs across all the Big Bang Theory scripts is useless on linear axes: one huge bar at "rare", and a long invisible tail. On **log-log axes** (with log-spaced bins) it becomes roughly a straight line, the signature of a **power law**. A power law is a straight line in log-log because $\log p = -\log i + \text{const}$.

### The Bag-of-Words matrix

- Rows = documents, columns = vocabulary words.
- This matrix is **huge**: e.g., Wikipedia ≈ 6M docs × 2M words → 12 trillion entries.
- **Solution**: sparse matrix representation — store only non-zero entries as triples `(doc_idx, word_idx, count)`. E.g., Wikipedia with ~2000 words/article on average → ~12 billion non-zero entries (fits in memory).
- Once in matrix form, any standard ML model can be applied.

### "Garbage in, garbage out" — preprocessing pitfalls

Converting raw text → BoW matrix requires care at several steps:

1. **Character encoding**
    
    - Mapping from abstract characters to bytes.
    - Old school: ASCII, Latin-1. New school: Unicode (UTF-8, UTF-16, UTF-32).
    - Must read files with the _same_ encoding used to write them — critical for non-English text (à, ê, ü, ć, …).
    - When writing files: always use UTF-8/UTF-16 and hard-code the output format.
2. **Language identification**
    
    - Content is increasingly multilingual (Twitter, Wikipedia).
    - Ideally the language is already tagged (HTML headers, API metadata) — but not always.
    - Good libraries exist (Python, Java); most commonly based on **letter trigram** statistics (e.g., "eau", "ghi", "ijs", "sch", "eiß", "ção", "jeg").
    - Much harder if character encoding was already mishandled.
3. **Tokenization**
    
    - Splitting text into discrete units (tokens).
    - Relatively easy for English; the "optimal" tokenizer differs per language (e.g., Swedish "Sankt Peter" → "S:t Peter").
    - Hard for languages without whitespace between words (e.g., Chinese).
    - Compound words are tricky (German, Sanskrit): e.g., "Donaudampfschifffahrtskapitän" → "Donau dampf schiff fahrts kapitän" — but should "Schiff fahrt" be one token ("Schifffahrt") or two?
4. **Stopword removal**
    
    - Very frequent, low-information "small" words (a, the, is, you, I, punctuation) can drown out informative content words.
    - Many stopword lists exist online, but be careful:
        - Different tasks need different stopword sets.
        - Heuristic: remove words appearing in ≥ $p%$ of documents (choosing $p$ is non-trivial).
        - Sometimes stopword removal _hurts_ — e.g., author identification and psychological modeling rely on function words and punctuation (pronoun usage, "!!!", ":-)" carry signal). Reference: James Pennebaker, _The Secret Life of Pronouns_.
5. **Word normalization**
    
    - **Casefolding**: lowercase everything so "Yams"/"yams" aren't separate features. But this creates ambiguity (e.g., "apple" the fruit vs. "Apple" the company). Its purpose was to **shrink the vocabulary** and so the sparsity of BoW, which mattered when every word was a column. The lecturer: *"these days, case folding is almost always not done"*, because modern tokenizers keep the vocabulary small anyway (with characters as tokens it's ~100 symbols), so there's no reason to throw away the case information.
    - **Stemming**: crude, rule-based stripping of affixes to a common root (e.g., "walking"/"walks"/"walked" → "walk"; "business"/"busy" → "busi"). Typically done heuristically (e.g., Porter stemmer).
        - Pro: reduces sparsity.
        - Con: discards information/introduces collisions (e.g., "business" vs. "busy"; "operating" as in "operating system").
        - Rarely used for English anymore (especially with big data); still useful for morphologically rich languages (German, Finnish, Sanskrit).
    - **Lemmatization**: "stemming++" — maps tokens to proper lexicon (dictionary) entries using grammatical context, e.g., "U.S.A."/"US" → "United States"; "Grüße"/"Gruesse" → "Grüße"; distinguishing "You **lie** in the grass" vs. "You **lie** to me". Requires a full lexicon + complex mapping rules, so it's frequently omitted; especially hard for non-English.
6. **Social media text is extra messy**
    
    - Example real tweet: _"ikr smh he asked fir yo last name so he can add u on fb lololol"_
    - Full of abbreviations (ikr = "I know, right?", smh = "shake my head", fb = Facebook, yo = "your"), misspellings (fir = "for"), and repeated letters/syllables ("yeahhh", "hahahaha").
    - Standard NLP tools struggle here — dedicated toolkits like **TweetNLP** are needed.
7. **Tokens vs. n-grams**
    
    - Plain BoW uses unigrams (single tokens) as columns.
    - Multi-word expressions carry meaning as a unit ("United States", "operating system").
    - Brute-force fix: include all bigrams ($n=2$), trigrams ($n=3$), etc., as extra features.
    - Using all 5-grams can even outperform neural nets on some tasks.
    - Problem: combinatorial explosion of the feature space as $n$ grows.

> [!example] Lab — the preprocessing pipeline in spaCy
> `ta_sessions/Exercises/01 - Intro to Tools/2. Handling Text - Part 1.ipynb`. It runs spaCy's pipeline (sentence splitting → tokenization → POS tagging → NER → stopwords → lemmatization → noun chunks → dependency parsing) on one sentence from Frankenstein. The parts the lecture list doesn't show:
> - **Sentence splitting is not trivial**: a period doesn't always end a sentence ("U.K.", "Mr. Bennet", "Yahoo!"). spaCy uses a **statistical model** for it, not a rule.
> - **Tokenization is split on whitespace + language-specific rules**: "don't" → `do` + `n't`, while "U.K." stays one token and "but," becomes `but` + `,`.
> - **Lemmatization** maps *am → be*, *streets → street*, *me → I*. Stemming could never do *am → be*, because there's no shared affix to strip.
> - **Tools make mistakes**: NER tags *London* correctly as a place (GPE), but tags *Petersburgh* as a PERSON.
> - spaCy's default stopword list has **326** entries and removes *already* and *upon* as well as *the*. Which list you use is a real design choice, as the lecture warns.
>
> The lab's comprehension MCQ on this: stopwords `[a, the, is, you, I, my, with, was]` applied to *"I was driving down the street with my CAR! :)"* give `{driving, down, street, car, :)}`. The answer is **stopword removal + casefolding** only. *CAR → car* is casefolding, and *driving* would have become *drive* under stemming, so there was **no stemming**.

### Postprocessing the BoW matrix: TF-IDF

Not all words are equally informative — beyond discarding stopwords, we want to down-weight generically common words (e.g., "per") relative to rarer, more informative ones (e.g., "perceptron").

**Inverse Document Frequency (IDF)**:

$$\text{idf}(w) = -\log!\left(\frac{\text{docfreq}(w)}{N}\right) = \log(N) - \log(\text{docfreq}(w))$$

where:

- $\text{docfreq}(w)$ = number of documents containing word $w$
- $N$ = total number of documents in the corpus

**Term Frequency (TF)**: $\text{tf}(d, w)$ = frequency of word $w$ in document $d$ — this is exactly what the raw BoW count captures.

**TF-IDF matrix**: entry for document $d$, word $w$:

$$\text{tfidf}(d, w) = \text{tf}(d, w) \times \text{idf}(w)$$

This is equivalent to multiplying each column $w$ of the BoW matrix by the constant $\text{idf}(w)$ — i.e., a form of **column normalization**.

**Row normalization**: longer documents have more non-zero entries → longer vectors → can throw off ML algorithms that rely on vector distance (long vectors end up far from short ones even if topically similar). Fix: normalize each document's row vector.

- **L2-normalization**: rescale each row to Euclidean length 1 (data points lie on a unit hypersphere).
- **L1-normalization**: rescale each row to sum to 1 (interpretable as a probability distribution).
- No universal rule for which is better — task-dependent.

**Other column normalization options** (beyond IDF-scaling):

- Min-max scaling
- Standardization (subtract mean, divide by standard deviation)

> [!example] Lab MCQ — more features than documents
> From `ta_sessions/Exercises/01 - Intro to Tools/2. Handling Text - Part 1.ipynb`, Part 2 Q3: which does **not** address having more features than documents in a TF-IDF matrix? **Min-max scaling.** Rescaling a column changes its range, not the *number* of columns. The problem is still there. The fixes either reduce the number of features (**SVD**/LSA, **feature selection**) or constrain the weights (**regularization**, as in the Frankenstein classifier above).

**Summary — "Classical NLP" recipe**: document retrieval, classification, sentiment analysis, and topic detection can all be tackled by building a **TF-IDF matrix** (entry = $\text{tf}(d,w) \times \text{idf}(w)$) and feeding it into a standard ML algorithm.

---

## 4. Transition to Modern (Neural) NLP

Setting up a "neural" NLP problem requires answering three questions:

1. **Embeddings**: how do we represent sequences of discrete words as vectors?
2. **Model**: how do we compose/combine these embeddings into higher-level (sequence-level) representations?
3. **Prediction**: how do we map the model's representation to a task-relevant output/prediction?

### Dense word vectors (embeddings)

- Each word is represented as a **high-dimensional** (though much lower than sparse BoW — still $O(10^3)$), **real-valued, dense** vector.
- Aliases: _word vectors_, _word embeddings_, _neural embeddings_, _dense embeddings_.
- Key property: **vector similarity ≈ meaning similarity** — words with related meaning end up close together in embedding space.

### Building a simple NLP model — worked example (sentiment analysis)

Goal: map a sentence like _"I really enjoyed the movie we watched on Saturday!"_ to a sentiment label (+/−).

1. For a sequence $S$ of $T{+}1$ tokens, build the corresponding sequence of embeddings: $$X = {x_0, x_1, \ldots, x_T}$$ Each $x_t$ is looked up from a **shared embedding dictionary/table** $E$ indexed over the whole vocabulary $V$ — identical words (even across different sentences) index the _same_ embedding in $E$.
    
2. **Model** step — modify/compose the embeddings into a representation useful for prediction. Candidate architectures (covered later in the course):
    
    - Recurrent Neural Networks (RNNs) and variants (LSTM, GRU) — Week 3
    - Self-attention & Transformers — Week 4/5
    - State-space models (SSMs) — mentioned but **not covered** in this course
    - Or something very simple: **sum-pool**, **average-pool**, **max-pool**
    
    **Worked toy example — sum-pooling**: $$h_t = \sum_{t=0}^{T} x_t$$ i.e., just sum all the word embeddings in the sentence to get a single fixed-length vector $h$ representing the whole sentence.
    
3. **Prediction** step — use $h_T$ (the pooled representation) as input features to a classification algorithm, e.g., **logistic regression**: $$P(y) = \sigma(W_0 h_T)$$ where $\sigma$ is the sigmoid function and $W_0$ is a learned weight matrix.
    
4. **Learning**: the whole pipeline (embeddings $E$ and classifier weights $W_0$) is learned jointly via **backpropagation** — computing gradients of the loss with respect to the initial embeddings $X$ (and all other parameters) so that the embeddings end up encoding whatever information is useful for solving the task.
    
    → **Learnable parameters in this toy system**: the embedding table $E$, and the logistic regression matrix $W_0$.
    
    (Discussion aside: for the sentence "I enjoyed the movie I watched on Saturday!" there are 7 _unique_ embeddings needed, since some words repeat and share the same embedding row.)
    

### Beyond single-label classification: other task setups using the same embeddings→model pipeline

- **Sequence labeling**: instead of one label for the whole sentence, produce a label _per token_ — e.g., tagging which words in a review are the "sentimental" ones. Requires a model (RNN, self-attention, Transformer) that produces a per-token hidden state $h_0, h_1, \ldots, h_T$, each independently mapped to a label (+/−).
- **Text generation**: instead of a classification label, predict the next token/sentence given everything seen so far (e.g., generating the next sentence of a review) — the foundation of modern generative language models.

### Three parts of a "Neural" NLP system (recap)

1. **Embeddings** — how to represent discrete words as vectors.
2. **Model** — how to compose embeddings into sequence-level semantic representations.
3. **Prediction** — how to map the model's representation to a task-specific output.

➡️ The course's next step: focus in depth on **embeddings** (word representations), setting up subsequent lectures on neural network basics, backpropagation, and word embedding methods (e.g., Word2Vec).

---

## Da ricordare

- **BoW** = multiset of words: counts kept, **order lost**. Vectors are huge ($10^5$–$10^6$) and **sparse**, partly because of **Zipf** ($p(w_i) \propto 1/i$, a straight line on **log-log** axes).
- **IDF** $= \log N - \log \text{docfreq}(w)$. **TF-IDF** $= \text{tf}(d,w)\cdot\text{idf}(w)$ is **column** scaling. L1/L2 normalization of each document is **row** normalization, which fixes the effect of document length.
- **Stemming** strips affixes by rule (*busy/business → busi*), while **lemmatization** maps to a dictionary form using context (*am → be*). If an output keeps *driving*, there was no stemming. *CAR → car* is casefolding.
- **Stopword removal can hurt**: function words and punctuation carry the signal in authorship and psychological tasks. In the lab, *my/me/which/and* are among the top features for Frankenstein.
- **More features than documents** → SVD, feature selection, regularization. ⚠️ **Not** min-max scaling (it doesn't change the dimensionality).
- Always compare accuracy with the **majority-class baseline** on imbalanced data (the Sheldon classifier: 78% vs 79.5% for "never Sheldon").
- A neural NLP system has three parts: **embeddings → model (composition) → prediction**. In the toy sentiment model the learnable parameters are the embedding table $E$ and $W_0$.
