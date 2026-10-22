## 1. Recap of postprocessing the BoW/TF-IDF matrix (from Lecture 1, with extra detail)

Quick refresher, since this lecture builds directly on it:

- **Row normalization** of the TF-IDF matrix: rescale each _document_ vector so that document length doesn't distort distance-based ML algorithms (a long document naturally has more non-zero entries → a longer vector → looks "farther" from short documents even if topically identical).
    - **L2-normalization**: rescale each row vector so its Euclidean norm is 1 → all rows land on the **unit circle/hypersphere** ($x_1^2 + x_2^2 = 1$ in 2D).
        - Example: original vector $(-1.2, 0.9)$ → L2-normalized to $(-0.8, 0.6)$.
    - **L1-normalization**: rescale each row so its entries sum to 1 (in absolute value) → rows land on the **unit diamond** ($|x_1| + |x_2| = 1$ in 2D), interpretable as a probability distribution.
        - Example: original vector $(-1.2, 0.4)$ → L1-normalized to $(-0.75, 0.25)$.
    - There's no universal answer for which is "better" — it's an empirical, task-dependent choice (see below).
- **Column normalization**: rescaling each _word_ column instead of each document row.
    - IDF-weighting itself is a form of column normalization (it multiplies every entry in column $w$ by the constant $\text{idf}(w)$).
    - Other options: **min-max scaling** (rescale each column using its min/max values) or **standardization** (subtract the column mean, divide by the column standard deviation) — both standard tricks from basic ML/linear algebra.

**How do you choose between normalization schemes (row-wise, column-wise, or none)?** There is no principled a priori answer — general ML practice: split data into training/validation sets, try different preprocessing choices, and empirically measure which one gives better downstream task performance (accuracy, etc.) on the validation set. This is the same logic used to tune any hyperparameter.

---

## 2. Today's focus: Modern (Neural) NLP

Three parts of a "neural" NLP system (introduced conceptually in Lecture 1, deep dive starts here):

1. **Embeddings** — how do we represent discrete words as vectors?
2. **Model** — how do we compose embeddings into higher-level, sequence-level representations?
3. **Prediction** — how do we map the model's representation to a task-relevant output?

> **This lecture's focus is almost entirely on part 1: embeddings.**

### Why move beyond sparse (BoW/TF-IDF) representations?

Historically, sequences were represented via a composition function over **sparse** vectors — either simple counts or counts reweighted by corpus statistics (TF-IDF). Example:

```
"I really enjoyed the movie!" → [1 … 1 1 0 1 … 0 1]              (simple counts)
"I really enjoyed the movie!" → [0.01 … 0.1 0.1 0 0.001 … 0 0.5] (TF-IDF weighted)
```

**The core problem**: with sparse vectors, similarity between two pieces of text is purely a function of _shared words_. Any distance/similarity function (cosine, Euclidean, Jaccard, …) computed over these vectors will only ever pick up on literal word overlap — never on _meaning_.

**Concrete example** — compare the words "enjoyed" and "loved":

$$ \text{enjoyed} \to [0 \ldots 0\ 0\ 0\ 1\ \ldots\ 0\ 0] $$ $$ \text{loved} \to [0 \ldots 1\ \ldots\ 0\ 0\ 0\ 0\ 0] $$ $$ \text{sim}(\text{enjoyed}, \text{loved}) = 0 $$

No matter what transformation you apply on top (TF-IDF, normalization, etc.), the similarity between two _different_ words will always be exactly 0 in a sparse bag-of-words space, because they occupy different, non-overlapping dimensions — even though "I really enjoyed the movie" and "I really loved the movie" are near-identical in meaning to a human reader.

**What we actually want**: to capture _semantic_ relationships between words — e.g., the relationship between "man" and "woman" should be analogous to that between "king" and "queen"; "walking" relates to "walked" the same way "swimming" relates to "swam"; "country" relates to "capital" consistently across many country–capital pairs (Canada–Ottawa, Turkey–Ankara, Vietnam–Hanoi, Japan–Tokyo, etc.). These are examples of **word analogies** that emerge naturally once words are placed in a well-trained vector space — this is the key goal of neural word embeddings.

![[appunti/assets/2026-09-01/nn-primer-036.png]]
_Male–Female, verb-tense, and country–capital relationships all show up as **consistent, parallel directions/offsets** in a well-trained embedding space — the same "arrow" (e.g., man→woman) works for king→queen too._

> [!example] Lab — analogies, and the bias they expose
> `ta_sessions/Exercises/03 - Language Modeling/0. Word Embeddings.ipynb`, §2. "*a* is to *b* as *c* is to ?" is computed as $b - a + c$ and then the nearest word by cosine: in gensim, `most_similar(positive=[b, c], negative=[a])`. On GloVe, *king − man + woman* → *queen*.
>
> The same arithmetic surfaces stereotypes: *man → doctor :: woman → ?* returns **nurse** ahead of *physician*; *man → boss :: woman → ?* returns relationship words (*girlfriend*, *lover*). Nobody programmed this in: the vectors encode the co-occurrence statistics of how people wrote. You can see it without analogies: *doctor* is about equally close to *man* and *woman* (0.61 vs 0.63), *nurse* is not (0.46 vs 0.61). Used as features, e.g. for CV screening, the association is inherited silently.
>
> **Caveat the lab points out:** `most_similar` never returns one of its input words. Rank *all* words and $\text{doctor} - \text{man} + \text{woman}$ is closest to **doctor itself**. So analogy tests **exaggerate** the bias somewhat, and they also flatter the analogy property in general, since the trivial answer is excluded. The bias is still real (*nurse* above *physician* isn't an artefact). The course comes back to this in the Responsible AI session.

### Dense word vectors (recap + goal)

- Each word → a **dense, real-valued vector**, still high-dimensional relative to a scalar, but _much_ smaller than sparse BoW vectors: typically $O(10^3)$ dimensions instead of $O(10^5)$–$O(10^6)$.
    - Aside from the lecturer: some of today's large models have an embedding dimensionality of **8,192**, or around 16,000 (always powers of 2). The lecturer wouldn't call that low-dimensional, but it's nowhere near the millions of dimensions of a sparse BoW vocabulary vector, and it isn't sparse.
- **Similarity of vectors should represent similarity of meaning.** A well-trained embedding space gives $\text{sim}(\text{enjoyed}, \text{loved}) \gg 0$, unlike the sparse case.

> [!example] Lab — cosine similarity, and what it actually measures
> `ta_sessions/Exercises/03 - Language Modeling/0. Word Embeddings.ipynb`, §1. The "sim" above is almost always **cosine similarity**:
>
> $$\cos(\mathbf u, \mathbf v) = \frac{\mathbf u \cdot \mathbf v}{\lVert \mathbf u\rVert\,\lVert \mathbf v\rVert} \in [-1, 1]$$
>
> where $\mathbf u, \mathbf v$ are the two word embeddings being compared (where $\mathbf u \cdot \mathbf v = \sum_i u_i v_i$, $\lVert \mathbf u\rVert\,\lVert \mathbf v\rVert = \sqrt{\sum_i u_i^2}\,\sqrt{\sum_i v_i^2}$).
>
> It compares **direction and ignores magnitude** (length reflects things like word frequency or document length, not meaning: a document pasted twice gives $2\mathbf v$, cosine 1 with $\mathbf v$), which is why it's preferred over Euclidean distance (the lab's MCQ 4.8 trap: it is *not* restricted to $[0,1]$, and it doesn't require pre-normalised vectors).
>
> **A refinement of "similarity of meaning"**: on the lab's own skip-gram model, $\cos(\text{man}, \text{woman}) \approx 0.78$ beats $\cos(\text{jury}, \text{court}) \approx 0.58$. Same pattern in GloVe, where *good–bad* = 0.77 and *cold* is the 2nd-nearest neighbour of *hot*. **Antonyms end up close**, because they occur in the same contexts (*the ___ said*, *it was ___*), and the distributional hypothesis (§5) only sees context. So embedding cosine measures **relatedness of usage, not synonymy**. Also, even unrelated pairs score clearly above 0 (*jury–car* ≈ 0.21): word vectors share a common overall direction, so compare cosines **with each other**, not against an absolute scale.

---

## 3. The "simple NLP model" recap, with new detail

For sentiment analysis on _"I really enjoyed the movie we watched on Saturday!"_:

1. Convert the sentence $S$ into a sequence of embeddings $X = {x_0, x_1, \ldots, x_T}$, where each $x_t$ is looked up from a **shared embedding dictionary** $E$ — every unique word in the vocabulary has exactly one row/entry in $E$, no matter how many times or where it appears in a sentence.
2. Feed $X$ into a **model** that composes/transforms the embeddings into a useful representation. Candidate model families (covered in coming weeks): RNNs/LSTM/GRU (Week 3), self-attention/Transformers (Week 4) — or, as a simple baseline, **sum-pool / avg-pool / max-pool**.
    - **Sum-pool example**: $h_t = \sum_{t=0}^{T} x_t$ — literally just add up all the word embeddings in the sentence into one fixed-length vector.

**Worked numeric example — sum/avg/max-pool side by side**: take the sentence _"I enjoyed the movie"_ ($T=4$ words) with toy embeddings of dimension $d=3$ (in practice $d$ is $O(10^3)$, kept tiny here so the arithmetic is visible):

| word | embedding $x_t$ |
|---|---|
| I | (0.1, 0.0, 0.2) |
| enjoyed | (0.9, 0.8, 0.1) |
| the | (0.0, 0.1, 0.0) |
| movie | (0.7, 0.6, 0.3) |

- **Sum-pool** — add component-wise:
    $$h_T = (0.1+0.9+0.0+0.7,\ 0.0+0.8+0.1+0.6,\ 0.2+0.1+0.0+0.3) = (1.7,\ 1.5,\ 0.6)$$
- **Avg-pool** — same as sum-pool, then divide by $T=4$ (normalizes away sentence length, so a 4-word and a 40-word sentence with similar content land in a similar range):
    $$h_T = (1.7/4,\ 1.5/4,\ 0.6/4) = (0.425,\ 0.375,\ 0.15)$$
- **Max-pool** — take the element-wise maximum across all word vectors (each output dimension independently picks whichever word "activates" it most):
    $$h_T = (\max(0.1,0.9,0.0,0.7),\ \max(0.0,0.8,0.1,0.6),\ \max(0.2,0.1,0.0,0.3)) = (0.9,\ 0.8,\ 0.3)$$

All three always produce a **fixed-length vector** (dimension 3 here, regardless of $T$) — this is what makes them usable as input to a fixed-size classifier $W_0$. Notice all three variants completely discard word order: shuffling the sentence to "movie enjoyed the I" gives the exact same $h_T$ in every case — a key weakness that motivates RNNs/self-attention (Weeks 3–4), which preserve positional/sequential information.
3. Pass the pooled representation $h_T$ into a **classifier**, e.g. logistic regression: $P(y) = \sigma(W_0 h_T)$.
    - **Logistic regression** isn't introduced anywhere in the course; the lecturer treats it as known from a basic ML class. It is the simplest binary classifier there is, and here it does two things. First, $W_0 h_T$ multiplies the vector by a weight matrix. For a binary task, $W_0$ has a single row, so the result is **one real number** (a score). Second, the **sigmoid** $\sigma(z) = \frac{1}{1+e^{-z}}$ squashes that number into $(0, 1)$, which is read as $P(+)$. A large positive score means "almost surely positive", a large negative score means "almost surely negative", and a score of 0 gives 0.5. Despite the name, it's a classifier, not a regression: "regression" refers to the linear score inside. (see [1:31:55](https://au.cloud.panopto.eu/Panopto/Pages/Viewer.aspx?id=e32acb17-9550-4cdc-adea-b4b80077ea46&start=5515), where he says logistic regression "is just applying the sigmoid activation function")
    - Seen as a neural network, this is **one linear layer plus a sigmoid**, the smallest network possible. The building blocks are in [[#Neural-network building blocks (lab only — not in the deck)|the section below]]. With more than two classes, the sigmoid becomes a **softmax** over a $W_0$ with one row per class ("multi-class logistic regression", which comes up again below).
4. **Learnable parameters**: the embedding table $E$ and the classifier weight matrix $W_0$ — both trained jointly via **backpropagation**, by computing gradients of the loss with respect to the initial embeddings (and all other parameters) so the embeddings end up encoding whatever helps solve the task.
    - **What that means in practice**: you have labeled training data (e.g., movie reviews labeled positive/negative). You initialize $E$ and $W_0$ randomly and run the forward pass (look up → pool → logistic regression) to get a predicted probability. Then you compute a **loss** measuring how far that prediction is from the true label (e.g., predicted 0.1 but the true label is 1). Finally, backprop computes how much each parameter contributed to that error, and each one is nudged so that the next prediction is a bit closer to the truth. Repeated over many examples, this is how the embeddings and the classifier are learned together. Backprop itself is covered properly in [[2026-09-15 - Language Modeling (Part 2)]] §2, and the NN basics in the lab section below.

> ⚠️ **Exam (2025 paper / mock exam, Q1):** the *primary purpose* of backprop is to compute **gradients of the loss with respect to the network's parameters**. Not w.r.t. the input data (that's rare, e.g. adversarial attacks), not to adjust the learning rate (that's the optimizer), and not the vague "propagate errors backward to refine predictions".

> [!example] Lab — this exact model on 50k tweets, and where it breaks
> `ta_sessions/Exercises/03 - Language Modeling/0. Word Embeddings.ipynb`, §3. The lab builds the pipeline above with one difference: the embeddings are **pre-trained GloVe vectors, frozen**, not learned jointly. Tweet → **avg-pool** of the known words' vectors (zero vector if none is known) → `Linear(100, 256)` → ReLU → `Linear(256, 1)` → sigmoid, trained with binary cross-entropy on Sentiment140. It levels off at **≈ 71% test accuracy**: a decent baseline for so little.
>
> Then the killer test:
>
> | sentence | $P(\text{positive})$ |
> |---|---|
> | *the movie was good, not bad at all* | 0.368 |
> | *the movie was bad, not good at all* | 0.368 |
>
> Same eight words, different order, so **the same average, so necessarily the same score**. The first is misclassified, and no amount of training can fix it: the information is gone before the classifier sees the input. This is the word-order weakness stated above, made concrete. Fixes, in increasing power: **n-gram features** (*not bad* ≠ *not good* as bigrams), an **RNN/LSTM** reading left to right ([[2026-09-08 - Language Modeling (Part 1)]], [[2026-09-15 - Language Modeling (Part 2)]]), a **transformer** with self-attention and positions ([[2026-09-22 - Seq2Seq Models and Transformers]]).

**Worked example — counting unique embeddings**: for the sentence _"I enjoyed the movie I watched on Saturday!"_, there are only **7 unique embeddings** needed in $X$, because the word "I" appears twice and shares the same row in the embedding table $E$ both times (not 9, which is the raw token count).

### Neural-network building blocks (lab only — not in the deck)

Despite its file name, the "Neural network primer" deck is almost entirely about embeddings. The NN basics themselves are only in the lab, `ta_sessions/Exercises/02 - NN basics/0. MLP and Training.ipynb`, which builds each piece by hand and checks it against `torch.nn`:

| piece | definition | what to remember |
|---|---|---|
| **ReLU** | $\max(0, x)$ | derivative 0 for $x<0$, 1 for $x>0$; undefined at exactly 0 (PyTorch picks 0, harmless in practice) |
| **Sigmoid** | $S(x) = \frac{1}{1+e^{-x}} \in (0,1)$ | $S'(x) = S(x)(1-S(x)) \le 0.25$, and → 0 in both tails: the root of the **vanishing gradient** ([[2026-09-15 - Language Modeling (Part 2)]] §6) |
| **Perceptron** | $y = \phi(\mathbf w\cdot\mathbf x + b)$ | a single one can only draw a **linear decision boundary** |
| **Linear layer** | $\mathbf y = W\mathbf x + \mathbf b$, $W \in \mathbb R^{m\times d}$ | $m$ perceptrons at once, one per **row**; shape is **(out × in)**, as in `nn.Linear`. Purely affine, no activation |
| **MLP** | Linear → nonlinearity → Linear → … | the **nonlinearity is what makes depth useful**: without it, stacked linear layers collapse into a single linear map |
| **MSE** | $\frac1N\sum_i(\hat y_i - y_i)^2$ | $\partial\mathcal L/\partial\hat y_i = \frac2N(\hat y_i - y_i)$ |

One training iteration is always, in this order: **forward pass → compute loss → backward pass (`loss.backward()`, after `optimizer.zero_grad()`) → update weights (`optimizer.step()`)**. At evaluation: `model.eval()` and `torch.no_grad()`, since no gradients are needed. The lab trains a 784 → 512 → 10 MLP on FashionMNIST with cross-entropy and SGD; the network outputs **logits**, and `nn.CrossEntropyLoss` applies the softmax (§6) internally.

### Beyond single-label classification

- **Sequence labeling**: instead of one label for the whole sentence, produce a label _per word_ — e.g., tag which specific words carry positive/negative sentiment ("enjoyed" → positive, "watched" → could be neutral/negative depending on context). This requires a model that keeps _per-token_ representations $h_0, h_1, \ldots, h_T$ (as opposed to pooling everything into one vector) — motivating more expressive architectures like RNNs, self-attention, and Transformers, which retain positional/sequential information.
    
- **Text generation**: instead of predicting a label, predict **the next word**, one word at a time.
    
    - **Worked example**: given _"I really enjoyed the movie we watched on Saturday!"_, generate the continuation word by word:
        - Step 1: feed $X = {x_0, \ldots, x_T}$ → model → $h_T$ → **multi-class logistic regression** → $P(y) = \text{softmax}(W_0 h_T)$ → predicts `"It"` = $x_{T+1}$.
        - Step 2: append `"It"` to the input, recompute $h_{T+1} = \sum_{t=0}^{T+1} x_t$ → predict `"was"` = $x_{T+2}$.
        - Repeat, feeding each newly generated word back in as input, one word at a time — this word-by-word feedback loop is literally how many generative language models operate.
    - **Key change needed**: this is no longer _binary_ classification (+/−) — it's **multi-class classification over the entire vocabulary** $V$. If your vocabulary has a million words, you effectively have a million "classes," one per possible next word. This requires **softmax** instead of a simple sigmoid, since softmax converts a raw vector of scores into a full probability distribution over many classes (see §5 below), rather than a single probability of a binary outcome.
    - **Shapes, as the lecturer put it**: with a million-word vocabulary and a 1,000-dimensional hidden vector, the weight matrix takes the 1,000-dimensional vector and turns it into a million-entry vector, one score per word. The softmax turns that into the distribution, the most likely word (*it*) is fed back in, the next prediction is *was*, and so on.

    > «If you have a million words in your vocabulary, you have a million classes. […] To do that, you cannot rely just on a sigmoid activation function. You need a softmax, which actually converts a vector into a probability distribution.» — [1:42:12](https://au.cloud.panopto.eu/Panopto/Pages/Viewer.aspx?id=e32acb17-9550-4cdc-adea-b4b80077ea46&start=6132)

---

## 4. Why not just learn embeddings from your one supervised task?

You _can_ — embeddings can be learned end-to-end as part of a supervised task (e.g., sentiment classification), using backpropagation exactly as above. But this has two real downsides:

- **Data scarcity**: high-quality embeddings need a lot of training data, and clean, labeled data (e.g., thousands of hand-labeled movie reviews) is expensive and limited to collect.
- **Task-specificity**: embeddings learned this way are optimized _only_ for that one task/objective — e.g., embeddings learned purely from a movie-sentiment classifier will only capture what's needed to distinguish positive/negative sentiment. They won't tell you which words are more positive or negative, or the relationship between, say, "Ankara" and "Turkey," or any broader semantic structure — because the model was never asked to learn that.

**Contrast**: today's large language models are trained not on a small labeled dataset, but essentially on the entire internet — with no realistic upper bound on available (unlabeled) training data — which is a big part of why their embeddings capture so much general-purpose knowledge.

---

## 5. Self-supervised learning: the distributional hypothesis

> "You shall know a word by the company it keeps." — J. R. Firth, 1957

This is the **foundational idea of distributional semantics**: instead of relying on labeled data for one specific downstream task, rely on the **context** in which a word naturally occurs — the surrounding words — to learn its meaning. Context here just means "the set of words that occur nearby."

**Example — fill-in-the-blank intuition**:

> I really enjoyed the ______ we watched on Saturday! The ______ growled at me, making me run away. I need to go to the ______ to pick up some dinner.

A human (or a good model) can narrow down plausible fillers purely from context — "movie" or "show" for the first, an animal for the second, a store/restaurant for the third — without ever being told the answer. This means a model can be trained to _predict a missing (or surrounding) word purely from naturally-occurring raw text_ — no manual labeling needed. This is **self-supervised learning**: the "labels" (missing words) come for free from the text itself.

Two classic ways to operationalize this idea, both part of the original **word2vec** framework:

|Method|Objective|
|---|---|
|**CBOW** (Continuous Bag of Words)|Predict the missing (center) word from its surrounding context window|
|**Skip-gram**|Predict the surrounding context words from a single given (center) word|

_(GloVe is a third well-known approach, not covered in this lecture.)_

---

## 6. Continuous Bag of Words (CBOW) — full worked example

### Intuition first, before any formula

Forget matrices for a second. The whole idea behind CBOW and skip-gram is a simple **game you could play with a friend**, and it comes straight from the Firth quote in §5 ("you shall know a word by the company it keeps"):

> "I really enjoyed the ___ we watched on Saturday!"

You instantly guess "movie" or "show" — not because anyone told you the definition of "movie," but because you've seen the words *enjoyed*, *watched*, *Saturday* sit next to "movie" (or words like it) many times before. **CBOW turns this guessing game into a training signal**: take real sentences, blank out the center word, and train a model to predict it from the words around it. Do this over millions of sentences from a huge corpus, and the *only* way the model gets good at the guessing game is if it learns to represent words that appear in similar contexts (e.g. "movie" and "show" and "film") with similar vectors — because similar contexts should produce similar predictions.

**Why does this give you *meaning* instead of just co-occurrence counts?** Two words never even have to appear in the exact same sentence to end up with similar vectors — they just need to show up in *similar kinds of surrounding words* across the corpus. That's the key upgrade over sparse BoW vectors (§2): meaning comes from *distributional pattern*, not from literal shared words.

**Mechanically, in plain terms** (the same projection + softmax as the text-generation example in §3): you take the embeddings of the context words (e.g. *enjoyed, the, we, watched*), squash them together into one vector (by summing them), then ask "given this squashed context, which word from my entire vocabulary is most likely to be the missing one?" That last question — a probability over *every* word in the vocabulary — is exactly what softmax computes (see the primer box just below). The formal setup and a fully worked numeric example follow.

> [!note] Quick primer: what softmax actually does
> Before CBOW: softmax's job is to turn an arbitrary vector of real numbers (**scores**, aka **logits** — can be negative, can be huge, don't sum to anything meaningful) into a **valid probability distribution** — all entries $\geq 0$ and summing to 1.
>
> $$ \text{softmax}(\mathbf{a})_i = \frac{e^{a_i}}{\sum_{j=1}^{|\mathbf{a}|} e^{a_j}} $$
>
> ($\mathbf{a}$ is a vector, $a_i$ is its $i$-th dimension.)
>
> Mechanically, in two steps:
>
> 1. **Exponentiate** every entry ($e^{a_i}$) — this makes everything positive, and it *exaggerates* differences: a slightly bigger score becomes a *much* bigger number after exponentiation.
> 2. **Normalize** by dividing each exponentiated entry by the sum of all of them — this forces the result to sum to 1, i.e. a genuine distribution.
>
> **Tiny worked example** — 3 classes ("cat", "dog", "car") with raw scores $\mathbf{a} = (2.0,\ 1.0,\ 0.1)$:
>
> | class | score $a_i$ | $e^{a_i}$ | softmax probability |
> |---|---|---|---|
> | cat | 2.0 | 7.3891 | 7.3891 / 11.2125 ≈ **0.66** |
> | dog | 1.0 | 2.7183 | 2.7183 / 11.2125 ≈ **0.24** |
> | car | 0.1 | 1.1052 | 1.1052 / 11.2125 ≈ **0.10** |
>
> (sum of $e^{a_i}$ = 7.3891 + 2.7183 + 1.1052 = **11.2125**, used as the denominator for every entry)
>
> Result: $(0.66,\ 0.24,\ 0.10)$ — non-negative, sums to 1, and it preserves the *ranking* of the original scores (cat > dog > car) while amplifying the gap between them (a score gap of only 0.9 becomes a probability gap of 0.42).
>
> ![[appunti/assets/2026-09-01/softmax-quick-example.png]]
> _Left: arbitrary raw scores. Right: after softmax — same ranking, but now a proper probability distribution._
>
> **Why not just normalize the raw scores directly (divide by their sum)?** Two reasons: raw scores can be negative (breaks "divide by sum = probability" if the sum is small/negative), and exponentiation's amplification effect is actually desirable — it pushes the model toward a confident, peaked distribution around the highest-scoring class rather than a flat one, which is what you want when training a classifier with gradient descent.
>
> In CBOW/skip-gram (and text generation, §3), the "classes" are every word in the vocabulary $V$ — softmax converts the $V$-dimensional score vector $\mathbf{U}h_t$ into $P(\text{word} \mid \text{context})$ for every possible word at once.
>
> Softmax will reappear constantly throughout the course whenever multi-class prediction is needed — it's worth understanding cold.

**Setup**: predict the missing word from a window of surrounding words.

> Context: _enjoyed, the, ___, we, watched_ → target: **movie**

Architecture: take the context word embeddings → **sum** them → **project** (via a matrix) into vocabulary-sized space → **softmax** → probability distribution over the whole vocabulary.

![[appunti/assets/2026-09-01/nn-primer-072.png]]
_CBOW architecture: context word embeddings are summed, passed through a projection matrix_ $\mathbf{U}$_, then softmaxed to predict the missing center word ("movie")._

**Formally**, with window size $N=2$ on each side:

$$ \max\ P(x_t \mid {x_s}_{s=t-2}^{s=t+2}) = \text{softmax}!\left(\mathbf{U} \sum_{\substack{s=t-2 \\ s \neq t}}^{t+2} \mathbf{x}_s\right) $$

where:

- $\mathbf{x}_s \in \mathbb{R}^{1 \times d}$: the embedding of a context word ($d$ = embedding dimension)
- $\mathbf{U} \in \mathbb{R}^{d \times V}$: the **projection matrix**, mapping the $d$-dimensional summed context representation up to a $V$-dimensional vector (one entry per vocabulary word), where $V$ = vocabulary size
- $\mathbf{y} = \mathbf{U}^T \mathbf{x}^T$ (equivalently $\mathbf{y} = \mathbf{x} , @, \mathbf{U}$ in code notation)

**Fully worked numeric toy example** — suppose the output vocabulary is just 4 words: _"movie," "show," "book," "shelf"_ (so $V = 4$), and context embeddings sum to a 3-dimensional vector $h_t = \langle -2.3,\ 0.9,\ 5.4 \rangle$ (so $d = 3$). Suppose the projection matrix is:

$$ \mathbf{U} = \begin{pmatrix} 1.2 & -0.3 & 0.9 \\ 0.2 & 0.4 & -2.2 \\ 8.9 & -1.9 & 6.5 \\ 4.5 & 2.2 & -0.1 \end{pmatrix} \begin{matrix} \text{movie} \\ \text{show} \\ \text{book} \\ \text{shelf} \end{matrix} $$

Each **row** of $\mathbf{U}$ holds the "feature weights" for one vocabulary word; each **dimension** of $h_t$ corresponds to some learned feature of the context. Computing $\mathbf{U} h_t$ is just the dot product of each row of $\mathbf{U}$ with $h_t$:

$$ \text{row}_{\text{movie}} \cdot h_t = (1.2)(-2.3) + (-0.3)(0.9) + (0.9)(5.4) = -2.76 - 0.27 + 4.86 = 1.83 \approx 1.8 $$

Repeating this for every row gives:

$$ \mathbf{U} h_t = \langle 1.8,\ -1.9,\ 2.9,\ -0.9 \rangle $$

Finally, applying **softmax** converts this raw score vector into a genuine probability distribution over the 4 vocabulary words:

$$ \text{softmax}(\mathbf{U} h_t) = \langle 0.24,\ 0.006,\ 0.73,\ 0.02 \rangle $$

i.e., the model currently assigns 73% probability to "book," 24% to "movie," and near-zero to the other two — this is _before_ training; once trained on real data, a well-fit model would assign the highest probability to "movie" for a context like "enjoyed the ___ we watched," and this shape would gradually emerge from data through backpropagation.

**Training objective**: maximize the probability of the correct (missing) word given its context — equivalently, in practice, **minimize the negative log probability** of the missing word (for numerical/computational reasons — this is standard in ML, same idea as minimizing negative log-likelihood elsewhere). The window size $N$ (how many words on each side count as "context") is a **hyperparameter**, tuned the same way as row/column normalization choices — via a train/validation split, picking whatever setting performs best empirically.

_(Aside: for computational efficiency at scale, the true softmax over the entire vocabulary is often replaced with a cheaper approximation such as hierarchical softmax — mentioned here only for awareness, not required knowledge for this course.)_

---

## 7. Skip-gram

### Intuition first

Skip-gram plays the *exact same guessing game* as CBOW, just backwards. Where CBOW is "here are the neighbors, guess the missing word in the middle," skip-gram is:

> Here's **one** word — "movie." If I picked a random nearby sentence containing it, what other words would probably be sitting around it?

You'd guess things like *watched, enjoyed, film, cinema, ticket* — not because you memorized a definition, but because those are the kinds of words that tend to surround "movie" in real text. Skip-gram trains a model to do exactly this: given a center word, predict each of its neighbors, one at a time. Just like CBOW, the only way to get good at this task across millions of examples is to end up placing words with similar "neighbor profiles" close together in vector space.

**Concrete comparison with CBOW**, same sentence "I really enjoyed the movie we watched on Saturday!", center word "movie", window ±2:

| | Input | Output(s) to predict |
|---|---|---|
| CBOW | *enjoyed, the, we, watched* | **movie** (one prediction, from all 4 context words combined) |
| Skip-gram | **movie** | *enjoyed, the, we, watched* (4 separate predictions, one per neighbor) |

So CBOW makes **one prediction per training example** (many inputs → one output), while skip-gram makes **several predictions per training example** (one input → many outputs). This is why skip-gram treats each neighbor prediction as independent (see the conditional-independence assumption below) — it can't predict all 4 neighbors jointly in one shot the way CBOW predicts the single missing word, so it just asks "given 'movie', how likely is 'enjoyed' nearby?", "given 'movie', how likely is 'the' nearby?", etc., separately, and adds up how well it did on all of them.

**Setup**: the reverse of CBOW — predict the _surrounding_ context words from a single given (center) word.

> Given: **movie** → predict: _watched, we, the, enjoyed_

![[appunti/assets/2026-09-01/nn-primer-087.png]]
_Skip-gram architecture: the center word's embedding is projected and used to independently predict each surrounding context word._

$$ \max\ P(\text{enjoyed}, \text{the}, \text{we}, \text{watched} \mid \text{movie}) \quad\Longrightarrow\quad \max\ P(x_{t-2}, x_{t-1}, x_{t+1}, x_{t+2} \mid x_t) $$

Since jointly modeling this full conditional distribution directly is hard, skip-gram makes a **conditional independence assumption**: treat each surrounding-word prediction as independent given the center word, and just sum their log-probabilities:

$$ \max \sum_{s \neq t} \log P(x_s \mid x_t), \qquad \text{where } P(x_s \mid x_t) = \text{softmax}(\mathbf{U} x_t) $$

In practice, this means training an independent softmax prediction for _each_ context word position, and summing (or maximizing) across them — again with $\mathbf{x}_t \in \mathbb{R}^{1\times d}$ and $\mathbf{U} \in \mathbb{R}^{d \times V}$.

**Window size as hyperparameter, in more depth**: a larger window gives the embedding more information about related words but is more expensive. A common practical trick: set a large maximum window (e.g., $N=10$) but _randomly_ sample the effective window size $i \in [1, N]$ for each training example — this way, on average, **closer words contribute more** to learning than farther ones, which tends to match intuition (nearby words are usually more informative about a word's meaning than distant ones).

> ⚠️ **Exam (2025 paper / mock exam, Q4, rated hard)** — three Word2Vec statements, only **(2)** is true:
> 1. ❌ *"Generalises better to rare **and unseen** words than one-hot."* Better for **rare but seen** words, yes (they share a dense space with everything else). A **truly unseen** word has **no row in $E$** at all, so word2vec can't generalise to it.
> 2. ✅ *"CBOW and skip-gram ignore the order of words in the context window."* The window is a bag: CBOW sums the context vectors, and skip-gram's independence assumption (above) treats each neighbour separately.
> 3. ❌ *"Shuffling the words within each sentence doesn't change the vectors."* Order is ignored **inside** a window, but shuffling the sentence changes **which words fall into which window**: $A\,[C\,B\,D]\,E \to A\,[D\,B\,E]\,C$. Order-free within the window ≠ order-free overall.

> [!example] Lab — training skip-gram yourself, and why data size matters
> `ta_sessions/Exercises/03 - Language Modeling/0. Word Embeddings.ipynb`, §1–2. Gensim's `Word2Vec(..., vector_size=50, window=3, sg=1)` on the Brown corpus (~1M tokens of 1960s American English, ~530k after stopword removal). `sg=1` is skip-gram, `sg=0` CBOW; the lab picks skip-gram because it "works better on small corpora" (stated, not demonstrated in the lab).
>
> Nearest neighbours of *jury*:
> - own skip-gram: *wexler, reservations, subpenas, karns, prosecution*
> - GloVe (6B tokens): *jurors, judge, trial, court, verdict*
>
> Why: *jury* occurs only **68 times** in Brown, most of them from one news story about a grand-jury investigation, so a surname that happened to appear near it becomes a "neighbour". About **38% of the vocabulary occurs exactly once**, and with `min_count=1` those words get vectors from a handful of windows: mostly noise, which then pollutes the neighbour lists of common words too. With ~10,000× more text the accidents average out. Also: words outside the corpus's era are **OOV** (*internet*, *email*), and *web*, *mouse*, *cloud* only have their 1960s sense. The distributional hypothesis only works with **a lot** of distribution.

---

## 8. Connection to classical NLP: matrix factorization / LSA revisited

### Intuition first: what problem is LSA actually solving?

Start from the TF-IDF matrix $T$ you already know from Lecture 1: rows = documents, columns = words, entries = how much each word matters in each document. It's huge (thousands of documents × thousands of words) and mostly zeros, because any one document only uses a tiny fraction of the vocabulary.

**The intuition**: a document isn't really "about" thousands of individual words — it's about a handful of **topics** (say, *sports*, *politics*, *cooking*...), and each topic tends to pull in a characteristic *set* of words. A sports article uses "goal," "match," "team," "score" together; a cooking article uses "oven," "recipe," "simmer," "flavor" together. So instead of describing a document by which of the 10,000 individual words it contains, you could describe it much more compactly by *how much of each of, say, 50 topics* it contains — e.g. "this document is 80% sports, 15% politics, 5% everything else."

**Concrete toy example**: imagine 4 documents and these word counts (simplified/pretend TF-IDF):

| | goal | match | team | oven | recipe | flavor |
|---|---|---|---|---|---|---|
| Doc 1 (football report) | 5 | 4 | 3 | 0 | 0 | 0 |
| Doc 2 (recipe blog) | 0 | 0 | 0 | 4 | 5 | 3 |
| Doc 3 (football + food festival) | 3 | 2 | 1 | 2 | 3 | 2 |
| Doc 4 (transfer news) | 4 | 5 | 4 | 0 | 0 | 0 |

A human looking at this table immediately sees **two underlying topics**: a "sports" topic (goal/match/team) and a "food" topic (oven/recipe/flavor) — Doc 1 and Doc 4 are ~100% sports, Doc 2 is ~100% food, Doc 3 is a mix of both. **This is exactly what matrix factorization automates**: instead of you eyeballing which columns cluster together, SVD finds those hidden topic groupings mathematically, and re-expresses every document as a short vector of "how much of topic 1, how much of topic 2, ..." instead of a long vector of raw word counts. That compact per-document vector (a row of $A$ below) is much more useful for clustering/classification than the sparse raw word-count row — and as a side effect, you also get, for each *word*, how strongly it loads onto each topic (a column of $B$) — which is itself a usable dense **word vector**: "goal" and "match" end up with similar topic-loadings because they both belong mostly to the sports topic, even though as raw TF-IDF columns they never technically "interacted."

**Formally**: recall from Lecture 1's classical-NLP toolkit: for topic detection, you can factorize the TF-IDF matrix (rows = docs, columns = words) into two smaller matrices $A$ (docs × topics) and $B$ (topics × words), such that $T \approx AB$. This is **Latent Semantic Analysis (LSA)**, efficiently computed via **Singular Value Decomposition (SVD)**: $T = USV^T$.

![[appunti/assets/2026-09-01/nn-primer-098.png]]
_LSA as matrix factorization: the (docs × words) TF-IDF matrix is approximated as the product of a (docs × topics) matrix_ $A$ _and a (topics × words) matrix_ $B$_. The optimization problem is to find_ $A, B$ _minimizing_ $\sum_{d,w}(T_{dw} - A_d B_w)^2$_, i.e. the squared reconstruction error over every doc/word entry._

- Both $U$ and $V$ have orthonormal columns; $S$ is diagonal with values in decreasing order (each capturing the "importance"/variance explained by a topic).
- To get $k$ topics: keep only the first $k$ columns of $U$, $V$ and the first $k$ rows/cols of $S$ → $U', S', V'$, then set e.g. $A = U'$, $B = S'V'^T$.
- Using $A$ (the low-dimensional doc representation) instead of the raw TF-IDF matrix also sidesteps the **curse of dimensionality** that plagues clustering/classification directly on sparse, high-dimensional TF-IDF rows (too many outliers, distances become meaningless in very high dimensions).

> [!example] Lab MCQ — what LSA topics are *not*
> `ta_sessions/Exercises/01 - Intro to Tools/2. Handling Text - Part 1.ipynb`, Part 2, Q2: which statement about LSA topic detection is **false**?
> - ✅ It represents documents and words in a latent "topic space" (rows of $A$, columns of $B$).
> - ✅ Topics are interpretable in "word space": read off the words with the largest weights in each row of $B$.
> - ✅ The topic vectors in word space are **linearly independent**: they're rows of $V^T$, which are orthonormal.
> - ❌ **"A document's topic representation is a probability distribution over topics."** False. SVD entries can be **negative** and don't sum to 1; a row of $A$ is a set of real-valued coordinates, not proportions.
>
> ⚠️ So the "80% sports, 15% politics" phrasing in the intuition above is a **loose** way of speaking, not what LSA outputs. Topic *proportions* are what probabilistic topic models such as LDA produce; LSA only gives coordinates. (Same lab, Q3, on the §1 recap: when a TF-IDF matrix has **more features than documents**, SVD, regularisation and feature selection all help, but **min-max scaling doesn't**: rescaling a column doesn't remove it.)

> [!note] Curse of dimensionality, in plain terms
> As the number of dimensions grows, distance-based metrics (Euclidean, cosine, ...) become progressively less informative: points tend to become **roughly equidistant** from one another, so "near" and "far" stop meaningfully separating similar items from dissimilar ones.
>
> Geometric intuition: in an $n$-dimensional cube, as $n$ grows most of the volume concentrates near the corners/surface rather than the center — random points end up scattered toward the "edges," at similarly large distances from each other, instead of clustering meaningfully near/far based on actual similarity.
>
> This is exactly why running clustering/classification directly on raw TF-IDF rows (dimension = vocabulary size, often $O(10^4)$–$O(10^6)$) is problematic: distances between document vectors stop reflecting real topical similarity. **Fix**: reduce dimensionality first (e.g. LSA/SVD → $A$, only tens/hundreds of dimensions) so distances become meaningful again — this is one of the practical motivations for both matrix factorization and dense embeddings in general.
>
> **Example**: clustering 1,000 news articles using raw TF-IDF rows with a 50,000-word vocabulary (50,000 dimensions) — k-means will likely produce near-meaningless clusters, since almost all document pairs end up at similar distances. Reducing to ~100 LSA topic-dimensions first fixes this.

**Word vectors from matrix factorization**: sparse TF-IDF columns (or dense LSA columns of $B$) can themselves be used as "word vectors" — but treating an entire document as the unit of context throws away all information about word order/proximity/syntax. **Fix**: instead of factorizing a doc × word matrix, factorize a matrix $M$ of **local context windows × words** (e.g., windows of $L=3$ words to the left/right of each target word) — this captures much more fine-grained co-occurrence information.

What goes in the entries of $M$?

- Simplest: same as TF-IDF, but treating each context window as a "pseudo-document": $M[c, w] = \text{TF-IDF}(c, w)$.
- Alternative, more principled statistical measure: **Pointwise Mutual Information (PMI)**:

$$ M[c, w] = \text{PMI}(c, w) = \log \frac{\Pr(c, w)}{\Pr(c),\Pr(w)} $$

— intuitively, "how much more likely are context $c$ and word $w$ to co-occur than if they were statistically independent?"

> [!example] Lab MCQ — PMI by hand
> `ta_sessions/Exercises/01 - Intro to Tools/2. Handling Text - Part 1.ipynb`, Part 2, Q5. Co-occurrence counts (total $N = 20$):
>
> | | W1 | W2 | W3 | row sum |
> |---|---|---|---|---|
> | C1 | 2 | 0 | 3 | 5 |
> | C2 | 6 | 4 | 0 | 10 |
> | C3 | 1 | 1 | 3 | 5 |
> | col sum | 9 | 5 | 6 | 20 |
>
> With probabilities as counts over $N$, $\text{PMI}(c,w) = \log\dfrac{n_{cw}\,N}{n_c\,n_w}$:
> - $\text{PMI}(W1, C2) = \log\frac{6\cdot20}{10\cdot9} = \log 1.33$
> - $\text{PMI}(W2, C2) = \log\frac{4\cdot20}{10\cdot5} = \log 1.6$
> - $\text{PMI}(W3, C3) = \log\frac{3\cdot20}{5\cdot6} = \log 2$
>
> So $\text{PMI}(W3,C3) > \text{PMI}(W2,C2) > \text{PMI}(W1,C2)$, answer (d). **The trap:** W1–C2 has the **largest raw count** (6) but the **smallest PMI** of the three, because both W1 and C2 are frequent everywhere, so a lot of co-occurrence is expected by chance. PMI rewards co-occurrence **above chance**, not co-occurrence as such.

![[appunti/assets/2026-09-01/nn-primer-104.png]]
_Same factorization idea, applied to a (contexts × words) PMI matrix instead of a (docs × words) TF-IDF matrix: word2vec is essentially learning the columns of_ $B$ _("latent dims" = embedding dimensions) as word vectors._

**Key theoretical connection**: it has been shown that **word2vec is implicitly performing matrix factorization of a (shifted) PMI matrix** — i.e., factorizing $M[c,w] = \text{PMI}(c,w)$ into $A$ (context vectors) and $B$ (word vectors), then using the columns of $B$ as word embeddings, is mathematically closely related to what word2vec learns via its neural training objective. In other words, the "classical" linear-algebra route (SVD/matrix factorization of a co-occurrence statistic) and the "neural" route (CBOW/skip-gram trained via backpropagation) are two different paths to a similar kind of representation. (For the curious: this connection is explored in the paper _"Neural Word Embedding as Implicit Matrix Factorization"_ and various accompanying blog explainers.)

> [!example] Lab — GloVe, the count-based cousin
> `ta_sessions/Exercises/03 - Language Modeling/0. Word Embeddings.ipynb`, §2. GloVe, "not covered in this lecture" (§5), is the pre-trained model the labs actually use: `glove-wiki-gigaword-100`, 100-d vectors trained on 6B tokens of Wikipedia + Gigaword (the Handling Text lab uses spaCy's 300-d GloVe on Common Crawl). Per the lab, GloVe **factorises global co-occurrence counts** directly, i.e. it takes the explicit matrix-factorisation route of this section, where word2vec gets there implicitly through a prediction task.
>
> Two lab observations that belong here:
> - **Dimensions carry no individual meaning** (Handling Text lab, MCQ Q4: *"each dimension of a word2vec embedding has a clear interpretation"* is the **false** statement). Meaning lives in *directions and distances*, which is why analogies work (§2), not in single coordinates. Contrast LSA, where each dimension is a topic you can at least try to read off from its top-weighted words.
> - **The training corpus freezes one sense per word**: GloVe's odd-one-out on *python, java, rust, banana* answers **rust**, because in 2014 Wikipedia/news text *rust* mostly means corrosion. A static embedding can't do better than its corpus's dominant sense (§9).

---

## 9. Beyond single-word embeddings

- So far: word2vec gives one fixed vector per unique word type in the vocabulary.
- To represent whole **sentences or documents** as vectors, options include:
    - Simple composition of pre-trained word vectors (e.g., sum/average all word vectors in the sentence) — note this is essentially the same idea as bag-of-words with one-hot word vectors, just with dense vectors instead.
    - Dedicated methods to embed sentences/documents directly, e.g. **Doc2Vec** ("Paragraph Vectors"), a well-known extension of word2vec — also mentioned in passing: **Cr5**, **sent2vec**, convolutional neural networks, and recurrent networks (LSTM, **ELMo**) as other approaches to learning representations for longer text units.
- **Limitation of classical word embeddings (word2vec/GloVe/LSA-based)**: each word gets exactly **one fixed vector**, regardless of context — a lookup-table/dictionary model. This means a polysemous word (one with multiple meanings depending on context) is forced into a single, averaged-out representation.

> **Motivating example**: "The **bat** flew into the cave" vs. "He swung the **bat** and hit a home run." A classic word2vec-style embedding uses the exact same vector for "bat" in both sentences — it cannot distinguish the animal from the piece of sports equipment.

- **What changed with Transformer-based models (e.g., BERT)**: instead of a static lookup table, these models produce **contextual embeddings** — the vector representing a word changes depending on its surrounding sentence context, so the same word can get different vectors in different usages (e.g., a word used as a noun with one sense vs. the same word used with a different sense elsewhere) — something fundamentally impossible with classical, context-free embeddings by design. This transition (static → contextual embeddings) is the conceptual bridge into the rest of the course, starting with language modeling next week.

![[appunti/assets/2026-09-01/nn-primer-108.png]]
_BERT (introduced by Google Research in 2018) takes a sequence of tokens and outputs one vector per token (plus an overall "doc" vector for the special_ `<START>` _token) — the vector for "bat" differs between the two sentences because it depends on the whole surrounding context, unlike a static word2vec vector._

> [!note] Static vs. contextual embeddings, side by side (my own summary — not on the slides)
> **The one-line version**: word2vec/LSA learn a **static dictionary** (one vector per word *type*, fixed once training ends), while BERT learns a **function** that computes a vector per word *token*, on the fly, from the whole surrounding sentence.
>
> | | word2vec / LSA | BERT (contextual) |
> |---|---|---|
> | Unit represented | word **type** (vocabulary entry) | word **token** (occurrence in a specific sentence) |
> | What's stored | a lookup table $E$ of $\|V\| \times d$ vectors | model *weights*; vectors are computed at inference time |
> | Polysemy | one averaged vector for all senses of "bat" | different vector per sense, driven by context |
> | Context used | only during *training* (window of ±$N$ words), discarded afterwards | at *every* forward pass, over the whole input sequence |
> | Word order | ignored (bag of context words) | preserved (positional encodings + self-attention) |
>
> **Why the static approach is limited**: since the vector for "bat" must serve both "the bat flew into the cave" and "he swung the bat," training pulls it toward a compromise point somewhere between the animal cluster and the sports-equipment cluster — a vector that represents *neither* sense well. The more polysemous the word, the worse the blur.
>
> **What contextual models do differently**: the embedding is no longer the *output* of training but an intermediate representation *recomputed for each input*. BERT still starts from a static input embedding table, but then passes it through several self-attention layers that let each token's representation absorb information from every other token in the sentence — so by the output layer, "bat" has been pushed toward whichever sense the surrounding words support.
>
> **Consequences worth remembering**:
> - You can no longer ask "what is the vector for *bat*?" without also specifying a sentence — the question is ill-posed for BERT.
> - Cosine similarity between two *occurrences* of the same word becomes meaningful (it measures sense agreement), which is impossible with word2vec (always exactly 1).
> - Static embeddings remain useful where cheapness matters: they're a single table lookup, no forward pass through a large network required.

---

## 10. Open question posed at the end of the lecture — and its answer

> **What is the major conceptual difference between the CBOW and Skip-gram methods for training word embeddings?**

Left open at the end of this lecture (last content slide of the deck) and **answered at the start of Lecture 3** (2026-09-08), in the recap.

### The answer given in Lecture 3

Instead of reasoning from the objective functions, the lecturer answered **empirically**: train both models on the same corpus (evidently recipe text, judging by the vocabulary), then ask each one for the 10 nearest neighbours of the word *cut* — `model.wv.most_similar('cut', topn=10)`.

![[appunti/assets/2026-09-08/lm-pt1-022.png]]

| rank | Skip-gram | CBOW |
|---|---|---|
| 1 | crosswise (0.729) | slice (0.662) |
| 2 | score (0.703) | crosswise (0.650) |
| 3 | slice (0.697) | score (0.631) |
| 4 | crossways (0.680) | tear (0.619) |
| 5 | 1/2-inch-thick (0.678) | dice (0.564) |
| 6 | diamonds (0.672) | lengthwise (0.557) |
| 7 | diagonally (0.670) | cutting (0.557) |
| 8 | lengthwise (0.665) | break (0.552) |
| 9 | cutting (0.664) | chop (0.542) |
| 10 | wise (0.657) | carve (0.538) |

**First observation**: there is a **large overlap** — the two methods are *not* learning entirely different things (*crosswise, score, slice, lengthwise, cutting* show up in both lists). The interesting part is the **pattern** in what each list adds on top of the shared core:

- **CBOW** → *slice, tear, dice, break, chop, carve*: words denoting the **same kind of action**, near-synonyms that could be **substituted for** "cut" in the same slot. This is **paradigmatic** similarity (same-role, interchangeable words).
- **Skip-gram** → *crosswise, crossways, 1/2-inch-thick, diamonds, diagonally, lengthwise, wise*: not other ways of cutting at all, but words describing **how** the cutting is done — i.e. words that literally **appear around** "cut" in real text ("cut *crosswise*", "cut *lengthwise*", "cut into *diamonds*", "cut *1/2-inch-thick*"). This is **syntagmatic** relatedness (co-occurrence, not substitutability).

> «When you look at skip-gram […] the one actually gives you a relationship that tells you how this word actually appears in the surrounding context, right? And the other [CBOW] actually gives you more of a relationship of this word with words that are similar to this in terms of the action.» — [50:12](https://au.cloud.panopto.eu/Panopto/Pages/Viewer.aspx?id=2ec320a2-fe53-47e8-a98f-b4bf00712f16&start=3012) (08 Sep recording)

**Why this falls out of the two objectives** (§6–§7): CBOW's training signal is *"given this bag of neighbours, which word fills the blank?"* — so any two words that plausibly fill the **same** blank get pushed together, which is essentially the definition of substitutability. Skip-gram's signal is *"given this word, what surrounds it?"* — so a word's vector becomes a compressed description of its **typical neighbourhood**, and words with overlapping neighbourhoods end up close even when they are not interchangeable.

**Methodological takeaway the lecturer stressed**: you don't always need to start from the objective function. Training both and simply **looking at the output** is a legitimate — and often faster — way to work out what a model has learned; you can then go back to the maths to explain what you observed.

### Related question asked after the lecture

> Do you get embeddings for both the context words and the target word?

Answered in the same 08 Sep recap: **yes** — in both CBOW and skip-gram you end up with **two sets of embeddings** per vocabulary word: one for the word in its role as **target**, one for the word in its role as **context**. These are exactly the two learnable matrices from §6: the input embedding table $E$ and the projection matrix $\mathbf{U} \in \mathbb{R}^{d \times V}$ (whose rows/columns are the "context" vectors). In practice the input embeddings are the ones usually kept and used as *the* word vectors, but the second set exists and can also be used.
---

## Da ricordare

- **Sparse vs dense:** two different words always have similarity **0** in BoW/TF-IDF space. Embeddings are **lower-dimensional and dense** ($O(10^2\text{–}10^3)$ vs $O(|V|)$). Embedding table $E$ is **$|V| \times d$**. A linear layer's $W$ is **(out × in)**.
- **CBOW** = predict the **center word from the context** (sum of context vectors → $\mathbf U$ → softmax). **Skip-gram** = predict **each context word from the center word** (conditional independence). CBOW neighbours are **substitutes** (*slice, chop*), skip-gram neighbours are **co-occurring** words (*crosswise, lengthwise*). Both give **two** sets of vectors ($E$ and $\mathbf U$).
- ⚠️ **Word2Vec traps (2025 exam Q4):** order inside the window is ignored ✅, but **shuffling a sentence does change the vectors** (it changes which words share a window), and **unseen words get no vector at all**. Individual dimensions have **no interpretation**.
- **Cosine** = direction only, range $[-1,1]$. It measures **relatedness of usage**, so **antonyms are close** (*good–bad*, *hot–cold*). Analogy: $b - a + c$, nearest by cosine, **input words excluded**, which flatters both the analogy property and the measured bias.
- **word2vec ≈ implicit factorisation of a (shifted) PMI matrix**; $\text{PMI} = \log\frac{P(c,w)}{P(c)P(w)} = \log\frac{n_{cw}N}{n_c n_w}$. **High raw count ≠ high PMI.** LSA rows are **coordinates, not probability distributions** (entries can be negative); topic vectors are orthonormal.
- **Averaging word vectors throws away order:** *"good, not bad"* and *"bad, not good"* get the **identical** vector, so no classifier on top can separate them. Static embeddings also give **one vector per word type** (*bat*, *rust*); contextual models (BERT) give one per token.
- **NN basics:** without a nonlinearity, stacked linear layers = one linear map. A single perceptron = **linear boundary**. $\sigma' = \sigma(1-\sigma) \le 0.25$. Backprop computes **gradients of the loss w.r.t. the parameters** (2025 exam Q1). Order: forward → loss → backward → update.
