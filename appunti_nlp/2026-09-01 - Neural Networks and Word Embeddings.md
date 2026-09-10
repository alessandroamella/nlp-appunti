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

![[assets/2026-09-01/nn-primer-036.png]]
_Male–Female, verb-tense, and country–capital relationships all show up as **consistent, parallel directions/offsets** in a well-trained embedding space — the same "arrow" (e.g., man→woman) works for king→queen too._

### Dense word vectors (recap + goal)

- Each word → a **dense, real-valued vector**, still high-dimensional relative to a scalar, but _much_ smaller than sparse BoW vectors: typically $O(10^3)$ dimensions instead of $O(10^5)$–$O(10^6)$.
    - Aside: even today's large language models typically use embedding dimensions that are "low" by LLM standards but still in the thousands (e.g., multiples of 1024/4096-ish ranges) — nowhere near the millions of dimensions of a sparse BoW vocabulary vector.
- **Similarity of vectors should represent similarity of meaning.** A well-trained embedding space gives $\text{sim}(\text{enjoyed}, \text{loved}) \gg 0$, unlike the sparse case.

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
4. **Learnable parameters**: the embedding table $E$ and the classifier weight matrix $W_0$ — both trained jointly via **backpropagation**, by computing gradients of the loss with respect to the initial embeddings (and all other parameters) so the embeddings end up encoding whatever helps solve the task.

**Worked example — counting unique embeddings**: for the sentence _"I enjoyed the movie I watched on Saturday!"_, there are only **7 unique embeddings** needed in $X$, because the word "I" appears twice and shares the same row in the embedding table $E$ both times (not 9, which is the raw token count).

**Backpropagation, in plain terms**: you have labeled training data (e.g., a movie review labeled positive/negative). You initialize embeddings randomly, run the forward pass (pool → logistic regression) to get a predicted probability, compute a **loss** measuring how far that prediction is from the true label (e.g., predicted 0.1 but the true label is 1), and then propagate that error backward through the network, adjusting the embeddings and weights so that the next prediction is closer to the truth. Repeated over many examples, this is how the embeddings and classifier weights are jointly learned. (Backpropagation will be covered in depth in the coming lectures on language modeling, and practiced hands-on in this week's lab.)

### Beyond single-label classification

- **Sequence labeling**: instead of one label for the whole sentence, produce a label _per word_ — e.g., tag which specific words carry positive/negative sentiment ("enjoyed" → positive, "watched" → could be neutral/negative depending on context). This requires a model that keeps _per-token_ representations $h_0, h_1, \ldots, h_T$ (as opposed to pooling everything into one vector) — motivating more expressive architectures like RNNs, self-attention, and Transformers, which retain positional/sequential information.
    
- **Text generation**: instead of predicting a label, predict **the next word**, one word at a time.
    
    - **Worked example**: given _"I really enjoyed the movie we watched on Saturday!"_, generate the continuation word by word:
        - Step 1: feed $X = {x_0, \ldots, x_T}$ → model → $h_T$ → **multi-class logistic regression** → $P(y) = \text{softmax}(W_0 h_T)$ → predicts `"It"` = $x_{T+1}$.
        - Step 2: append `"It"` to the input, recompute $h_{T+1} = \sum_{t=0}^{T+1} x_t$ → predict `"was"` = $x_{T+2}$.
        - Repeat, feeding each newly generated word back in as input, one word at a time — this word-by-word feedback loop is literally how many generative language models operate.
    - **Key change needed**: this is no longer _binary_ classification (+/−) — it's **multi-class classification over the entire vocabulary** $V$. If your vocabulary has a million words, you effectively have a million "classes," one per possible next word. This requires **softmax** instead of a simple sigmoid, since softmax converts a raw vector of scores into a full probability distribution over many classes (see §5 below), rather than a single probability of a binary outcome.
    - **In the lecturer's own words** (from the recording, paraphrased since the auto-transcript is rough): "it takes a sequence as input and it generates the next word — you feed in all the words you want, then you generate the next word, then you feed that new word back in as well and generate one more, and you keep going like this." On why it must be multi-class rather than binary: "if you have a million words in your vocabulary, you have a million classes — because you want to see which word comes next, all the vocabulary words are your class labels. You cannot just use a sigmoid, you need a softmax, which converts a vector into a probability distribution — the weight matrix transforms the hidden vector into a vector the size of your vocabulary, and softmax turns that into a probability distribution over all words. That tells you the most likely next word. Then you feed it back in and keep going."

---

## 4. Why not just learn embeddings from your one supervised task?

You _can_ — embeddings can be learned end-to-end as part of a supervised task (e.g., sentiment classification), using backpropagation exactly as above. But this has two real downsides:

- **Data scarcity**: high-quality embeddings need a lot of training data, and clean, labeled data (e.g., thousands of hand-labeled movie reviews) is expensive and limited to collect.
- **Task-specificity**: embeddings learned this way are optimized _only_ for that one task/objective — e.g., embeddings learned purely from a movie-sentiment classifier will only capture what's needed to distinguish positive/negative sentiment. They won't necessarily capture the general relationship between, say, "Ukraine" and "Kyiv," or any broader semantic structure — because the model was never asked to learn that.

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

**Mechanically, in plain terms** (matches the "in the lecturer's own words" quote in §3): you take the embeddings of the context words (e.g. *enjoyed, the, we, watched*), squash them together into one vector (by summing them), then ask "given this squashed context, which word from my entire vocabulary is most likely to be the missing one?" That last question — a probability over *every* word in the vocabulary — is exactly what softmax computes (see the primer box just below). The formal setup and a fully worked numeric example follow.

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
> ![[assets/2026-09-01/softmax-quick-example.png]]
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

![[assets/2026-09-01/nn-primer-072.png]]
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

![[assets/2026-09-01/nn-primer-087.png]]
_Skip-gram architecture: the center word's embedding is projected and used to independently predict each surrounding context word._

$$ \max\ P(\text{enjoyed}, \text{the}, \text{we}, \text{watched} \mid \text{movie}) \quad\Longrightarrow\quad \max\ P(x_{t-2}, x_{t-1}, x_{t+1}, x_{t+2} \mid x_t) $$

Since jointly modeling this full conditional distribution directly is hard, skip-gram makes a **conditional independence assumption**: treat each surrounding-word prediction as independent given the center word, and just sum their log-probabilities:

$$ \max \sum_{s \neq t} \log P(x_s \mid x_t), \qquad \text{where } P(x_s \mid x_t) = \text{softmax}(\mathbf{U} x_t) $$

In practice, this means training an independent softmax prediction for _each_ context word position, and summing (or maximizing) across them — again with $\mathbf{x}_t \in \mathbb{R}^{1\times d}$ and $\mathbf{U} \in \mathbb{R}^{d \times V}$.

**Window size as hyperparameter, in more depth**: a larger window gives the embedding more information about related words but is more expensive. A common practical trick: set a large maximum window (e.g., $N=10$) but _randomly_ sample the effective window size $i \in [1, N]$ for each training example — this way, on average, **closer words contribute more** to learning than farther ones, which tends to match intuition (nearby words are usually more informative about a word's meaning than distant ones).

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

![[assets/2026-09-01/nn-primer-098.png]]
_LSA as matrix factorization: the (docs × words) TF-IDF matrix is approximated as the product of a (docs × topics) matrix_ $A$ _and a (topics × words) matrix_ $B$_. The optimization problem is to find_ $A, B$ _minimizing_ $\sum_{d,w}(T_{dw} - A_d B_w)^2$_, i.e. the squared reconstruction error over every doc/word entry._

- Both $U$ and $V$ have orthonormal columns; $S$ is diagonal with values in decreasing order (each capturing the "importance"/variance explained by a topic).
- To get $k$ topics: keep only the first $k$ columns of $U$, $V$ and the first $k$ rows/cols of $S$ → $U', S', V'$, then set e.g. $A = U'$, $B = S'V'^T$.
- Using $A$ (the low-dimensional doc representation) instead of the raw TF-IDF matrix also sidesteps the **curse of dimensionality** that plagues clustering/classification directly on sparse, high-dimensional TF-IDF rows (too many outliers, distances become meaningless in very high dimensions).

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

![[assets/2026-09-01/nn-primer-104.png]]
_Same factorization idea, applied to a (contexts × words) PMI matrix instead of a (docs × words) TF-IDF matrix: word2vec is essentially learning the columns of_ $B$ _("latent dims" = embedding dimensions) as word vectors._

**Key theoretical connection**: it has been shown that **word2vec is implicitly performing matrix factorization of a (shifted) PMI matrix** — i.e., factorizing $M[c,w] = \text{PMI}(c,w)$ into $A$ (context vectors) and $B$ (word vectors), then using the columns of $B$ as word embeddings, is mathematically closely related to what word2vec learns via its neural training objective. In other words, the "classical" linear-algebra route (SVD/matrix factorization of a co-occurrence statistic) and the "neural" route (CBOW/skip-gram trained via backpropagation) are two different paths to a similar kind of representation. (For the curious: this connection is explored in the paper _"Neural Word Embedding as Implicit Matrix Factorization"_ and various accompanying blog explainers.)

---

## 9. Beyond single-word embeddings

- So far: word2vec gives one fixed vector per unique word type in the vocabulary.
- To represent whole **sentences or documents** as vectors, options include:
    - Simple composition of pre-trained word vectors (e.g., sum/average all word vectors in the sentence) — note this is essentially the same idea as bag-of-words with one-hot word vectors, just with dense vectors instead.
    - Dedicated methods to embed sentences/documents directly, e.g. **Doc2Vec** ("Paragraph Vectors"), a well-known extension of word2vec — also mentioned in passing: **Cr5**, **sent2vec**, convolutional neural networks, and recurrent networks (LSTM, **ELMo**) as other approaches to learning representations for longer text units.
- **Limitation of classical word embeddings (word2vec/GloVe/LSA-based)**: each word gets exactly **one fixed vector**, regardless of context — a lookup-table/dictionary model. This means a polysemous word (one with multiple meanings depending on context) is forced into a single, averaged-out representation.

> **Motivating example**: "The **bat** flew into the cave" vs. "He swung the **bat** and hit a home run." A classic word2vec-style embedding uses the exact same vector for "bat" in both sentences — it cannot distinguish the animal from the piece of sports equipment.

- **What changed with Transformer-based models (e.g., BERT)**: instead of a static lookup table, these models produce **contextual embeddings** — the vector representing a word changes depending on its surrounding sentence context, so the same word can get different vectors in different usages (e.g., a word used as a noun with one sense vs. the same word used with a different sense elsewhere) — something fundamentally impossible with classical, context-free embeddings by design. This transition (static → contextual embeddings) is the conceptual bridge into the rest of the course, starting with language modeling next week.

![[assets/2026-09-01/nn-primer-108.png]]
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

![[assets/2026-09-08/lm-pt1-022.png]]

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

> **In the lecturer's own words** (paraphrased — the auto-transcript is rough): "when you look at skip-gram, it gives you a relationship of how this word actually appears in a surrounding context, and the other one [CBOW] gives you more of a relationship of this word with words that are similar to it in terms of the action."

**Why this falls out of the two objectives** (§6–§7): CBOW's training signal is *"given this bag of neighbours, which word fills the blank?"* — so any two words that plausibly fill the **same** blank get pushed together, which is essentially the definition of substitutability. Skip-gram's signal is *"given this word, what surrounds it?"* — so a word's vector becomes a compressed description of its **typical neighbourhood**, and words with overlapping neighbourhoods end up close even when they are not interchangeable.

**Methodological takeaway the lecturer stressed**: you don't always need to start from the objective function. Training both and simply **looking at the output** is a legitimate — and often faster — way to work out what a model has learned; you can then go back to the maths to explain what you observed.

### Related question asked after the lecture

> Do you get embeddings for both the context words and the target word?

**Yes** — in both CBOW and skip-gram you end up with **two sets of embeddings** per vocabulary word: one for the word in its role as **target**, one for the word in its role as **context**. These are exactly the two learnable matrices from §6: the input embedding table $E$ and the projection matrix $\mathbf{U} \in \mathbb{R}^{d \times V}$ (whose rows/columns are the "context" vectors). In practice the input embeddings are the ones usually kept and used as *the* word vectors, but the second set exists and can also be used.