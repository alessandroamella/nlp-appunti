## 1. What is a language model?

> An **LM is a probabilistic model of a sequence of tokens**: how likely is a given phrase / sentence / paragraph / document?

A sequence $w_1, w_2, \ldots, w_n$ is modelled as a **joint distribution** over its tokens, and the language model's entire job is to estimate that number:

$$P(w_1, w_2, w_3, \ldots, w_n)$$

Two points the lecturer insisted on:

- **Tokens, not words.** In the previous lectures we said "word", but a token is the more general unit: it can be a word, a sub-word piece, three characters, a character. Nothing in the maths below cares which — it only cares that the text has been chopped into a fixed inventory of discrete units.
- **What "good" means.** A good LM assigns a **high** probability to sequences that plausibly occur in the language, and a **low** probability to junk. That is the *only* definition of quality we have at this stage, and it is what every evaluation metric in §7–§10 tries to operationalise.

Two pieces of notation used throughout the lecture (and worth getting right, because they are easy to swap by accident):

| term | meaning |
|---|---|
| **word type** | a unique word in the vocabulary $V$ — an *entry in the dictionary* |
| **token** | an *instance* of a word type in the corpus |

So "the cat sat on the mat" has 6 tokens but 5 types ("the" appears twice).

---

## 2. Language models are everywhere

The deck's framing: *predicting the next token is important in many situations.*

- **Classical — text generation**: $P(a\ \textbf{smooth}\ finish) > P(a\ \textbf{flat}\ finish)$.
- **Classical — spell checking**: $P(high\ school\ \textbf{principal}) > P(high\ school\ \textbf{principle})$. Note this is a *homophone* error that a dictionary-based spell checker cannot catch — both words exist; only the sequence probability distinguishes them. This is the cleanest argument for why you need a model of sequences and not of words.
- **Modern**: information extraction, question answering, instruction-following LLMs.

Everyday examples the lecturer walked through: Google search autocomplete, Gmail's "Smart Compose" suggestions, and your phone keyboard's next-word bar. His point, said explicitly: these do **not** have to be LLMs — *"your phone keyboard is a language model. It can be a stupid language model, but it is still a language model."* The autocomplete in Gmail is powered by an **LSTM**-class model (covered next week), not a frontier LLM.

---

## 3. Chain rule: turning a joint probability into next-token predictions

We cannot estimate $P(w_1, \ldots, w_n)$ directly (see §4). The chain rule of probability rewrites it as a product of conditionals:

$$P(w_1, w_2, \ldots, w_n) = P(w_1) \times P(w_2 \mid w_1) \times P(w_3 \mid w_1, w_2) \times \cdots \times P(w_n \mid w_1, \ldots, w_{n-1})$$

$$P(w_1, w_2, \ldots, w_n) = \prod_i P(w_i \mid \underbrace{w_1, w_2, \ldots, w_{i-1}}_{\textbf{prefix}})$$

The conditioning context $w_1 \ldots w_{i-1}$ is called the **prefix** — a term used constantly for the rest of the course.

**This is the conceptual pivot of the whole lecture**: it converts "score a whole sentence" (impossible) into "repeatedly predict the next token given everything before it" (a classification problem over $V$, which we already know how to do from [[2026-09-01 - Neural Networks and Word Embeddings]]). It is also why *scoring* and *generating* are the same model: to generate, you sample from the same conditional you use to score.

**Worked example** — *"the cat sat on the mat"*:

$$
\begin{aligned}
P(\text{the cat sat on the mat}) = &\ P(\text{the}) \cdot P(\text{cat} \mid \text{the}) \cdot P(\text{sat} \mid \text{the cat}) \\
&\cdot P(\text{on} \mid \text{the cat sat}) \cdot P(\text{the} \mid \text{the cat sat on}) \\
&\cdot P(\text{mat} \mid \text{the cat sat on the})
\end{aligned}
$$

Note the chain rule is **exact** — no approximation has been made yet. The approximation arrives in §5.

---

## 4. Count-based modelling, and why it does not scale

The simplest possible estimator: count occurrences of sequences in a corpus.

$$P(sat \mid the\ cat) = \frac{count(the\ cat\ sat)}{count(the\ cat)} \qquad P(on \mid the\ cat\ sat) = \frac{count(the\ cat\ sat\ on)}{count(the\ cat\ sat)}$$

i.e. count how often the full sequence appears, and normalise by how often the prefix appears. Intuitively: *given that we have already seen "the cat sat", what fraction of the time does "on" come next?*

**The Menti question**: *do you see any problems with counting sequences?* The class raised — and the lecturer endorsed — cost of counting, processing time, long texts, unseen words, and the amount of data required. All of these are symptoms of one underlying fact:

> With a vocabulary of size $V$, the number of possible sequences of length $n$ is $V^n$ — **exponential in the sequence length**.

Concretely, with a typical $V \approx 40{,}000$ and sentences of length $\le 11$:

$$40{,}000^{11} \approx 4 \times 10^{50} \text{ sequences}$$

which is **more than the number of atoms in the Earth** ($\sim 10^{50}$). And 11 tokens is a *short* sentence. You cannot store this table, you cannot fill it, and no corpus in existence contains enough data to estimate its entries.

Note the shape of the problem: the vocabulary $V$ is large but finite and manageable; what kills you is the **exponent $n$**, not the base.

---

## 5. The Markov assumption and n-gram models

**Fix**: use only the **recent past** to predict the next token.

> **Effect**: reduce the number of estimated parameters, in exchange for modelling capacity.

That trade-off sentence is the whole idea. Without a restriction, the model's capacity is in principle infinite (arbitrarily long history), but unestimable; truncating the history makes estimation possible and throws information away.

- **1st order Markov**: $P(mat \mid the\ cat\ sat\ on\ the) \approx P(mat \mid the)$
- **2nd order Markov**: $P(mat \mid the\ cat\ sat\ on\ the) \approx P(mat \mid on\ the)$

**$k$-th order Markov**, in general — keep only the last $k$ tokens:

$$P(w_i \mid w_1 w_2 \ldots w_{i-1}) \approx P(w_i \mid w_{i-k} \ldots w_{i-1})$$

$$P(w_1 w_2 \ldots w_n) \approx \prod_i P(w_i \mid w_{i-k} \ldots w_{i-1}) \qquad \text{(ignore } w_j\ \forall j < 0\text{)}$$

The last parenthesis is just bookkeeping at the start of a sentence: for $i \le k$ the context is shorter than $k$, so you condition on whatever exists.

### n-gram models

| model | Markov order | formula |
|---|---|---|
| **Unigram** | 0 | $P(w_1, \ldots, w_n) = \prod_{i=1}^{n} P(w_i)$ |
| **Bigram** | 1 | $P(w_1, \ldots, w_n) = \prod_{i=1}^{n} P(w_i \mid w_{i-1})$ |
| **Trigram**, 4-gram, … | $n-1$ | $\prod_i P(w_i \mid w_{i-k} \ldots w_{i-1})$ |

Watch the off-by-one: an **n-gram model is an $(n-1)$-order Markov model**. A unigram model uses *no* history at all — it is a 0-order model that just reports how frequent each word is in the corpus.

> [!example] Lab — sentence-boundary tokens
> `ta_sessions/Exercises/03 - Language Modeling/1. Perplexity.ipynb` builds a bigram model by hand, and before counting it wraps every sentence as `<s> … </s>`. That is the practical answer to the "ignore $w_j$ for $j<0$" bookkeeping above, and the lecture didn't spell it out:
> - `<s>` gives the first word a real context, so the model learns which words **start** a sentence: $P(\text{the} \mid \texttt{<s>})$.
> - `</s>` lets the model assign probability to the sentence **stopping**. Without it, "the cat" and "the cat sat on the mat" can't be told apart as complete sentences.
>
> Both tokens are counted as vocabulary entries ($|V|$ includes them) and `</s>` is a predicted token, so a sentence of $m$ words gives $m+1$ bigrams to score.
>
> ⚠️ Not a universal convention: the mock-exam n-gram question says explicitly *"we don't consider adding any special tokens unless specifically stated"*. Read the question for this before you count.

> *Larger n leads to more accurate and better language models (but also higher costs).*
> **\*Caveat: assuming infinite data!**

That caveat is doing a lot of work and the lecturer repeated it: larger $n$ is only better **if you have enough data to estimate the parameters**. With a finite corpus, a larger $n$ makes the counts sparser (§11), so past some point it makes the model *worse*.

---

## 6. What n-grams actually produce

Sampling from models trained on the same corpus (Trump tweets, evidently):

| model | sample |
|---|---|
| **Unigram** | *release millions See ABC accurate President of Donald Will cheat them a CNN megynkelly experience @ these word out- the* |
| **Bigram** | *Thank you believe that @ ABC news, Mississippi tonight and the false editorial I think the great people Bill Clinton . "* |
| **Trigram** | *We are going to MAKE AMERICA GREAT AGAIN! #MakeAmericaGreatAgain https://t.co/DjkdAzT3WV* |

Read as a progression: **unigram** is just the corpus' most frequent words shaken in a bag — no local coherence at all. **Bigram** starts producing locally-plausible pairs ("ABC news", "Bill Clinton"). **Trigram** produces fluent multi-word chunks. Sampling is done as $w_i \sim \prod_{i=1}^{n} P(w_i \mid w_{i-k} \ldots w_{i-1})$.

### The wall: long-range dependencies

> Typical LMs are not sufficient to handle **long-range dependencies**.
>
> *"**Alice/Bob** could not go to work that day because **she/he** had a doctor's appointment"* — **requires at least a 9-gram model.**

To get the pronoun right, the model must still "see" the name 9 tokens back. A trigram model has no idea a name was ever mentioned. And by §11 we will see a 9-gram model is unestimable from any real corpus — so this sentence is simply outside what count-based LMs can do. That dead end is exactly what motivates neural LMs (§14) and RNNs (§16).

---

## 7. Evaluating a language model: the setup

A good LM assigns higher probability to typical, grammatically correct sentences: $P(a\ smooth\ finish) > P(a\ flat\ finish)$, $P(high\ school\ principal) > P(high\ school\ principle)$.

The research process is the standard ML pipeline:

- **Train** parameters on a suitable training corpus.
- **Assumption**: *observed sentences are probably good sentences.* This is what makes LM training **self-supervised** — there are no labels; you feed it text and the text itself is the supervision, exactly as in CBOW/skip-gram ([[2026-09-01 - Neural Networks and Word Embeddings]]). The assumption is load-bearing: garbage training text means the model learns that garbage is likely.
- **Test** on a different, unseen corpus.
- **Training on any part of the test set is not acceptable!!** The lecturer's analogy: it is like *having access to the book while writing the exam* — you are doing a lookup, not modelling. The measurement stops meaning anything.
- **Evaluate** with a suitable metric.

---

## 8. Extrinsic evaluation

Train LM → apply to a downstream task → observe accuracy → refine.

- **Pro**: directly optimised for downstream tasks; higher task accuracy → better model, by definition of what you actually want.
- **Con**: expensive and time consuming — a new expensive evaluation, and a new tuned model, for **every** task.
- **Con**: hard to optimise the downstream objective (indirect feedback).

The lecturer's sharper version of the last point, which is *not* on the slide: you end up training the model to be a better movie-review classifier or a better email writer, **not a better model of language**. It can take shortcuts — learn a few simple task-specific rules that get the accuracy up without ever modelling language in a general sense (see [1:34:10](https://au.cloud.panopto.eu/Panopto/Pages/Viewer.aspx?id=2ec320a2-fe53-47e8-a98f-b4bf00712f16&start=5650)). Since an LM is trained in a deliberately generic manner with no task objective, grading it on one task defeats the point.

So we want an **intrinsic** metric: one computed from the model's own probabilities, on held-out text, with no downstream task attached.

---

## 9. Perplexity

> A measure of how well a probability distribution (or LM) **predicts a sample**.

For a corpus $S$ with sentences $S^1, S^2, \ldots, S^n$:

$$ppl(S) = 2^x \quad \text{where} \quad x = -\frac{1}{W} \sum_{i=1}^{n} \log_2 P(S^i)$$

where **$W$ is the total number of words (tokens) in the test corpus**. The exponent $x$ is exactly the **cross-entropy** of the test corpus under the model — perplexity is its exponentiation, i.e. the **exponentiated token-level negative log-likelihood**.

Reading the formula piece by piece:

- $\log_2 P(S^i)$ — the log-probability the model assigns to sentence $i$. Logs turn the chain-rule *product* into a *sum*, which is both numerically stable (products of thousands of small probabilities underflow) and what makes the per-token average below meaningful.
- $-\frac{1}{W}\sum$ — negate (so lower is better) and average **per token**, so that corpora of different lengths are comparable.
- $2^{(\cdot)}$ — undo the log, bringing the number back onto a "number of choices" scale (see the intuition below).

For a **unigram model**, since $P(S) = \prod_j P(w_j)$, this collapses to $x = -\frac{1}{W}\sum_{i=1}^{n}\sum_{j=1}^{m} \log_2 P(w_j^i)$.

> **Minimising perplexity is the same as maximising the probability of the corpus** $P(S^1, S^2, \ldots, S^n)$.

That equivalence is why perplexity is not an arbitrary choice of metric: it *is* the training objective, read on a different scale.

### Intuition: perplexity = the model's uncertainty about the next token

Take an n-gram model that is maximally ignorant — it puts a uniform distribution over the vocabulary at every step:

$$P(w_i \mid w_{i-n}, \ldots, w_{i-1}) = \frac{1}{|V|} \quad \forall w_i$$

Then

$$ppl = 2^{-\frac{1}{W} \cdot W \cdot \log_2\left(\frac{1}{|V|}\right)} = |V|$$

i.e. **any word is equally likely at the next step**, and the perplexity equals the vocabulary size.

> **Intuition: perplexity measures a model's uncertainty about the next word** — how *surprised*, on average, the model is by the token that actually comes next.

The useful mental model: **perplexity ≈ the effective number of choices the model is agonising over at each step**. Perplexity 1 = fully confident (it knew exactly what was coming). Perplexity 2 = as uncertain as if it were choosing between 2 equally likely words. Perplexity 10 = as if choosing between 10. This is what makes perplexity interpretable in a way that raw cross-entropy (a number of bits) is not.

**Base of the logarithm**: theory uses base 2 (information theory, bits); in practice base $e$ is used all the time and *it doesn't matter* — it is a constant rescaling, as long as the exponentiation base matches.

### Two things the lecturer drew out of the Menti polls

**(a) Perplexity depends on the vocabulary, so it is not comparable across models with different vocabularies.** The uniform case above makes this obvious: $ppl = |V|$, so a model over a 100,000-word vocabulary and a model over a 1,000-word vocabulary are on completely different scales before you even look at their quality. Comparing their perplexities is *"literally comparing apples and oranges"* (see [1:46:40](https://au.cloud.panopto.eu/Panopto/Pages/Viewer.aspx?id=2ec320a2-fe53-47e8-a98f-b4bf00712f16&start=6400)). **Same vocabulary and same test set, or no comparison.**

**(b) What does $ppl > |V|$ mean?** Poll options were: better than a random LM / the same as a random LM / **worse than a random LM**. The answer is **worse than random**, and the reasoning is worth re-hearing (see [1:53:00](https://au.cloud.panopto.eu/Panopto/Pages/Viewer.aspx?id=2ec320a2-fe53-47e8-a98f-b4bf00712f16&start=6780)):

- A "random" LM = the uniform model, which by the computation above sits at exactly $ppl = |V|$. It is the *expectation* baseline — it is not faithful to the true distribution, but it is not actively wrong either.
- A model with $ppl > |V|$ is **systematically putting mass in the wrong places**: it is not merely uninformed, it is confidently mis-predicting. It behaves as if it were *deliberately* doing worse than chance.
- A *good* LM is faithful to the true distribution — if a word genuinely has probability 0.9 in context, it predicts ≈0.9. A random LM flattens everything; an above-$|V|$ model is worse than flat.

> [!example] Lab — what perplexity actually ranks
> `ta_sessions/Exercises/03 - Language Modeling/1. Perplexity.ipynb` scores sentences with the add-1 bigram model built in §12 (4 training sentences, $|V| = 16$ including `<s>`/`</s>`):
>
> | ppl | sentence | why |
> |---|---|---|
> | 8.03 / 8.37 | the two sentences copied from training | every bigram seen |
> | 8.93 | *the cat likes the mat* | one unseen bigram (*likes the*) |
> | 12.17 | *a soft cat likes the park* | 3 of 7 bigrams unseen |
> | 16.93 | *quantum physics is difficult* | 3 of 4 words not in the vocabulary |
>
> What it shows:
> - **The ranking just tracks overlap with the training data.** Seen bigrams get large numerators, unseen ones only get the $+1$.
> - **Even verbatim training sentences are nowhere near ppl 1.** With a corpus this tiny, adding $|V|$ to every denominator gives most of the mass to bigrams that never occurred.
> - **The last sentence lands at ≈ $|V|$, i.e. the uniform model above.** For a history word never seen, $C(w_{i-1}) = 0$ and add-1 collapses to $\frac{0+1}{0+|V|} = \frac{1}{|V|}$, a uniform guess. The notebook reads this as "no better than random". Strictly it's 16.93 > 16, so a bit *worse* than random, which is case (b) above: the two steps that leave a *seen* history get less than uniform, $P(\text{quantum} \mid \texttt{<s>}) = \frac{0+1}{4+16} = \frac{1}{20}$ and $P(\text{difficult} \mid \text{is}) = \frac{1}{1+16} = \frac{1}{17}$. A seen history spends its mass on the continuations it observed, so an unseen continuation gets less than $\frac{1}{|V|}$. (That extra step is my reading of the numbers; the notebook doesn't make it.)
> - Strictly, OOV words shouldn't get any probability at all. Real models map them to an `<UNK>` token, and the more words you map to `<UNK>`, the lower the perplexity, with no better model. That's point (a) again: **same test set, same vocabulary/tokenizer, or no comparison.**
>
> Note on the normaliser: the notebook divides by $N$ = **number of bigrams scored**, which includes the `</s>` prediction. That's the same $W$ as in the formula above once `</s>` counts as a token: you average over *predictions*, not over words typed.

---

## 10. Perplexity as a metric: pros and cons

| Pros | Cons |
|---|---|
| Easy to compute | Requires **domain match** between train and test |
| Standardised (roots in information theory) | Might not correspond to end-task optimisation |
| Directly useful — easy to use to correct sentences | $\log 0$ is **undefined** |
| Nice theoretical interpretation — matching distributions | Can be **"cheated"** by predicting common tokens |
| | Size of test set matters |
| | Can be sensitive to low-probability tokens/sentences |

Unpacking the ones that are not self-explanatory:

- **Domain match**: if train and test come from different distributions, a perplexity you optimised on training data tells you nothing about behaviour at test time — you are optimising a number you cannot trust.
- **$\log 0$**: one single unseen n-gram in the test corpus makes $P(\text{test corpus}) = 0$ and perplexity **undefined**. Not "large" — undefined. This is what §11–§12 are about.
- **Can be cheated**: a model that hedges toward frequent tokens ("the", "of", ",") scores well without understanding anything.
- **Size of test set matters**: a large test set dominated by easy sentences gives a beautiful perplexity for a model that is not doing what you want.

Despite all of this, the lecturer's closing point: perplexity **remains a central metric even for today's LLMs** — it is always tracked during pre-training. We come back to it in the LLM lectures.

---

## 11. Major shortcoming #1 of n-gram LMs: sparsity

### Exponential decay of counts

What happens as we lengthen the context? (Numbers from the deck, one corpus:)

| conditional | corpus frequency of the prefix |
|---|---|
| $P(w \mid \text{to})$ | *to* occurs 1M times |
| $P(w \mid \text{go to})$ | *go to* occurs 50,000 times |
| $P(w \mid \text{to go to})$ | 1,500 times |
| $P(w \mid \text{want to go to})$ | only 100 occurrences |

Each extra token of context cuts the count by roughly an order of magnitude — **counts decay exponentially in the prefix length**, which is the same $V^n$ blow-up from §4 seen from the data side rather than the storage side. And we routinely want information from **5+ words away** (recall the Alice/Bob example, which needs 9).

> **Slide inconsistency**: the third row is annotated "*go to* occurs 1500 times" but the conditional is $P(w \mid \text{to go to})$; it should read "*to go to* occurs 1500 times". The annotation on row 2 is the correct one for *go to*. The intended reading is clearly one prefix per row.

### Not all n-grams are in the training data

- Train on *Google news*, test on *Shakespeare*.
- $P(\text{affray} \mid \text{voice doth us}) = 0 \;\Rightarrow\; P(\text{test corpus}) = 0 \;\Rightarrow\;$ **undefined perplexity**.

One single zero anywhere in the test corpus poisons the entire product. The lecturer's framing: you would never guess that a model trained on something as general as Google News could fail — but the moment the test distribution shifts (Shakespeare, or news from a different country), the whole probability space changes.

### Fun fact — Shakespeare as a corpus

- $N = 884{,}647$ tokens, $V = 29{,}066$ word types.
- Shakespeare produced ~300K **bigram types** out of $V^2 \approx 844M$ possible bigrams.
- → **99.96% of possible bigrams were never seen** (zero entries in the table).

And that is for **bigrams**, the second-simplest model there is. Trigrams are hopeless by comparison.

### Why this is unavoidable: Zipf's law

![[assets/2026-09-08/lm-pt1-051.png]]
_Word frequency vs. rank in Romeo and Juliet: a handful of words ("and", "the", "I", "to") dominate, then a very long, very flat tail._

$$p(w_i) \propto 1/i \qquad (w_i = \text{the } i\text{-th most frequent word})$$

The probability of observing a word scales **inversely with its frequency rank** — a power law. There is a long tail of infrequent words, and **most finite-size corpora will have this problem**, no matter how big. You cannot buy your way out with more data: the tail just gets longer. (Same plot as in [[2026-08-25 - Introduction to NLP]], where it explained TF-IDF matrix sparsity — it is the same phenomenon reappearing one level up, at the level of sequences instead of single words.)

---

## 12. Smoothing

> Handle sparsity by making sure **all probabilities are non-zero** in our model.

Four families, in increasing sophistication:

- **Additive**: add a small amount to all probabilities.
- **Discounting**: redistribute probability mass from observed n-grams to unobserved ones.
- **Back-off**: use lower-order n-grams if the higher-order ones are too sparse.
- **Interpolation**: use a combination of different granularities of n-grams.

### The intuition

![[assets/2026-09-08/lm-pt1-054.png]]
_Left: sparse counts for $P(w \mid \text{denied the})$ — 4 words share all 7 observations, everything else is exactly 0. Right: shave a little mass off each observed count and pool it as "other", so unseen continuations get a small but non-zero probability._

The slogan from the slide: **"steal probability mass to generalise better."** You are deliberately making the model fit the training data slightly *worse* in exchange for it not being catastrophically wrong on anything it has not seen — the classic bias/variance trade.

### Laplace smoothing (add-$\alpha$)

The simplest heuristic: **add $\alpha$ to all counts and renormalise.**

Maximum-likelihood estimate for bigrams:

$$P(w_i \mid w_{i-1}) = \frac{C(w_{i-1}, w_i)}{C(w_{i-1})}$$

After smoothing:

$$P(w_i \mid w_{i-1}) = \frac{C(w_{i-1}, w_i) + \alpha}{C(w_{i-1}) + \alpha|V|}$$

Note the denominator: you added $\alpha$ to the count of **every one of the $|V|$ possible continuations**, so the normaliser must grow by $\alpha|V|$ for the distribution to still sum to 1. Getting this term wrong is the standard mistake.

> [!example] Lab — a bigram LM by hand, from zero to perplexity
> `ta_sessions/Exercises/03 - Language Modeling/1. Perplexity.ipynb`. Training corpus: *the cat sat on the mat* / *the dog ran in the park* / *a cat likes fish* / *the mat is soft*, each wrapped in `<s> … </s>` (see §5), so $|V| = 16$. The test sentence is *the cat likes the mat*. It never appears in training, but all of its words do.
>
> **MLE breaks.** $P_\text{MLE}(\text{the} \mid \text{likes}) = 0/1 = 0$, because *likes the* never occurs, even though it's a perfectly normal phrase. One zero factor makes the whole sentence probability 0. In log space it's $\log_2 0 = -\infty$ (Python's `math.log2(0)` raises `ValueError`), so the perplexity is infinite, and the sentence can't be scored at all. That's §10's "$\log 0$ is undefined" in one line of code.
>
> **Add-1 fixes it.** Every step of the test sentence:
>
> | bigram | $\frac{C(w_{i-1},w_i)+1}{C(w_{i-1})+16}$ | $P$ |
> |---|---|---|
> | $\texttt{<s>} \to$ the | $(3+1)/(4+16)$ | 0.200 |
> | the → cat | $(1+1)/(5+16)$ | 0.095 |
> | cat → likes | $(1+1)/(2+16)$ | 0.111 |
> | likes → the | $(0+1)/(1+16)$ | 0.059 |
> | the → mat | $(2+1)/(5+16)$ | 0.143 |
> | mat → `</s>` | $(1+1)/(2+16)$ | 0.111 |
>
> $\text{ppl} = 2^{-\frac{1}{6}\sum \log_2 P} \approx 8.93$, over $N = 6$ predictions (5 words + `</s>`).
>
> The notebook's sanity check is worth keeping: for a fixed $w_{i-1}$, $\sum_{w} P_\text{add-1}(w \mid w_{i-1})$ must equal 1 **over the whole vocabulary**. If it doesn't, the $+|V|$ in the denominator is wrong (or $|V|$ was counted without the boundary tokens).

> ⚠️ **Exam — this exact computation is in the 2025 exam and in the mock exam** (same question, `exams/exam_2025_with_solutions.md` and `ta_sessions/Exam/mock_exam.ipynb`). Six sentences (*the cat sat on the mat*, *she sat on the sofa*, …), $|V| = 18$, **no special tokens**, add-one smoothing. Asked: $P(\text{sat} \mid \text{on})$ and $P(\text{on} \mid \text{sat})$.
> - *sat → on* occurs **2** times, *on → sat* **0** times, $C(\text{sat}) = C(\text{on}) = 2$.
> - $P(\text{on} \mid \text{sat}) = \frac{2+1}{2+18} = \frac{3}{20}$, and $P(\text{sat} \mid \text{on}) = \frac{0+1}{2+18} = \frac{1}{20}$.
> - The trap is the **direction**: $P(A \mid B)$ counts the bigram $(B, A)$, i.e. *B followed by A*. The wrong option simply swaps the two values.

### Back-off

If the long prefix was never seen, fall back to a shorter one: if $C(\text{want to go to}) = 0$, estimate using $P(w \mid \text{to go to})$, and if that fails, $P(w \mid \text{go to})$, and so on. You use the longest context that has enough support.

### Linear interpolation

Rather than picking *one* granularity, combine them all:

$$\hat{P}(w_i \mid w_{i-1}, w_{i-2}) = \lambda_1 \underbrace{P(w_i \mid w_{i-1}, w_{i-2})}_{\text{trigram}} + \lambda_2 \underbrace{P(w_i \mid w_{i-1})}_{\text{bigram}} + \lambda_3 \underbrace{P(w_i)}_{\text{unigram}}$$

$$\sum_i \lambda_i = 1$$

The $\lambda_i$ are **hyperparameters tuned on a validation set** (or learned from data). The constraint $\sum \lambda_i = 1$ is what keeps the result a valid probability distribution. **Strong empirical performance** — this is the smoothing method that actually gets used.

### Wrap-up of the n-gram half

- n-gram LMs are **simple, effective** methods for estimating the probability of sequences.
- **Problem #1**: the number of sequences to model grows **exponentially** with sequence length $n$.
- **Problem #2**: smoothing **heuristics** must be used to estimate the probability of sequences unseen during training.

The word *heuristics* is deliberate: none of §12 is principled. It patches the symptom (zeros) without touching the cause.

---

## 13. Major shortcoming #2: no notion of similarity

This is the deeper problem, and the one that actually motivates neural LMs.

> With n-gram LMs, there is **no notion of similarity unless sequences overlap**. We treat all words / prefixes as **independent of one another!**

```
students opened their ___          undergraduates opened their ___
pupils opened their ___            students turned the pages of their ___
scholars opened their ___          students attentively perused their ___
```

Every one of these prefixes means essentially the same thing and should lead to the same prediction (*books*, *laptops*, …). A count-based model sees six unrelated strings and estimates six unrelated distributions — whatever it learned about "students opened their" contributes **nothing** to "pupils opened their".

> **Ideally, we should share information across these same prefixes!**

This is *exactly* the same failure as bag-of-words in [[2026-08-25 - Introduction to NLP]] — where two documents were similar only if they shared literal words — and exactly the failure that dense embeddings fixed in [[2026-09-01 - Neural Networks and Word Embeddings]]. The lecturer drew the parallel explicitly: with counting, "cut" and "slice" are simply different symbols, and no amount of data changes that. The fix will be the same fix: **stop counting symbols, start composing dense vectors.**

---

## 14. Fixed-context neural language models

Recall the neural NLP pipeline: embeddings → model composes them into $h_T = f(X)$ → prediction. For language modelling, **predict the next word from the composed prefix representation.**

### From a vector back to a distribution over words

![[assets/2026-09-08/lm-pt1-064.png]]
_A low-dimensional representation of "students opened their" is projected into a probability distribution over the entire vocabulary; "books" and "laptops" get the mass._

**Vocab projection + softmax layers** convert a vector representation into a probability distribution over the entire vocabulary. Given a $d$-dimensional representation $\mathbf{x}$ of a prefix:

1. **Project** it to a $V$-dimensional vector with a matrix–vector product (a "linear layer" / "feedforward layer"), $V$ = vocabulary size.
2. **Apply softmax** to turn the resulting vector into a probability distribution.

This is exactly the projection+softmax machinery from CBOW/skip-gram ([[2026-09-01 - Neural Networks and Word Embeddings]]) — same operation, different input: there it was a context window, here it is a prefix.

### The architecture

![[assets/2026-09-08/lm-pt1-068.png]]
_Fixed-context neural LM computing $P(\text{mat} \mid \text{the cat sat on the})$: five embeddings are **concatenated**, passed through one hidden layer, then projected to $|V|$ and softmaxed._

With a fixed context of $n = 5$:

- **Input layer**: $\mathbf{x} = [\mathbf{e}_{the}; \mathbf{e}_{cat}; \mathbf{e}_{sat}; \mathbf{e}_{on}; \mathbf{e}_{the}] \in \mathbb{R}^{dn}$ — the embeddings are **concatenated (stacked)**, not summed.
- **Hidden layer**: $\mathbf{h} = \tanh(\mathbf{W}\mathbf{x} + \mathbf{b}) \in \mathbb{R}^{h}$
- **Output layer (softmax)**: $\mathbf{z} = \mathbf{U}\mathbf{h} \in \mathbb{R}^{|V|}$, and
$$P(w = i \mid \text{the cat sat on the}) = \text{softmax}_i(\mathbf{z}) = \frac{e^{z_i}}{\sum_k e^{z_k}}$$

**Menti: dimensionalities.** $\mathbf{W} \in \mathbb{R}^{h \times dn}$ (here $h \times 5d$) and $\mathbf{U} \in \mathbb{R}^{|V| \times h}$. These are the **only** learned parameters besides the embedding table — the model's size depends on $d$, $h$, $n$ and $|V|$, **not** on the number of sequences.

**Why concatenate instead of sum?** Summing (as in CBOW) destroys word order — *"the cat sat"* and *"sat the cat"* would get the same representation. Concatenating keeps each position in its own slice of the input vector, so $\mathbf{W}$ can learn position-specific behaviour. The price is that $\mathbf{W}$ grows linearly with the context length $n$.

> **Could the hidden layer just learn to do a sum?** A student asked this, and the answer is **yes — it can**: a suitable $\mathbf{W}$ reproduces a sum exactly. But then you would be "wasting compute to learn an operation you could have hard-coded". The point of the learned layer is that it discovers whichever composition suits your **data and task**; if summing happens to be optimal for your data, backprop will find something like it, but you don't have to commit to it in advance (see [2:44:55](https://au.cloud.panopto.eu/Panopto/Pages/Viewer.aspx?id=2ec320a2-fe53-47e8-a98f-b4bf00712f16&start=9895)).

**Menti: which word-embedding algorithm does this resemble most?** → **CBOW.** Same shape — take surrounding/preceding words as input, predict a single target word through a projection + softmax. The difference is *which* words count as context (CBOW: symmetric window; fixed-context LM: only the preceding $n$) and that the composition is a learned layer rather than a plain sum.

---

## 15. Fixed-context NLMs: advantages and limits

### Advantages

- **No more sparsity problem.** All sequences can be estimated with **non-zero probability**.
    - *Why?* Because softmax, $\frac{e^{a_i}}{\sum_j e^{a_j}}$, is **strictly positive for every $i$**, for any finite input vector. There is no counting step, so there is no way to land on an exact zero, so $\log 0$ can never occur and perplexity is always defined (see [2:52:20](https://au.cloud.panopto.eu/Panopto/Pages/Viewer.aspx?id=2ec320a2-fe53-47e8-a98f-b4bf00712f16&start=10340)). Every smoothing heuristic of §12 becomes unnecessary — not patched, but structurally impossible to need.
- **Model size is much smaller.** It depends on the **number of weights**, not on the number of sequences. The n-gram table grew as $V^n$; here the parameter count is fixed by $d, h, n, |V|$.
- **Information is shared across similar prefixes**, which was the §13 complaint: "students opened their" and "pupils opened their" produce nearby input vectors because *students* and *pupils* have nearby embeddings, so what the model learns about one transfers to the other for free.

### Disadvantages

- **Fixed windows are still too small to encode long-range dependencies.** The Alice/Bob problem is unsolved.
- **Enlarging the window makes $\mathbf{W}$ larger** (more computationally expensive) — $\mathbf{W} \in \mathbb{R}^{h \times dn}$ grows linearly in $n$, and the hidden layer may not even have the capacity to usefully transform a 1000-word concatenation.
- **Weights in $\mathbf{W}$ aren't shared across embeddings in the window** (computationally inefficient). Each position gets its own block of $\mathbf{W}$, so what the model learns about a word appearing in position 2 does not transfer to the same word in position 4. Sharing is what lets you learn more from less.

### And the killer: sequences are not of consistent length

```
The cat chased the mouse
The starving cat chased the mouse
The starving cat fanatically chased the mouse
The starving cat fanatically chased the elusive mouse
The starving cat, who had not eaten in six days, fanatically chased the elusive mouse
```

The *cat–mouse* relationship is the same in all five, but the **distance** between them keeps changing.

![[assets/2026-09-08/lm-pt1-074.png]]
_A 5-word fixed context handles the first four prefixes, but once "cat" falls outside the window the model simply cannot see it._

A fixed-context model would need a **different model per context length** — or one enormous model sized for the worst case, wasteful for every ordinary sentence. Real sequences are never of fixed length, so this is not an edge case; it is the normal case.

---

## 16. Recurrent neural networks (RNNs) — introduction

> **Solution**: recurrent neural networks — NNs with **feedback loops**.

![[assets/2026-09-08/lm-pt1-076.png]]
_The compact ("folded") view: an input $x_t$ enters the unit, the unit maintains a state $h_t$ fed back into itself, and emits an output $z_t$._

![[assets/2026-09-08/lm-pt1-077.png]]
_**Unrolling** the RNN across all time steps gives the full computation graph. Allows learning from the entire sequence history, regardless of length._

The two views are the same network: the folded one is how you *draw* it in a paper, the unrolled one is how it actually *executes* and how backpropagation sees it. At each step $t$ the unit receives the current input $x_t$ **and** the state $h_{t-1}$ accumulated from everything before it, and produces (i) an output $z_t$ for this step and (ii) a new state $h_t$ passed forward.

**Why this fixes §15**: the history is carried in a *fixed-size state vector* instead of a growing concatenation, so the model does not care how long the sequence is and does not grow with it.

### Putting a classifier on top

![[assets/2026-09-08/lm-pt1-083.png]]
_The classifier is just an output projection followed by a softmax — exactly the same prediction head as in every model so far._

$$\text{Binary: } P(y) = \sigma(W_o z_T) \qquad \text{Multi-class: } P(y) = \text{softmax}(W_o z_T)$$

> **Question on the slide: why use the output of the *last* recurrent unit to predict a label?**
>
> Because we read text **left to right**, treating position as temporal order — so the last state is the only one that has accumulated information about the *entire* sequence. Any earlier state has only seen a prefix (see [3:06:20](https://au.cloud.panopto.eu/Panopto/Pages/Viewer.aspx?id=2ec320a2-fe53-47e8-a98f-b4bf00712f16&start=11180)).

> [!example] Lab — the last state, and what padding does to it
> `ta_sessions/Exercises/04 - RNNs/0. RNNs.ipynb` builds exactly this model for news-topic classification (AG News): `nn.Embedding` → `nn.RNN` → `nn.Linear` on the **final hidden state**, raw logits, with the softmax left to the cross-entropy loss. PyTorch's `nn.RNN` returns two tensors, and it's easy to mix them up (the notebook's MCQ 6.6 tests it):
> - `output`, shape `(B, L, H)`: $h_t$ at **every** step. That's what sequence labelling uses.
> - `hidden`, shape `(1, B, H)`: only $h_T$. It equals `output[:, -1, :]`, and that's what the classifier reads.
>
> A batch needs equal-length sequences, so short texts get `[PAD]` tokens, and **the RNN does not skip them**. `padding_idx` only makes the `[PAD]` embedding zero, so on a pad step the update becomes $h_t = \tanh(W_{hh}h_{t-1} + b)$. The state keeps getting rewritten with no input, 30 times in a row for a 20-token text padded to 50. **Padded on the right**, the "last state" is a washed-out, article-independent vector, which defeats the whole point of reading it. The notebook's fix is to **pad on the left**: the idle steps happen *before* the text, and $h_T$ is computed right after the last real token. (The general fix is `pack_padded_sequence`, which really skips the pads.)
>
> The vanilla RNN suffers most because it rewrites its whole state at every step. Gated models can learn to carry it across; see [[2026-09-15 - Language Modeling (Part 2)]].
>
> Small mismatch with the slide: `nn.RNN` uses $\tanh$ as the default nonlinearity, where the Elman formula below uses $\sigma$. Same structure.

You can attach a classifier at **any** step; which steps you use is determined by the task:

![[assets/2026-09-08/lm-pt1-085.png]]
_**Sequence labelling** (e.g. part-of-speech tagging): one classifier per time step, because each token needs its own label._

![[assets/2026-09-08/lm-pt1-086.png]]
_**Text generation**: the token predicted at step $t$ is **fed back in** as the input at step $t+1$, and the loop continues until `<EOS>`._

That feedback loop in generation is worth noticing: it is the chain rule of §3 made operational — each step conditions on everything generated so far.

**Results** shown: character-level RNNs generating passable Shakespeare (correct speaker headings, verse layout, plausible-but-meaningless lines) and Wikipedia markup (valid `[[links]]`, headings, URLs). Locally fluent, globally nonsense — but a world away from the trigram samples in §6.

### The classical RNN: the Elman network

![[assets/2026-09-08/lm-pt1-093.png]]
_The Elman network, with the two components that make it work highlighted._

$$h_t = \sigma(W_{hx} x_t + W_{hh} h_{t-1} + b_h)$$
$$z_t = \sigma(W_{zh} h_t + b_z)$$

**Two major components:**

1. **Memory** is maintained across time steps by the state $h$ — specifically by the $W_{hh} h_{t-1}$ term, which is the slide's answer to *"what part of this formulation allows the model to maintain a state of previously input tokens?"* Delete that one term and you have an ordinary feedforward network applied independently per token.
2. **Parameter matrices are shared across all time steps** — $W_{hx}, W_{hh}, W_{zh}, b_h, b_z$ are the *same* at every $t$. This is precisely the weight sharing the fixed-context model lacked (§15), and it is why **model size does not grow with sequence length**.

> **What should $h_0$ be?** The question was left open here and settled at the start of the next lecture. Any of these works: **zeros** (there's no history yet), something **task-dependent/pre-trained**, or a **learnable** parameter trained with everything else. See [[2026-09-15 - Language Modeling (Part 2)]].

**Objective**: learn the parameter matrices $W_{hh}, W_{hx}, W_{zh}$ such that the RNN can represent textual sequences.

### Wrap-up

- Neural LMs let us **share information among similar sequences** by learning representations that represent them similarly.
- **Problem**: fixed-context LMs can only process a limited window of the word history at a time.
- **Solution**: RNNs can **theoretically** learn to model an **unbounded context length**, with **no increase in model size**, because weights are shared across time steps.

The word **theoretically** is flagged in the slide and was flagged aloud: in practice training RNNs over long sequences runs into vanishing/exploding gradients. **Challenges in training RNNs** and **mitigations (LSTMs, GRUs)** are next week's lecture.

---

## Da ricordare

- **LM = probabilistic model of a sequence of tokens.** Chain rule: $P(w_1 \ldots w_n) = \prod_i P(w_i \mid w_1 \ldots w_{i-1})$; the conditioning context is the **prefix**. This is exact — the approximation is the Markov assumption.
- **n-gram = $(n-1)$-order Markov.** Unigram = order 0. Larger $n$ is better *only assuming infinite data*.
- **Why counting fails**: $V^n$ sequences of length $n$; with $V \approx 40$K and $n \le 11$ that is $> 4 \times 10^{50}$. And counts decay exponentially in prefix length (Zipf), so 99.96% of Shakespeare's possible bigrams never occur.
- **Perplexity**: $ppl(S) = 2^x$, $x = -\frac{1}{W}\sum_i \log_2 P(S^i)$, $W$ = **total tokens in the test corpus**. $x$ is the cross-entropy. **Minimising ppl = maximising corpus probability.** For a uniform model $ppl = |V|$ — hence $ppl$ is roughly *"how many equally-likely choices is the model torn between"*.
- **Perplexity traps**: (a) **never compare perplexities across different vocabularies or test sets**. ⚠️ Exam 2025 / mock: *ppl 20 on dataset A, 40 on B* → "cannot be compared", **not** "twice as good on A"; (b) $ppl > |V|$ means **worse than a random model**; (c) one unseen n-gram → $P = 0$ → $\log 0$ → **undefined**, not "large".
- **Smoothing**: additive / discounting / back-off / interpolation. Laplace add-$\alpha$ bigram: $P(w_i \mid w_{i-1}) = \frac{C(w_{i-1}, w_i) + \alpha}{C(w_{i-1}) + \alpha|V|}$ — remember the $\alpha|V|$ in the denominator. ⚠️ Exam 2025 / mock: $P(A \mid B)$ counts the bigram **$(B, A)$**. Swapping the direction is the wrong option ($P(\text{on} \mid \text{sat}) = \frac{2+1}{2+18} = \frac{3}{20}$, $P(\text{sat} \mid \text{on}) = \frac{1}{20}$). Check whether `<s>`/`</s>` count: the exam says **no special tokens** unless stated. Linear interpolation needs $\sum_i \lambda_i = 1$.
- **The two shortcomings of n-gram LMs**: (1) exponential growth / sparsity, patched only by heuristics; (2) **no notion of similarity unless sequences overlap** — prefixes are treated as independent symbols. (2) is the one neural LMs actually fix.
- **Fixed-context NLM**: $\mathbf{x} = [\mathbf{e}_1; \ldots; \mathbf{e}_n] \in \mathbb{R}^{dn}$ (**concatenation**, to preserve order), $\mathbf{h} = \tanh(\mathbf{W}\mathbf{x} + \mathbf{b})$, $\mathbf{z} = \mathbf{U}\mathbf{h} \in \mathbb{R}^{|V|}$, softmax. $\mathbf{W} \in \mathbb{R}^{h \times dn}$, $\mathbf{U} \in \mathbb{R}^{|V| \times h}$. Resembles **CBOW**. No sparsity **because softmax is always strictly positive**.
- **Fixed-context limits**: window too small for long-range dependencies; $\mathbf{W}$ grows with $n$; **weights not shared across window positions**; sequences aren't of fixed length.
- **Elman RNN**: $h_t = \sigma(W_{hx}x_t + W_{hh}h_{t-1} + b_h)$, $z_t = \sigma(W_{zh}h_t + b_z)$. Two components: **memory via $W_{hh}h_{t-1}$**, and **parameters shared across all time steps** (→ no size growth, unbounded context *in theory*). Predict from the **last** state because text is read left-to-right and only the last state has seen everything. In PyTorch that's `hidden.squeeze(0)` = `output[:, -1, :]`. Pad on the **left**: the RNN doesn't skip pad steps, and right padding would overwrite $h_T$.
