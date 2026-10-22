The Transformer of [[2026-09-22 - Seq2Seq Models and Transformers]] is an architecture. This lecture is about what happened when people **pretrained** it on huge amounts of raw text and then reused it for everything. There are three families, one for each part of the Transformer you keep:

| family | keeps | pretraining objective | example | good for |
|---|---|---|---|---|
| **decoder-only** | the decoder (masked self-attention, no cross-attention) | **causal LM**: predict the next token | **GPT** | generating text |
| **encoder-only** | the encoder (full self-attention) | **masked LM**: predict hidden tokens | **BERT** | classification, sequence labelling |
| **encoder–decoder** | both, with cross-attention | **denoising**: rebuild a corrupted input | **BART**, **T5** | both, including generation |

Plus one non-Transformer step in between: **ELMo**, contextual embeddings from bidirectional LSTMs.

The deck (`06 - PLMs.pdf`) first finishes **position embeddings**, which were left over from last week (§1), and the price of Transformers (§2). The PLM part starts at slide 31.

---

## 1. Position embeddings

### Why they're needed

Self-attention compares tokens **in parallel**, as a set. Unlike an RNN, there is no left-to-right recurrence that tells the model which token came first. *dog bites man* and *man bites dog* contain the same tokens, so without positions self-attention builds the same representation for each word (the lab demonstration is in [[2026-09-22 - Seq2Seq Models and Transformers#4.7 Position embeddings (preview — covered next lecture)]]).

> **Self-attention knows which tokens interact. Positional information tells it where they are in the sequence.**

So each token's input gets two parts:

- the **token embedding** $e_i$: *what* the token is ("cat"),
- the **position vector** $p_i$: *where* it is ("position 7"),

and the Transformer sees $x_i = e_i + p_i$.

### Addition vs concatenation

![[appunti/assets/2026-09-29/plms-020.png]]

| | addition $x_i = e_i + p_i$ | concatenation $[e_i; p_i]$ |
|---|---|---|
| dimensionality | stays $d$ | doubles to $2d$ |
| rest of the Transformer | unchanged | every downstream weight matrix changes size |
| content vs position | still separable: $W_Q(e_i + p_i) = W_Q e_i + W_Q p_i$, a "what" part and a "where" part | separate by construction, but this is "usually unnecessary" |

**Addition is what's used.** The lecturer was candid that the reason is partly empirical. For bidirectional LSTMs ([[2026-09-15 - Language Modeling (Part 2)#5. Bidirectional RNNs]]) people tried both, concatenation worked better, and that's what got used. For position embeddings addition worked better. You could fix concatenation's size problem with a learned projection, as for the bidirectional encoder ([[2026-09-22 - Seq2Seq Models and Transformers#Bidirectional encoders]]). Several options are plausible, and you keep what works: "sometimes deep learning is a dark art" (see [19:28](https://au.cloud.panopto.eu/Panopto/Pages/Viewer.aspx?id=fc85844c-5719-4381-b9c9-b4d400774366&start=1168)).

### Sinusoidal encodings: many clocks

![[appunti/assets/2026-09-29/plms-022.png]]

The original Transformer used fixed **sinusoids**:

$$PE(pos, 2i) = \sin\!\left(\frac{pos}{10000^{2i/d}}\right) \qquad PE(pos, 2i+1) = \cos\!\left(\frac{pos}{10000^{2i/d}}\right)$$

The intuition from the slide: **every dimension is a clock hand** turning at its own speed.

- **Fast** clocks (low dimensions) change a lot from one position to the next, so they **tell nearby positions apart**.
- **Slow** clocks (high dimensions) barely move between neighbours but tell you the **broad location** in the sequence.
- A position is the **set of readings of all the clocks** at once, the way a time is read off hour, minute and second hands together.

**Why 10,000?** A student asked. The lecturer's answer: "I don't know", it's the setting from the original paper. It has to be large to get the range from very fast to very slow clocks, and very long documents might need more (see [24:12](https://au.cloud.panopto.eu/Panopto/Pages/Viewer.aspx?id=fc85844c-5719-4381-b9c9-b4d400774366&start=1452)).

### Learned position embeddings

> **In practice, the easiest thing is to learn position embeddings from scratch.**

There's one trainable vector per position, as an `nn.Embedding(max_len, d)`. The sinusoids are a nice hand-designed choice, but they may not suit every corpus, so "put a weight matrix and learn the crap out of it".

**Menti: what's a disadvantage of learned position embeddings?**

> **Poor generalization to sequences longer than the maximum position embedding you have learned.**

If the longest training sequence had $n$ tokens, **there is no vector for position $n+1$**. The model has never seen it, so it can't have learned it. The lecturer also went through the other answers (see [31:01](https://au.cloud.panopto.eu/Panopto/Pages/Viewer.aspx?id=fc85844c-5719-4381-b9c9-b4d400774366&start=1861)):

- "**Not parallelizable**": wrong. Position vectors don't depend on each other, so nothing sequential is involved.
- "**Need more training / computation**": yes, but only in the trivial sense that there are more parameters.
- "Only works for the corpora they're trained on", "hard to allow indefinite context windows": the same point as the correct answer.
- "Might learn more than just position": valid, and harmless. Anything extra they learn is "a byproduct for free".

> [!example] Lab — five ways to encode position, compared (`06 - Pretraining/2. Positional Encodings.ipynb`)
> Learned and sinusoidal are **absolute**: a vector for "position 17" is added to the input. Today's LLMs mostly encode **relative** position *inside* attention instead:
>
> | method | where | what | used in |
> |---|---|---|---|
> | none (**NoPE**) | — | nothing; only the causal mask carries order | some layers of recent models |
> | learned | input | a trained vector per position | BERT, GPT-2 |
> | sinusoidal | input | fixed sin/cos vector per position | original Transformer |
> | **RoPE** | attention | **rotate** queries and keys by an angle ∝ position | Llama, Mistral, Qwen, Gemma |
> | **ALiBi** | attention | add a penalty $-m_h\,(i-j)$ ∝ distance to the scores | BLOOM, MPT |
>
> - **RoPE:** rotating $q$ by $m\theta$ and $k$ by $n\theta$ gives $\langle R_{m\theta}q, R_{n\theta}k\rangle = \langle q, R_{(n-m)\theta}k\rangle$. The score depends **only on the distance $n-m$**. Only Q and K are rotated, never V.
> - **ALiBi:** no parameters. Each head has a fixed slope ($2^{-8h/H}$), which gives steep heads a short recency window and flat heads a long one. It **extrapolates** well ("train short, test long") because far tokens are penalised so much that a long sequence looks like a short one locally. The price: it can only say "closer is better". It is bad at using distant information and can't express "exactly 5 tokens back".
> - **NoPE works in a decoder** because the **causal mask** treats positions differently: position $t$ attends to exactly $t+1$ tokens, and later layers can read that signal. An encoder (no mask) without positions really is order-blind.
>
> Results on a 4× longer test sequence than in training: learned = impossible (no vector), **sinusoidal = blows up** (being computable for any position ≠ generalising to it), RoPE = degrades (fixable with **position interpolation**, i.e. squeezing positions back into the trained range, plus a short fine-tune), ALiBi = flat. On a "copy the token 5 steps back" task, RoPE solves it at any length, ALiBi and NoPE don't.

---

## 2. Transformers: a new era, and their price

The recap of last week: the **temporal bottleneck** (vanishing gradients stop RNNs from learning long-range dependencies) is fixed by **attention**; the **parallelisation bottleneck** (each RNN state needs the previous one) is fixed by **self-attention**; and self-attention needs **position embeddings** to know the order.

**Why "beginning of a new era":** before 2017, the best MT systems were each a specialised, heavily engineered architecture (GNMT, ConvS2S, mixtures of experts…). The base Transformer matched or beat them on BLEU (higher = better; it comes back in the NLG lecture) with **no task-specific tricks**, at a fraction of the training cost (EN–DE: $3.3\cdot10^{18}$ FLOPs vs $10^{19}$–$10^{20}$), because it parallelises.

**Menti: what could be a disadvantage of Transformers over RNNs?**

> **Time complexity: quadratic in the sequence length (Transformers) vs linear (RNNs).**

Every token attends to every other, so a layer computes $T^2$ scores. The lecturer added that this is quadratic **in the sequence length** "and nothing else". Other answers, discussed (see [38:27](https://au.cloud.panopto.eu/Panopto/Pages/Viewer.aspx?id=fc85844c-5719-4381-b9c9-b4d400774366&start=2307)):

- "**Overfitting**": not specific to Transformers, it applies to every architecture. What's partly true is that Transformers **need more data** (more parameters, more pairwise associations to learn). That comes with scaling laws in later lectures.
- "**Fixed sequence size**": that's a property of *learned* position embeddings, not of Transformers. With sinusoidal (non-learned) encodings it "goes away".

> ⚠️ The lab qualifies that last point. Sinusoidal encodings are *defined* for every position, but a model that never saw positions ≥ 64 still breaks on them (lab box in §1). Transformers in practice do have a context length.

### Sparse attention

![[appunti/assets/2026-09-29/plms-030.png]]

Fixing the quadratic cost is still an active research line, called **sparse attention**. You don't need *every* pair of tokens:

- **Global:** a few **hub** tokens attend to (and are attended by) everything, which gives a global summary.
- **Local:** each token attends only to its neighbours, because in language **local context matters most**. This is the same assumption as word2vec's window ([[2026-09-01 - Neural Networks and Word Embeddings]]).
- **Random:** a few random links. The intuition is from network science: **small-world networks**, where a few random long-range links make any two nodes reachable in a handful of hops ("six degrees of separation").
- **BigBird** (Google) = global + local + random. It is cheaper than full attention and worked as well (see [39:37](https://au.cloud.panopto.eu/Panopto/Pages/Viewer.aspx?id=fc85844c-5719-4381-b9c9-b4d400774366&start=2377)). In class it came out as "BigBERT"; the model on the slide is BigBird.

> **Aside (not in the syllabus):** since ~2021–22, people have revisited RNN-like architectures that are linear in $T$: **state-space models (SSMs)**, e.g. **Mamba**. The lecturer mentioned the terms for anyone curious and said they won't be covered.

---

## 3. GPT: Generative Pretrained Transformer

### Architecture

- Called a **decoder** transformer, but the block **mixes** the two halves of the original Transformer:
  - from the **decoder**: **masked** multi-head self-attention. It generates, so a token can't attend to future tokens.
  - from the **encoder**: **no cross-attention**. There is no encoder to attend to, only self-attention over its own history.
- 12 blocks. Input = text embedding + position embedding (learned).

### Pretraining

![[appunti/assets/2026-09-29/plms-035.png]]

- **Data:** the Toronto **BooksCorpus**, ~7,000 unpublished books (~13 GB). In the lecturer's words, "back in the day these companies were nice" and used open data.
- The corpus is cut into **windows of 512 tokens**, so GPT can model context up to 512 tokens. In June 2018 that was a lot.
- **Objective: next-token prediction**, also called **causal language modelling**. "Causal" because the prediction conditions only on the past:
  $$\mathcal L = -\sum_{t=1}^{T} \log P\big(x^*_t \mid \{x^*_s\}_{s<t}\big)$$
  the negative log-likelihood of the **gold** sequences in the dataset.
- **Targets = inputs shifted by one**, with `[END]` appended. With the causal mask, all positions are trained **in parallel** in one pass. This is teacher forcing again ([[2026-09-22 - Seq2Seq Models and Transformers#Training]]), with input and target now coming from the *same* text. No one has to label anything. This is the objective all of today's LLMs are still trained on.

### Fine-tuning

The pretrained model is the **initialisation** for training on each task. You append a special token (`[CLS]` on the slide) at the **end** of the input, put a **linear layer** on its final representation, and minimise the cross-entropy of the gold label:

$$\mathcal L = -\log P(y^* \mid X)$$

The model stays the same. **Only the way you format the input changes:**

![[appunti/assets/2026-09-29/plms-039.png]]

| task | input format | why |
|---|---|---|
| classification (e.g. movie review → positive/negative) | `[START] text [CLS]` → linear | one text, one label |
| **entailment** (premise / hypothesis → entailed, contradicted, neutral) | `[START] premise [DELIM] hypothesis [CLS]` | two texts: a **delimiter token** separates them |
| **similarity** ("Akhil was giving a lecture" / "Akhil was teaching") | **both orders**, `text1 [DELIM] text2` and `text2 [DELIM] text1`, combined → linear | similarity is **symmetric**, but a left-to-right model reads the two orders differently, so feed both |
| **multiple choice** | `context [DELIM] answer_k [CLS]` for each answer → linear → softmax over answers | the gold distribution is one-hot on the right answer; the model learns a distribution over the answers |

**Massive improvements (back then):** one pretrained model, fine-tuned per task, beat the specialised state of the art on most of the tasks in the paper's table (entailment, similarity, QA, commonsense reasoning…). Before, every task had its own purpose-built model. That was the "revelation".

### Paradigm shift

> **Before:** train a task architecture (LSTM, GRU…) whose embeddings are initialised with pretrained **word embeddings** (word2vec, GloVe, fastText).
> **After:** **fine-tune the full pretrained model**.

This was "pushed under the carpet" in the RNN lectures: in practice, those RNNs weren't learning their embeddings from scratch. They used the static embeddings from week 2. With GPT, *everything* is pretrained and *everything* is updated during fine-tuning, embeddings included.

> ⚠️ Lab MCQ 8.3 (`06 - Pretraining/0. Mini GPT.ipynb`) asks exactly this: *Which of the following was the paradigm shift introduced by pre-trained language models (PLMs) like GPT and BERT?*
> - ❌ **A.** Training from scratch on small datasets. That's what each task-specific model did *before*, on top of pretrained embeddings.
> - ❌ **B.** Using pretrained embeddings (Word2Vec, GloVe) with RNNs. That's the *old* paradigm.
> - ✅ **C.** Fine-tuning full pretrained models for downstream tasks.
> - ❌ **D.** Training only with unsupervised objectives. Pretraining is self-supervised, but fine-tuning uses labelled data.

### GPT-2 and scale

- **Same architecture** (plus pre-norm, from the lab), **more data**: 40 GB of **WebText** (pages linked from Reddit), and **four sizes** (117M/12 layers/768 → 1542M/48 layers/1600).
- **Less focus on fine-tuning, more on zero-shot.**
  - **Zero-shot:** you just ask, without training on the task. That's how you use an LLM today.
  - **Few-shot:** you also give a few examples in the prompt. Next lecture: [[2026-10-06 - From GPT to ChatGPT]].
- GPT-2's results tables started including a **human** score as reference. That's where "superhuman" comparisons began.
- The unicorn-style demo (slide 44: a news continuation about Miley Cyrus shoplifting) is fluent and broadly coherent. Some details are invented.

| model | date | parameters | data |
|---|---|---|---|
| GPT | Jun 2018 | 117M | 13 GB (~1B tokens) |
| GPT-2 | Feb 2019 | 1.5B | 40 GB |
| GPT-3 | Jul 2020 | 175B | ~500 GB (~300B tokens) |
| GPT-4 | Mar 2023 | 1.8T (rumoured) | ? |

The lecturer's point: the main driver was **scale**, more parameters and more data, and that trend was already visible in 2018–19. It comes back in weeks 7 and 13. (Also a rant: OpenAI was named for being open; now "nothing is open there".)

> [!example] Lab — a mini GPT from scratch (`06 - Pretraining/0. Mini GPT.ipynb`)
> A 0.8M-parameter character-level GPT on Tiny Shakespeare (65 characters, context 64).
> - **Baselines first:** uniform loss $= \ln 65 = 4.17$ (perplexity 65 = "choosing among 65 equally likely characters"); unigram ≈ 3.3 (PPL ≈ 27); the trained model ≈ 2.0 (PPL ≈ 7.4). Perplexity as "effective number of choices", as in [[2026-09-08 - Language Modeling (Part 1)]].
> - **One chunk of 65 characters = 64 training examples** (every prefix length), all in one pass thanks to the causal mask. No `<pad>`/`<s>`/`</s>` needed: all chunks have the same length, and there's no source/target split. Real GPTs do use an end-of-text token to separate *documents*.
> - **GPT-2 details:** **pre-norm** `x = x + SubLayer(LayerNorm(x))` (vs last week's post-norm `LayerNorm(x + SubLayer(x))`), which leaves an un-normalised residual "highway" and makes deep stacks easier to train, so a **final LayerNorm** is needed; **GELU** instead of ReLU; one fused `Linear(d, 3d)` for Q, K, V (same parameters as three, just faster).
> - **Weight tying:** the output layer reuses the token-embedding matrix. In GPT-2 small that's $50{,}257 \times 768 \approx 38.6$M parameters saved, ~31% of the 124M model, because of the big vocabulary.
> - **What it learns:** the *form* of Shakespeare (speaker names, line lengths, real-looking words), not meaning. Asked "The capital of France is", it continues in verse: **knowledge comes only from pretraining data**.
> - **Compute rule of thumb:** training costs $C \approx 6ND$ FLOPs ($N$ parameters, $D$ tokens). GPT-3: $6 \cdot 175\text{B} \cdot 300\text{B} \approx 3\cdot10^{23}$. **Chinchilla:** for fixed compute, train on ~**20 tokens per parameter**. GPT-3 had 1.7, so it was "too big for its data".

---

## 4. Contextual embeddings: ELMo

### Was GPT the first large-scale pretrained neural representation?

**No: word2vec.** Some students argued for RNN language models. The lecturer's answer: they modelled sequences but weren't used to *extract* a semantic representation, and they were never trained at web scale (long-range dependencies were too hard). Word2vec was the first to learn something semantic at web scale. Its vectors were what initialised the RNNs.

### What's wrong with word embeddings?

> **Words have different meanings in different contexts, and word2vec-style "static" embeddings learn one fixed representation for each word.**

The slide's example is *play*: a baseball play, a Broadway play, a sports play again, a theatre play again, and "concepts play an important role in cognition". That's three meanings in five sentences. Word2vec is a dictionary: one word, one vector, wherever it appears.

Careful with the vocabulary in the Menti answers. The *problem* is that static embeddings are **static / non-contextual** and therefore **not** able to handle polysemy. **Polysemy** (one word, several meanings) is the property of *words* that we *want* the embeddings to capture.

### How to add context? Answer 1 (2018): bidirectional LSTMs

**ELMo** (*Embeddings from Language Models*, one of the best papers of 2018):

- Train a **two-layer LSTM language model** on a **large** corpus.
- It's **bidirectional**: a **forward** LM predicts token $k$ from tokens $1..k-1$, and a **backward** LM predicts token $k$ from tokens $k+1..N$. They are trained jointly:
  $$\sum_{k=1}^{N}\Big(\log p(t_k \mid t_1,\dots,t_{k-1};\ \Theta_x, \overrightarrow\Theta_{LSTM}, \Theta_s) + \log p(t_k \mid t_{k+1},\dots,t_N;\ \Theta_x, \overleftarrow\Theta_{LSTM}, \Theta_s)\Big)$$
  The two directions have **separate LSTM parameters** but **share** the input embeddings $\Theta_x$ and the output (vocabulary) projection $\Theta_s$.
- The **embedding of each word** is computed from the LSTMs' **hidden states** at that token. So *bank* in "I deposited money in the **bank**" and in "I sat on the **bank** of the river" gets **two different vectors**: the forward LSTM summarises the left context, the backward one the right context.

**Training data: the 1B Word Benchmark, made of single sentences. Why is that a problem?** (see [1:32:00](https://au.cloud.panopto.eu/Panopto/Pages/Viewer.aspx?id=fc85844c-5719-4381-b9c9-b4d400774366&start=5520)) A word's meaning can depend on context **beyond its sentence**. ELMo only ever learns context *within* a sentence. Splitting documents into sentences was standard preprocessing back then (compute, weaker tokenizers).

### Using ELMo embeddings

![[appunti/assets/2026-09-29/plms-059.png]]

$$\text{ELMo}_k^{task} = \gamma^{task} \sum_{j=0}^{L} s_j^{task}\, \mathbf h_{k,j}^{LM}$$

- $\mathbf h_{k,0}$ (layer 0) is the **context-independent** token embedding, "nothing different from word2vec". Higher layers are the concatenated forward/backward hidden states, where context comes in.
- $s_j^{task}$: **softmax-normalised weights over the layers**, learned per task.
- $\gamma^{task}$: a scalar that lets the task model **scale** the whole ELMo vector.
- **Both are learned** for each downstream task.

**Why a weighted average of all layers instead of the last one** (as with an LSTM classifier, or GPT)? **Different layers capture different kinds of information.** Lower layers carry more **local/syntactic** information, higher layers more **abstract/semantic** information. A syntactic task like **POS tagging** puts more weight on the lower layers; a semantic one like **NLI** on the higher ones.

> **ELMo doesn't assume the final layer is the best. Each downstream task learns which layers are most useful.**

Result: improvements across the board over the SOTA (QA, NLI, SRL, coreference, NER, sentiment).

### What's the issue with ELMo?

![[appunti/assets/2026-09-29/plms-063.png]]

> **ELMo (and bidirectional RNNs in general) are unidirectional models masquerading as bidirectional.** Two separate LMs encode the forward and backward sequence, and their representations are concatenated.

It is **bidirectional at the representation level, not jointly bidirectional during prediction.** When the forward LM processes *bank*, it has never seen "of the river", and the backward LM has never seen "The fisherman sat". No single model looks at both sides at once. The lecturer admitted this sounds "nitpicky", but it's what BERT fixes, and BERT's ablation (§5) shows it matters.

Other Menti answers that were right: only trained on sentences, **not parallelisable** and the other **LSTM issues** (long-range dependencies), short context.

**"Future token influence" was wrong**, and the lecturer insisted on why (see [1:42:31](https://au.cloud.panopto.eu/Panopto/Pages/Viewer.aspx?id=fc85844c-5719-4381-b9c9-b4d400774366&start=6151)): seeing the future is only a problem when you train a model to **generate**. ELMo produces embeddings of a text you already have in full. There's nothing to leak and no cheating. This is the same rule as [[2026-09-15 - Language Modeling (Part 2)#5. Bidirectional RNNs]].

---

## 5. BERT: Bidirectional Encoder Representations from Transformers

### Model

- **Oct 2018** (Google), a few months after ELMo and GPT. "This is also where the OpenAI–Google fight started."
- **A Transformer encoder**, exactly the one from last week. Full self-attention, no mask, so **every token sees both sides in the same model**.
- **BERT-Base:** 12 layers, $d = 768$, 12 heads, **110M** parameters. **BERT-Large:** 24 layers, $d = 1024$, 16 heads, **340M**.
- Data: BooksCorpus (800M tokens) + English Wikipedia (2.5B tokens), ~3.3B tokens.

![[appunti/assets/2026-09-29/plms-065.png]]

**Input = token embedding + segment embedding + position embedding** (summed):

- **Token embeddings** over a vocabulary of ~30k **WordPiece** subwords (`play` + `##ing`). Tokenizers that are *learned* rather than split on spaces started around this time. They get their own lecture later.
- **Segment embeddings** $E_A$, $E_B$: BERT is trained on **pairs of sentences** (`[CLS] my dog is cute [SEP] he likes playing [SEP]`), so each token is marked as belonging to sentence A or B. This is BERT's addition over the plain Transformer.
- **Position embeddings**: **learned**.

The pair format serves BERT's second pretraining task, **next sentence prediction (NSP)**: is sentence B the real continuation of A? The lecture only mentioned it through the ablation below. (From the BERT paper, not the lecture: B is the real next sentence half of the time and a random sentence otherwise.)

### Two ways to pretrain

| | **causal (left-to-right) LM** | **masked LM** |
|---|---|---|
| example | "I really enjoyed the movie we watched on ___" | "I really enjoyed the ___ we watched on Saturday!" |
| context used | left only | **both sides, in one model** |
| who | GPT (OpenAI) | BERT (and BART) |

### Masked language modelling

**Pretraining is self-supervised**, done at scale on naturally occurring text: the labels are the text itself.

- Pick **15% of the tokens** to predict. Why 15%? "I don't know": it's what worked. 5% or 30% would also have been possible.
- For each of those tokens:
  - **80%**: replace it with `[MASK]` ("Akhil taught **[MASK]** today" → *class*),
  - **10%**: replace it with a **random token** ("Akhil taught **chocolate** today"),
  - **10%**: **keep** the original token ("Akhil taught **class** today").
- The model predicts the original token at all of those 15% positions.

**Why not always `[MASK]`?** The lecturer's explanation: so the model **can't learn a simple rule**. With so many parameters, it could learn "where there's a `[MASK]`, predict from these surrounding words" and overfit to the mask as *the* signal. Mixing in random and unchanged tokens makes the training robust (see [2:08:53](https://au.cloud.panopto.eu/Panopto/Pages/Viewer.aspx?id=fc85844c-5719-4381-b9c9-b4d400774366&start=7733)).

> The reason given in the BERT paper, which is sharper (my addition, not from the lecture): **`[MASK]` never appears at fine-tuning time.** If the model only learned to work at `[MASK]` positions, there would be a **pretraining/fine-tuning mismatch**. Random and unchanged tokens mean that the model **can't tell which tokens were corrupted**, so it must build a good contextual representation of **every** input token, not only the masked ones.

**Self-supervised vs unsupervised? An open question.** A student asked, and the lecturer didn't give a crisp answer (see [2:04:08](https://au.cloud.panopto.eu/Panopto/Pages/Viewer.aspx?id=fc85844c-5719-4381-b9c9-b4d400774366&start=7448)). Both learn patterns from the data alone, without external labels. What the lecturer said makes pretraining "self-supervised": the supervision comes from the fact that **the text is correct**. Shuffle it randomly and there's nothing to learn. The lecturer promised a better answer with an example next time. It wasn't given on 6 Oct.

> My working definition, not the lecturer's: self-supervised learning is unsupervised with respect to *human* labels, but the training is set up like supervised learning, with targets **generated automatically from the input** (the next word, the masked word) and a supervised loss (cross-entropy) on them. "Unsupervised" in the classical sense (clustering, PCA…) has no prediction target at all.

### Fine-tuning BERT

- **Prepend** a special **`[CLS]`** token. Its **output embedding** is a contextual representation of the whole sequence.
- **Put a linear classifier on it.** During fine-tuning, **all** of BERT's parameters are updated along with the classifier, using the gold labels.

![[appunti/assets/2026-09-29/plms-071.png]]

One pretrained model as the starting point for many tasks:

- **classification** (single sentence, e.g. sentiment SST-2, acceptability CoLA; or sentence **pairs** with `[SEP]`, e.g. entailment MNLI, paraphrase QQP/MRPC, similarity STS-B): use the **`[CLS]` output**;
- **sequence labelling** (e.g. NER on CoNLL-2003): use **each token's output** and classify it separately.

**Menti: why put `[CLS]` at the *front*?** (In the LSTM classifier and in GPT it went at the end.)

> **It's the easiest place to put it. Bidirectionality ensures that it attends to the representations of all other tokens.**

In GPT, the classification token has to come **last**: with a causal mask, only the last position has seen everything. In BERT every position sees everything, so any position would do. They tried both, and the front **worked better**. Another "deep learning: it just works better" (see [2:14:57](https://au.cloud.panopto.eu/Panopto/Pages/Viewer.aspx?id=fc85844c-5719-4381-b9c9-b4d400774366&start=8097)).

### GLUE and results

**GLUE** is a natural-language-*understanding* (**NLU**) benchmark of nine tasks: acceptability (CoLA), sentiment (SST-2), paraphrase (MRPC, QQP), sentence similarity (STS-B), NLI (MNLI, QNLI, RTE), coreference/NLI (WNLI). BERT-Large raised the average from 75.2 (OpenAI GPT) to **81.9**.

- **Performance increases across the board.** The big jump over GPT **highlights the importance of bidirectionality.**
- ⚠️ **It's not one model.** For each task, **a different BERT is fine-tuned** on that task's data, and the table reports each one. That's unlike today's LLMs, where one model does everything. The achievement in 2018 was that one *pretrained* model, fine-tuned on a small amount of task data, beat specialised systems everywhere.

### Ablations

> An **ablation** controls everything and removes **one** thing, to measure its effect. The lecturer recommended it for the project: "when you have to prove a point, ablation is the best way to prove it" (see [2:19:30](https://au.cloud.panopto.eu/Panopto/Pages/Viewer.aspx?id=fc85844c-5719-4381-b9c9-b4d400774366&start=8370)).

- **No NSP:** a small drop.
- **LTR & No NSP** (left-to-right, i.e. trained like GPT): clearly worse.
- **LTR & No NSP + BiLSTM on top:** adding bidirectionality *afterwards* recovers only part of the gap. This is ELMo's problem again: **bidirectionality has to be inside the model**, not stuck on at the representation level.
- **Model size:** more layers, hidden size and heads → better on every task. **More parameters = better.** Releasing a model in several sizes (Base/Large) also started here.

### Fine-tuning vs embeddings: ELMo is not BERT

![[appunti/assets/2026-09-29/plms-079.png]]

_❄ = frozen, 🔥 = fine-tuned, Δ = 🔥 − ❄._

The two kinds of model are meant to be used differently (see [2:23:03](https://au.cloud.panopto.eu/Panopto/Pages/Viewer.aspx?id=fc85844c-5719-4381-b9c9-b4d400774366&start=8583)):

- **ELMo is an embedding model.** Its **frozen** embeddings are already strong, and fine-tuning the ELMo network can make it **worse** (negative Δ on NLI and similarity). You use it like word2vec: extract the vectors and feed them to *any* downstream model (logistic regression, an LSTM…).
- **BERT (and GPT) are end-to-end models.** Their frozen embeddings are decent, but **fine-tuning the whole model** gives much more (positive Δ). You can't throw away the architecture and keep the vectors.

### Transfer learning

| **pretraining** | **fine-tuning** |
|---|---|
| learn **embeddings** to seed a downstream model (word2vec, ELMo) **or** a **model** to fine-tune (GPT, BERT) | design a new model initialised with pretrained embeddings, **or** take the pretrained model and keep training it on task data |
| **simple** objectives (causal LM, masked LM, NSP), so no labels are needed and data is naturally found on the web | **smaller** datasets with **labels**, i.e. more complex structure |
| needs **tons of data** | applied to the **task of interest** |
| the resulting model is often **not useful yet** on its own | **cheaper**: you can afford multiple runs and hyperparameter tuning |
| **slow and expensive**, often done **once** | |

"Once" is becoming a misnomer: companies now release new pretrained versions every few weeks when the competition heats up.

**Has anyone trained a BERT-style model on as much data as today's LLMs?** The lecturer didn't know of one with comparable data. Today's LLMs are mostly **decoder-only**, with mixture-of-experts, RL and more on top.

### What does BERT learn?

Different **attention heads** learn different relationships: attending broadly, to the **next token**, to `[SEP]`, to **periods**; **direct objects → their verbs**, **noun modifiers (determiners) → their noun**, **coreferent mentions → their antecedent**.

> **Transformer heads learn diverse concepts that map to positional, semantic and syntactic relationships.**

These are the attention maps promised in [[2026-09-22 - Seq2Seq Models and Transformers#Interpretability]], with the same caveat about reading too much into them. Interpretability gets a later lecture, possibly a guest lecture on mechanistic interpretability.

### What to mask? Whole-word masking

"Obama was the president…" is tokenised into `_O bama _was _the …`. Masking only `_O` (with `bama` visible) is **too easy**: the rest of the word gives it away, so the model learns less. **Whole-word masking** masks all subwords of a word together: `[MASK] [MASK] _was …`. The lecturer then corrected the slide: "I lied", it's about **tokens**, and the **tokenizer** decides what a "word" is (see [2:33:23](https://au.cloud.panopto.eu/Panopto/Pages/Viewer.aspx?id=fc85844c-5719-4381-b9c9-b4d400774366&start=9203)).

### Variants

- **RoBERTa** ("Robustly Optimised BERT"): **same architecture**, a collection of training improvements, mainly **far more data**, **160 GB instead of 13 GB** (≈12×; the lecturer said "4 times… no, 10 times"), plus larger batches and longer training. It beats BERT-Large. The data is the main reason.
- **DistilBERT**: do we need all those parameters for, say, movie reviews? **Distillation** is a **student–teacher** setup: a small model (student) learns to match the **soft output probabilities** of the full BERT (teacher), plus the usual MLM loss:
  $$\mathcal L_{distil} = -\sum P_{BERT}(y \mid \dots)\log P_{dbert}(y \mid \dots),\quad \mathcal L_{tot} = \gamma_1 \mathcal L_{distil} + \gamma_2 \mathcal L_{mlm}$$
  It keeps **~97% of BERT's performance** with a much smaller model. (The lecturer also gave a general figure of "10× smaller for 90–95%". The 97% on the slide is DistilBERT's. Why not just train the small model directly? A small model trained from scratch can't capture as many patterns. That answer is promised for week 13, on compression.)
- **ELECTRA:** BERT learns only from the **15%** masked tokens, which is **inefficient**. ELECTRA uses a small **generator** (a small MLM) that fills the masks with plausible tokens ("cooked" → "ate") and a **discriminator** (the real model) that predicts, **for every token**, whether it's **original or replaced**. Learning from **all tokens** speeds up pretraining a lot. For the same compute, ELECTRA beats BERT and RoBERTa, especially at small budgets.

![[appunti/assets/2026-09-29/plms-089.png]]

> [!example] Lab — what BERT learned (`06 - Pretraining/1. Pretrained Models.ipynb`)
> - **Context on both sides:** "I went to the [MASK] to buy some bread" → bakery/store; "… to see a doctor" → hospital/clinic. The words *before* the mask are identical, so a left-to-right model would have to give the same distribution in both. The flip side: BERT is **bad at generating**. It was trained to fill ~15% holes in an otherwise complete text, never to write left to right.
> - **Social bias:** "The [nurse/engineer] said that [MASK] would be late" → *she*/*he* along stereotypes. It's not a bug: the model reproduces the statistics of its data, **and fine-tuning doesn't remove it**. A CV-ranking model built on BERT can discriminate by gender without gender ever being a feature.
> - **Contextual vectors:** the cosine similarity of *bank* across 4 sentences groups the river senses vs the money senses. With word2vec all four would be **1.0**. Vectors of the same sense aren't identical (they encode the whole context), and the *relative* similarities carry the information.

---

## 6. Sequence-to-sequence pretraining: BART and T5

GPT uses only the decoder, BERT only the encoder. **Why not the full Transformer?**

### BART: Bidirectional and Auto-Regressive Transformer

- **Oct 2019** (Facebook), 140M (Base, 6+6 layers) / 400M (Large, 12+12 layers). BooksCorpus, many more books, Wikipedia.
- The **classic Transformer**: a **bidirectional encoder** feeds an **autoregressive decoder**. **Cross-attention is back.**

![[appunti/assets/2026-09-29/plms-094.png]]

**Pretraining combines BERT and GPT:**

- **BERT-style:** the input text is **corrupted** before it goes into the bidirectional encoder.
- **GPT-style:** the decoder is trained with a **language-modelling objective**: it regenerates the **original** text, predicting the next token.

This is a **denoising autoencoder**. Since the decoder always rebuilds the full original, **any corruption** is allowed, and with a decoder you can also corrupt the *order*, which BERT can't:

![[appunti/assets/2026-09-29/plms-095.png]]

| corruption | what it does |
|---|---|
| token masking | tokens → `[MASK]` (as BERT) |
| token deletion | tokens removed; the model must also find *where* |
| text infilling | a whole **span** → a single mask; the model must also find *how many* tokens |
| sentence permutation | sentences shuffled |
| document rotation | the document starts at a random token; the model finds the real start |

(The lecturer called these making the model "permutation invariant, rotational invariant". Strictly, the model learns to **undo** permutations and rotations.)

**Which corruption?** **Text infilling** is the best single one, and **text infilling + sentence permutation** is best overall. Document rotation and sentence shuffling alone are clearly worse. The differences among the good ones are small, and plain token masking/deletion already work well.

**BART vs BERT:**

- **BART can do everything BERT does.**
  - **Classification:** give the input to **both encoder and decoder** ("input the sequence twice"), append `[CLS]` to the decoder sequence, and classify its output.
  - **Sequence labelling:** same trick, and classify each decoder output.
- On GLUE/SQuAD it's **almost as good as RoBERTa** and **way better than BERT**. **Why? It was trained on way more data** (the same ~160 GB as RoBERTa).
- **And BART can also generate** (summarisation, translation, abstractive QA). **Why? It has an autoregressive decoder.**

### T5: Text-to-Text Transfer Transformer

![[appunti/assets/2026-09-29/plms-100.png]]

- **The idea: any problem can be cast as sequence-to-sequence.** Translation: `translate English to German: That is good.` → `Das ist gut.`. Acceptability: `cola sentence: …` → `not acceptable`. Similarity: `stsb sentence1: … sentence2: …` → `3.8`, a number **written as text**.
- This is how we use LLMs today: one model, trained only to produce text, used for classification, regression, clustering… The lecturer sees this as **T5's main novelty** over BART.
- Same encoder–decoder as BART. It's from **Google** (Raffel et al.); the lecturer wasn't sure ("Meta or Google, don't quote me").

![[appunti/assets/2026-09-29/plms-101.png]]

**Pretraining: span corruption** (infilling). Random spans are replaced by **sentinel** tokens `<X>`, `<Y>`. The target lists **only the missing spans**, each after its sentinel:

```
original  Thank you for inviting me to your party last week.
input     Thank you <X> me to your party <Y> week.
target    <X> for inviting <Y> last <Z>
```

**T5 pretraining decisions:** they explored many dimensions (objective: LM / BERT-style / deshuffling; corruption: mask / replace spans / drop; corruption rate 10–50%; span length 2–10), then used the findings to train a much larger model, **11B parameters**. The lecturer went quickly here because these are engineering choices you could explore yourself.

> [!example] Lab — T5 and choosing a family
> - T5 does classification by **writing the label** (`acceptable`, `positive`) and regression by writing **a number as a string**. There are no task-specific heads: one loss, one decoding procedure.
> - The **prefixes aren't instructions it understands**. They're text that always preceded a task in training. With an unseen prefix (`translate English to Danish:`) or a typo, it doesn't error out; it does *something* (German, French, or a copy). Following new instructions came only with much larger instruction-tuned models ([[2026-10-06 - From GPT to ChatGPT]]).
> - **Which family?** Classify reviews → **encoder** (BERT + head on `[CLS]`). Write stories → **decoder** (GPT). Translate → **encoder–decoder** (T5/BART/opus-mt). Find similar documents → **encoder** (embed once, cosine similarity).
> - T5's span corruption replaces a whole span with **one** sentinel, and the target contains **only the missing parts**, which makes pretraining cheaper than reconstructing the full text (BART).

---

## Wrap-up

| | architecture | pretraining | good for | generates? |
|---|---|---|---|---|
| **ELMo** | 2 separate unidirectional LSTM LMs (fwd + bwd), concatenated | 2 LM objectives, 1B-word sentences | **pretrained embeddings** (frozen, fed to another model) | ✗ |
| **GPT** | Transformer **decoder** (masked SA, no cross-attn) | **causal LM** | **generating text** as a language model | ✓ |
| **BERT** | Transformer **encoder** | **masked LM** (+ NSP) | **classification and sequence labelling** | ✗ |
| **BART / T5** | full **encoder–decoder** | **denoising** (infilling, permutation…) / **span corruption** | **extendable to all task types** | ✓ |

**Contextual representations** let us model words and sequences conditioned on their context. These models laid the foundations for today's LLMs.

> ⚠️ **Exam (2025 Q10 = mock exam, rated medium):** *Which of the following statements about model applicability is correct?* The table's last column answers it.
> - ❌ **A.** T5 can only perform generation tasks without any modification and cannot perform text classification tasks. T5 frames **every** task as text-to-text, so it classifies by writing the label.
> - ✅ **B.** BERT cannot perform text generation tasks without any modification. It's an encoder: no autoregressive decoder.
> - ❌ **C.** A fixed-window language model using only average pooling can perform generation tasks. Marked false; the official reasoning is debatable, see [[2026-09-08 - Language Modeling (Part 1)#14. Fixed-context neural language models]].
> - ❌ **D.** ELMo can perform both generation and classification tasks without any modification. ELMo only provides contextual embeddings, fed to another model.

Chronology: **ELMo** (early 2018) → **GPT** (Jun 2018) → **BERT** (Oct 2018) → **GPT-2** (Feb 2019) → **BART**, **T5** (Oct 2019) → **GPT-3** (2020). (In class the lecturer once said GPT and GPT-2 "had already been done" when ELMo came out. The order above is the right one: ELMo predates both.)

---

## Da ricordare

- **Positions:** $x_i = e_i + p_i$ (**added**, so dimensionality is unchanged). Sinusoidal = fixed "clocks" at different speeds (fast ones for nearby positions, slow ones for global position). **Learned** = easiest, but **no vector beyond the max trained length**. Learned positions *are* parallelisable. **RoPE** (rotate Q, K → score depends on $n-m$) and **ALiBi** (distance penalty, extrapolates, recency bias) are *relative*. A **decoder** can learn order with no PE at all thanks to the causal mask; an encoder can't.
- **Transformer vs RNN disadvantage:** **$O(T^2)$ vs $O(T)$ in the sequence length.** Not overfitting. Fix: **sparse attention** (global hubs + local window + random links = BigBird).
- **GPT** = decoder-only (masked self-attention, **no cross-attention**), **causal LM** on BooksCorpus, 512-token windows. Fine-tune = linear head on the **last** token + task-specific **input formatting** (delimiters; similarity fed in **both orders**). **Paradigm shift = fine-tuning the full pretrained model** (not "pretrained embeddings + RNN", which is the old way). GPT-2 = same model, more data, **zero-shot**.
- **Static embeddings fail on polysemy.** **ELMo** = 2-layer **biLSTM** LM; the task learns **softmax weights over layers** ($s_j$) + a scale $\gamma$ (low layers = syntax, high layers = semantics). Issue: **two unidirectional models masquerading as bidirectional** (bidirectional at the representation level, not jointly). Trained on **single sentences**. **"Seeing the future" is no problem when you don't generate.**
- **BERT** = encoder; input = token + **segment** + position embeddings; **MLM: 15% of tokens → 80% `[MASK]`, 10% random, 10% unchanged** (the model can't rely on `[MASK]`, which never appears in fine-tuning). **`[CLS]` at the front** (bidirectional, so it sees everything; it worked best) + linear classifier; per-token outputs for tagging. **On GLUE, each task has its own fine-tuned BERT.** The ablation shows **bidirectionality must be inside the model** (LTR + BiLSTM < BERT).
- **ELMo = frozen embeddings** (fine-tuning can even hurt). **BERT/GPT = fine-tune end-to-end.** **Whole-word masking**: masking one subword is too easy. **RoBERTa** = same BERT, ~12× data. **DistilBERT** = student matches the teacher's soft probabilities (~97% performance). **ELECTRA** = generator + discriminator, learns from **all** tokens.
- **BART** = bidirectional encoder + autoregressive decoder, **denoising** pretraining (best: **text infilling + sentence permutation**). **Beats BERT because of more data**; **can generate because of the decoder**. Classification by feeding the input to both encoder and decoder. **T5** = **every task as text-to-text** (labels and numbers written as text), span corruption with sentinels.
- ⚠️ **Exam (2025 Q10 / mock):** **BERT cannot generate text without modification.** T5 *can* classify (text-to-text). ELMo only gives embeddings (no generation, no classification on its own). "A fixed-window LM with average pooling can generate" is marked false too (debatable, see [[2026-09-08 - Language Modeling (Part 1)#14. Fixed-context neural language models]]). **2025 Q9:** in BART/T5 cross-attention, **keys/values = final encoder layer output, queries = decoder states**.
