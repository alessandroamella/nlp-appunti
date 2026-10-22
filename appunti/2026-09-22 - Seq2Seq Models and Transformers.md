This lecture takes the RNNs of [[2026-09-15 - Language Modeling (Part 2)]] and goes in three steps. Each step fixes a problem left open by the one before:

1. **Encoder–decoder models.** Input and output sequences have *different properties* (e.g. French in, English out), so use two RNNs.
2. **Attention.** The decoder only sees *one vector* summarising the whole input (the **temporal bottleneck**), so let it look back at every encoder state.
3. **Transformers and self-attention.** The encoder is *still recurrent*, so it's slow and can't be parallelised. Drop recurrence entirely and use attention everywhere.

The deck (`05 - Transfomers.pdf`) opens with a recap of vanishing gradients and the skip/residual/gate slides (8–16). That part is covered in [[2026-09-15 - Language Modeling (Part 2)#11. Recap from 22 Sep: gates as learned skip connections]] and isn't repeated here. This note starts at slide 17.

> **Deferred:** position embeddings (slides 91–96) were only shown in the last two minutes, and the lecturer said the next lecture would start from them. §4.7 below is a placeholder from the slides.

---

## 1. Encoder–decoder models

### Why two models?

> **What could we do if the sequence we're encoding and the sequence we want to generate have different properties?** Example: machine translation.

What "different properties" means, from the class: different **grammar and word order**, different **alphabet/script**, different **syntax**, and different lengths. A single RNN LM ([[2026-09-08 - Language Modeling (Part 1)#16. Recurrent neural networks (RNNs) — introduction]]) models *one* sequence. You can put a classifier on its last state, or tag every token. But it has no natural way to read one sequence and then produce a *different* one.

So use **two** models:

- An **encoder** reads the whole input sequence and produces a representation of it (its final hidden state).
- A **decoder** is seeded with that representation and generates the output sequence. It is **autoregressive**: one token at a time, with each generated token fed back in as the next input, exactly like an RNN LM.

![[appunti/assets/2026-09-22/transformers-020.png]]

_Encoder LSTM reads "Je m' appelle Akhil"; the green vector (its last state) seeds the decoder, which starts from `<START>` and generates "I am Akhil `<END>`", feeding each prediction $\hat y_t$ back as the next input._

`<START>` and `<END>` are special tokens added during training. The model learns to begin from `<START>` and to emit `<END>` when it's done, which is how generation knows when to stop.

The lecturer related this to how an LLM chat feels: your prompt is *encoded* into some representation, and the response is *decoded* from it. This is only an analogy. See the warning below.

### Applications

The template fits any "sequence in → different sequence out" task:

| task | input | output |
|---|---|---|
| machine translation | sentence in language A | sentence in language B |
| code generation | natural-language spec ("Write Hello World Function") | code (`print("Hello...`) |
| image captioning | an image (encoded by a vision model instead of an RNN) | its description |
| summarization | long document | short summary |
| style transfer | text in one style | same content, another style |

- ⚠️ **Slide warning: "Not how Cursor/Copilot works!"** Modern code assistants aren't encoder–decoder RNNs. The lecturer said this comes in the next lectures. (My guess: they're decoder-only Transformer LMs.)
- The **input doesn't have to be text** (image captioning). The encoder is just "whatever turns the input into a vector".

### Bidirectional encoders

The encoder sees the whole input before anything is generated. So it's allowed to be **bidirectional** ([[2026-09-15 - Language Modeling (Part 2)#5. Bidirectional RNNs]]). This is the "legitimate bidirectional" case promised at the end of that section. The **decoder must stay unidirectional**: it is autoregressive and **can't know the future**.

![[appunti/assets/2026-09-22/transformers-026.png]]

**Does the decoder have to be twice the dimensionality of the encoder?** The bidirectional encoder gives $[h^{fwd}_n; h^{bwd}_n]$, which has size $2d$. **No.** Project it first with a learned linear layer:

$$h_0^{dec} = W_{dec}\,[h_n^{fwd}; h_n^{bwd}] + b_{dec}$$

The decoder can then have whatever size you like. This is the usual deep-learning move: when shapes don't match, add a learned transformation.

### Training

The objective is the same as for a language model: **minimise the average cross-entropy over the output tokens**, now conditioned on the input too:

$$\mathcal L = \sum_{t=1}^{T} -\log P(y_t \mid y_1,\dots,y_{t-1},\, x_1,\dots,x_n)$$

- Notation change: **encoder inputs are indexed by $n$, decoder outputs by $t$.**
- The gradients **backpropagate through both the decoder and the encoder**. The two are trained **end-to-end** as one network, and the encoder learns whatever representation helps the decoder.
- During training, the decoder's inputs are the **gold** previous tokens $y_0, y_1, \dots$ (see the slide diagram), not its own predictions. The standard name for this is **teacher forcing**. The lecturer didn't use the term; the Week-5 lab does.

> [!example] Lab — teacher forcing and greedy decoding
> `ta_sessions/Exercises/05 - Transformers/1. Transformer.ipynb` (§5) trains a small Transformer to "translate" dates (`saturday, 12 september 2026` → `2026-09-12`). Teacher forcing there is literally a shift by one:
>
> ```
> target          <s>  2  0  2  6  -  0  9  -  1  2  </s>
> decoder input   <s>  2  0  2  6  -  0  9  -  1  2        = target without its last token
> labels           2   0  2  6  -  0  9  -  1  2  </s>     = target without its first token
> ```
>
> At **inference** there is no gold target, so the lab uses **greedy decoding**: start from `<s>`, run the model, take the argmax of the **last** position only, append it, repeat until `</s>`. The asymmetry is the point:
> - **Training** is parallel: every decoder input is known in advance, so all 11 positions are predicted in **one** forward pass. With a Transformer decoder this needs the causal mask (§4.6).
> - **Inference** is sequential: the input at $t+1$ *is* the prediction at $t$, so 11 tokens take **11** forward passes. Real implementations cache the keys/values of earlier positions so they don't recompute everything each time.
>
> Hugging Face's `model.generate(num_beams=1)` is this same greedy loop (`2. Pretrained Translation.ipynb`). The library's default for that model is **beam search** (4 beams), a bit better and ~4× slower. The lab says decoding strategies come later in the course.

### Practical challenge: paired data

- A **language model** has practically **unlimited data**: any text will do, because it only learns which word follows which.
- An **encoder–decoder** learns which sequences **align** with which. It needs **paired data**: parallel sentences for MT, image–caption pairs, code–comment pairs, and so on. ⚠️ **Paired data is much harder to find in the wild.**

The lecturer added a point that came out of a break discussion. Many inputs have **more than one valid output**: several correct translations of a sentence, many ways to implement the same function. A paired dataset usually gives you *one*, so the model sees a narrow slice of the valid outputs and generalises worse (see [1:03:07](https://au.cloud.panopto.eu/Panopto/Pages/Viewer.aspx?id=fa3c84d3-b9d6-47b6-b8bd-b4cd0076eb7a&start=3787)). The lecturer also said this architecture was genuinely used in industry before Transformers. When the lecturer worked at American Express, "9, 10 years ago", they trained one of these to turn textual queries into SQL queries for analysts (see [1:04:25](https://au.cloud.panopto.eu/Panopto/Pages/Viewer.aspx?id=fa3c84d3-b9d6-47b6-b8bd-b4cd0076eb7a&start=3865)).

---

## 2. The temporal bottleneck

> «You can't cram the meaning of a whole %&@#&ing sentence into a single $*(&@ing vector!» — Ray Mooney (NLP professor, UT Austin)

This is the RNN problem from last lecture, now in a worse position:

- The state is a **single vector**, which means **massive compression** of the information.
- It's **re-computed at every step**. So it's hard to learn long-range dependencies without ignoring the immediate ones, or the other way round.
- Nearby words *should* affect each other more than distant ones. The problem is that RNNs make it hard to learn **any** long-range interaction.

In the encoder–decoder, **everything** the decoder knows about the input comes through that one green vector.

![[appunti/assets/2026-09-22/transformers-034.png]]

**Toy example.** To generate "I", the decoder needs "Je". But "Je" was the **first** token encoded. Since then the state has been overwritten by 4 more tokens ("m'", "appelle", "Akhil", `<START>`), and for this prediction those are noise. With 4 tokens it may not matter. With a long document, or with a long prompt of the kind people give LLMs today, it is a real problem.

> ⚠️ **Exam (2025 Q5 = mock exam, rated easy):** *Which of the following accurately describes a key disadvantage of using recurrent neural networks (RNNs) for modeling long sequences?*
> - ✅ **A.** RNNs compress all historical context into a single fixed-size state vector, causing information loss. That's this section.
> - ❌ **B.** RNNs always have quadratic computational complexity with respect to sequence length, making them impractical for long sequences. No, an RNN is **linear** in $T$ (one step per token). Quadratic is the *Transformer's* cost (§4).
> - ❌ **C.** RNNs require extremely large datasets for training because each timestep uses different parameter sets. No, the **same weights are shared across all steps**. That's what lets an RNN handle any length.
> - ❌ **D.** RNNs cannot learn nonlinear representations because their hidden states must remain linear combinations of past inputs. No, $h_t = \sigma(W_{hx}x_t + W_{hh}h_{t-1} + b)$ goes through a nonlinearity at every step.
>
> The three distractors are each false about RNNs in general ([[2026-09-08 - Language Modeling (Part 1)#16. Recurrent neural networks (RNNs) — introduction]]).

### Menti: how can we mitigate the temporal bottleneck?

Many answers were "attention", which is where the lecture goes next. Some said **skip connections**. The lecturer explained why that isn't the fix here. The LSTM blocks *already* have learned, residual-like paths ([[2026-09-15 - Language Modeling (Part 2)#11. Recap from 22 Sep: gates as learned skip connections]]), which are stronger than a plain skip. You can still add skips between the blocks, and they'd still help a bit, "but it will not be as useful" (see [1:09:25](https://au.cloud.panopto.eu/Panopto/Pages/Viewer.aspx?id=fa3c84d3-b9d6-47b6-b8bd-b4cd0076eb7a&start=4165)).

> My reading of why (not stated this way in the lecture): skip connections fix **gradient flow**. The bottleneck is a **capacity** problem: all the input must pass through one fixed-size vector. Better gradient paths don't give the decoder more to look at.

---

## 3. Attention

### The idea

The encoder produces an output $h^e_n$ at **every** step, not just the last one. We've been throwing all of them away except $h^e_N$.

**Idea:** at each decoder step, use the decoder's current state to compute an **attention** (a weighted **mixture**) over **all** the encoder outputs $h^e_1,\dots,h^e_N$.

**Intuition:** the decoder can **focus on different parts of the input at each time step**. When translating "Je", look mostly at "Je".

> **What is attention?** A **weighted average over a set of inputs**.

The whole question is how to choose the weights.

**Why not just sum (or average) the encoder states?** Someone in class suggested it. If you sum first, you've collapsed everything into one vector again, and the information is lost. The weights have to depend on **what the decoder needs right now** (see [1:14:35](https://au.cloud.panopto.eu/Panopto/Pages/Viewer.aspx?id=fa3c84d3-b9d6-47b6-b8bd-b4cd0076eb7a&start=4475)).

### The attention function, step by step

![[appunti/assets/2026-09-22/transformers-049.png]]

1. **Score.** Compute a pairwise similarity between the decoder state (the "idea of what to decode") and **each** encoder state:
   $$a_n = f(h^e_n, h^d_t)$$
   There is **one query vector and many key vectors**.
2. **Normalise.** Turn the scores into a probability distribution over encoder positions with a **softmax** (primer in [[2026-09-01 - Neural Networks and Word Embeddings]]):
   $$\alpha_n = \frac{e^{a_n}}{\sum_j e^{a_j}}$$
3. **Weighted sum.** The attention output is the weighted average of the encoder states:
   $$\tilde h^d_t = \sum_{n=1}^{N} \alpha_n\, h^e_n$$

### Keys, query, values

The terminology comes from databases/retrieval:

| role | here | what it's used for |
|---|---|---|
| **query** | the decoder state $h^d_t$ | "what am I looking for right now?" |
| **keys** | the encoder states $h^e_n$ **when scoring** | "what does each position offer?", compared against the query to get $\alpha$ |
| **values** | the encoder states $h^e_n$ **when summing** | what actually gets mixed into $\tilde h^d_t$ |

In the RNN version, keys and values are **the same vectors** used in two roles. The distinction is still worth making. **The attention distribution says *which* positions matter and how much. The values are *what* you take from them.** Transformers give the two roles separate projections (§4).

### Scoring functions

![[appunti/assets/2026-09-22/transformers-045.png]]

| name | formula |
|---|---|
| multiplicative (bilinear) | $a = h^{e\top} W h^d$ |
| "linear" (usually called **additive**, MLP-style) | $a = v^\top \phi(W[h^e; h^d])$ |
| **scaled dot product** | $a = \dfrac{(W h^e)^\top (U h^d)}{\sqrt d}$ |

- The slide labels the second row "Linear", but $\phi$ is a nonlinearity. The lecturer called it the "additive / MLP-style" function, which is the standard name (Bahdanau attention).
- **Scaled dot product is the one used in practice** and the one in Transformers. The lecturer's reasons: it's efficient to compute (it's just matrix multiplications), and it scales well with the dimensionality (see [1:17:18](https://au.cloud.panopto.eu/Panopto/Pages/Viewer.aspx?id=fa3c84d3-b9d6-47b6-b8bd-b4cd0076eb7a&start=4638)).
- **Why divide by $\sqrt d$?** This wasn't explained in class, but the Week-5 lab derives it (below).

> [!example] Lab — why divide by $\sqrt{d_k}$
> `ta_sessions/Exercises/05 - Transformers/0. Attention.ipynb` (§1–2). The whole mechanism is four lines:
>
> ```python
> scores  = Q @ K.transpose(-2, -1)   # one dot product per (query, key) pair
> scores  = scores / math.sqrt(d_k)   # scale the SCORES (not the keys)
> weights = F.softmax(scores, dim=-1) # softmax over the keys
> output  = weights @ V               # weighted sum of the values
> ```
>
> **The derivation.** At initialisation the entries of $q$ and $k$ are roughly independent, mean 0, variance 1. Then $q\cdot k = \sum_{i=1}^{d_k} q_ik_i$ is a sum of $d_k$ terms of variance 1, so $\text{Var}(q\cdot k) = d_k$ and its typical size is $\sqrt{d_k}$ (≈ 23 for $d_k=512$). The lab measures it: the variance comes out ≈ $d_k$ for $d_k$ from 4 to 1024, and ≈ 1 after dividing by $\sqrt{d_k}$.
>
> **Why that's a problem.** The softmax exponentiates *differences*: a gap of 20 between two scores is a factor $e^{20}$. Unscaled, one random key takes practically all the weight, so the softmax becomes a hard argmax **by chance, at initialisation**. A saturated softmax has ≈ **zero gradient**: nudging the scores doesn't change a one-hot output, so no signal reaches $W^Q, W^K$ telling them which key *should* have won. The model starts on a plateau (the lab compares it to a closed forget gate, [[2026-09-15 - Language Modeling (Part 2)#8. Long Short-Term Memory (LSTM)]]). Scaled, attention starts soft and near-uniform and can sharpen where training wants it.
>
> A sharp attention distribution isn't bad in itself. It's bad when it's there before the model has learned anything.

> ⚠️ **Exam (2025 Q8 = mock exam, rated medium):** *In a Transformer's self-attention layer, which one of the following statements is correct?* Each distractor tests a different fact.
> - ❌ **A.** The self-attention mechanism ensures that each token attends only to previous tokens, which allows the model to preserve causal order. That's **masked** self-attention, used only in the decoder (§4.6). Plain self-attention (the encoder, BERT) sees the whole sequence.
> - ❌ **B.** The attention weights are computed by applying a softmax over the Query matrix, normalizing the importance of each input dimension within a single token representation. The softmax is over the **scores $QK^\top$**, one row per query, and it normalises **across keys** (positions), not across embedding dimensions.
> - ❌ **C.** Increasing the dimension of the Query and Key vectors does not change the computational complexity of the attention matrix. It does. $QK^\top$ is $T^2$ dot products of length $d_k$, so $O(T^2 d_k)$, and $W^Q, W^K$ get more parameters.
> - ✅ **D.** The scaled dot-product attention divides the dot product of Query and Key by $\sqrt{d_k}$ to prevent the softmax function from entering regions with extremely small gradients when the vector dimension is large. That's the derivation above.

### The attentive encoder–decoder

![[appunti/assets/2026-09-22/transformers-050.png]]

At each decoder step $t$:

1. Run the decoder LSTM to get $h^d_t$.
2. Compute the attention output $\tilde h^d_t$ over all encoder states, using $h^d_t$ as the query.
3. **Concatenate** $[\tilde h^d_t; h^d_t]$ (or combine them some other way) and pass it to the **output layer**, which predicts $\hat y_t$.
4. Feed $\hat y_t$ back in and repeat. **The attention distribution is recomputed at every step.** What mattered for step 1 need not matter for step 2.

$\tilde h^d_t$ contains the information from the encoder states that got **high attention**.

**Why concatenate rather than add?** The two vectors carry different information:

- $h^d_t$ carries *what has been generated so far*.
- $\tilde h^d_t$ carries *which part of the source is relevant for this token*.

**Adding** them mixes the two "too early", before the model has used either. **Concatenating** keeps them separate and lets the output layer **learn** how to combine them (see [1:23:25](https://au.cloud.panopto.eu/Panopto/Pages/Viewer.aspx?id=fa3c84d3-b9d6-47b6-b8bd-b4cd0076eb7a&start=5005)).

The contrast with §1: without attention, the decoder always used **the same** green vector. With attention, it gets a **different, task-specific composition** of the encoder states at every step.

### Interpretability

![[appunti/assets/2026-09-22/transformers-053.png]]

The attention weights can be **visualised** as a heatmap (output tokens × input tokens). Doing so shows **what was focused on when each word was generated**. For EN→FR, the map is close to diagonal, but not exactly: "zone économique européenne" attends to "European Economic Area" in *reversed* order, which is the adjective–noun flip between the two languages. So **training with attention gives you an alignment between the languages for free**, with no alignment labels.

The lecturer said this style of attention map comes back next week, used to probe what early pre-trained models (BERT) learn.

> [!example] Lab — cross-attention maps, toy and real
> `ta_sessions/Exercises/05 - Transformers/1. Transformer.ipynb` (§6) and `2. Pretrained Translation.ipynb` (§5) plot the decoder's cross-attention (rows = output tokens, columns = source tokens).
> - **Dates** (`saturday, 12 september 2026` → `2026-09-12`): no diagonal but **blocks**. Writing the year, the heads look at the far **right** end of the source; writing the month, they land on the **third letter** of the month name (`p` in *september*: the first letter doesn't identify a month, the first three do); the weekday gets ≈ no weight anywhere. The source is read in the order the *output* needs it.
> - **Real EN→FR model** (`opus-mt-en-fr`): a rough diagonal, broken exactly where the languages disagree. *European economic area* → *espace économique européen* gives the same **anti-diagonal** as the slide. The `</s>` column (and the full stop) collects weight in almost every row: a "default" slot where a head puts its weight when it needs nothing from the source.
>
> ⚠️ **Refinement of "interpretability" above.** The lab warns that attention maps are a good **debugging/sanity-check** tool but **not a proof** of how the model decides. After several encoder layers of self-attention, the vector "at *area*" already mixes in the whole sentence; the plot is usually one layer averaged over heads; and information also reaches the output through the decoder's self-attention and the residuals. Attention shows where information *could* have been read.

### Menti: in what range can an attention weight fall?

**$[0, 1]$.** Each $\alpha_n$ comes out of a softmax. The "trick" part: that's not all. **The weights also sum to 1** over the encoder positions, because the softmax output is a probability distribution. So they are not independent numbers in $[0,1]$ (see [1:33:20](https://au.cloud.panopto.eu/Panopto/Pages/Viewer.aspx?id=fa3c84d3-b9d6-47b6-b8bd-b4cd0076eb7a&start=5600)).

> Strictly, a softmax never outputs exactly 0 or 1, so the range is the open interval $(0,1)$. The lecturer's answer was "between 0 and 1, and they sum to 1".

### How does attention fix the temporal bottleneck?

> **Direct connections between decoder states and encoder hidden states.**

Nothing is compressed into a single vector any more. At every step the decoder has access to **all** the input information, weighted by what it needs now.

### Wrap-up

- **Main idea:** the decoder computes a **weighted sum of the encoder outputs**. The weights come from pairwise scores between each encoder state and the current decoder state.
- **Many possible scoring functions** (dot product, bilinear, additive…). Scaled dot product dominates.
- **Temporal bottleneck fixed**: there is a direct link between decoder and encoder states. **This also helps with vanishing gradients.** The link is a short path from the loss back to every encoder step, instead of going through the whole chain.
- **Interpretability**: attention maps let you investigate the model's behaviour.
- **Attention is agnostic to the type of RNN** in the encoder and decoder (vanilla, LSTM, GRU…).

---

## 4. Transformers and self-attention

### 4.1 What's still wrong: the encoder is recurrent

Menti question: *do any other inefficiencies remain in our seq2seq pipeline?* The encoder is still an RNN, so:

- It still has **vanishing gradients and trouble with long-range dependencies** inside the encoder. Attention fixed the decoder→encoder link, not the encoder itself.
- **It can't be parallelised.** $h^e_n$ needs $h^e_{n-1}$, so the encoder states are **computed in series**.

**When is this a problem?** In **summarization**: the input is a long document, and encoding it token by token is slow.

**Solution: Transformers.** This is the architecture from *Attention Is All You Need* (Vaswani et al., 2017, Google). The lecturer called it the game-changing paper, and the architecture is the base of essentially every modern language model.

### 4.2 The full architecture, at a glance

![[appunti/assets/2026-09-22/transformers-063.png]]

- It is made of an **encoder** (left) and a **decoder** (right).
- Both are **stacks of $N$ transformer blocks**. The blocks are slightly different in the encoder and the decoder.
- The blocks are built from **multi-headed attention** layers (self-attention, orange) and **feed-forward** layers (blue).
- **There are no recurrent computations.** Sequences are encoded with **self-attention**.

It looks complicated, but taken block by block it isn't.

### 4.3 Self-attention

- **Original idea (§3):** use the *decoder* state to compute attention over the *encoder* states.
- **New idea:** use the encoder states to compute attention **over themselves**.
- **Drop recurrence:** compute all the encoder representations **in parallel**.

Notation: $h^\ell_t$ is the encoder state at position $t$ in **layer** $\ell$. The subscript is back to $t$, because input and output are no longer being distinguished. $h^0_t$ is the input (the token embedding). Layer $\ell+1$ is computed from layer $\ell$.

**Worked example: computing the new representation of position $s$.**

![[appunti/assets/2026-09-22/transformers-070.png]]

Take position $s$ as the **query**. **Every** position $t$, $s$ included, acts as a **key** and a **value**:

$$a_{st} = \frac{(W^Q h^\ell_s)^\top (W^K h^\ell_t)}{\sqrt d} \qquad \alpha_{st} = \frac{e^{a_{st}}}{\sum_j e^{a_{sj}}} \qquad \tilde h^\ell_s = \sum_{t=1}^{T} \alpha_{st}\,(W^V h^\ell_t)$$

It's exactly the §3 recipe (score, softmax, weighted sum). Two things changed:

1. **Query, keys and values all come from the same sequence.** For each attention computation, **every element is a key and a value, and one element is the query.** The sum runs over $\{1,\dots,T\}$, which **includes $s$ itself**. That is what makes it *self*-attention.
2. **Three separate learned projections**, $W^Q, W^K, W^V$. In the RNN version the same $h^e_n$ served as both key and value. Here each role gets its own weight matrix, so the model can learn *what to look for*, *what to be matched on* and *what to hand over* separately. These matrices are where the layer's learnable parameters are. Without them there'd be nothing to learn.

Then **do this for every position**, all at once:

![[appunti/assets/2026-09-22/transformers-073.png]]

$$\tilde h^1_s = \text{Attention}\big(h^0_s,\ \{h^0_t\}_{t=1}^{T}\big) \quad \text{for } s = 1,\dots,T$$

None of these computations depends on another, so they **all run in parallel**. In matrix form (the standard way to write it; the slides use a slightly different arrangement), with the $h$'s stacked as rows of $X$:

$$Q = XW^Q,\quad K = XW^K,\quad V = XW^V,\qquad \text{Attention}(Q,K,V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt d}\right)V$$

The slides also apply an output projection $W^O$ at the end. It matters for multi-head attention (below).

The deck's "Nobel committee awards Strickland who advanced optics" slides animate the same thing: every token gets a $Q$, $K$ and $V$ vector; each query produces an attention distribution $A$ over all tokens; the output $M$ is the weighted sum of the values.

> [!example] Lab — two things the hand-computed example shows
> `ta_sessions/Exercises/05 - Transformers/0. Attention.ipynb` (§1) runs self-attention by hand on *He sat river bank* with 2-d vectors.
> - **Weights are relative.** A key with score 0 (orthogonal to the query) still gets a non-zero weight ($e^0$ vs $e^{1/\sqrt2}$): the softmax only compares the scores *within a row*.
> - **Attention is directed, not symmetric.** Even with $Q = K$ (so $QK^\top$ is symmetric), the weight matrix isn't: each **row** is normalised separately, so the same raw score can be the top of one row and lose to strong competition in another. How much A attends to B says nothing about how much B attends to A. With separate $W^Q, W^K$ not even the raw scores are symmetric.

### 4.4 Multi-head self-attention

![[appunti/assets/2026-09-22/transformers-080.png]]

Run **several attention heads in parallel**, each with its **own** $W^Q, W^K, W^V$. Each head produces its own attention distribution and its own output $M_1,\dots,M_H$. The outputs are then combined, via another learned transformation ($W^O$), into one representation per token. That representation is passed to the next layer.

**Why?** One attention distribution per token can only focus on one "kind" of relationship. **Different heads can learn different aspects** of the sequence: for example, one head syntax, another coreference. Which aspects they learn depends on the task and the data; nobody assigns them.

> [!example] Lab — heads are a split, not extra capacity
> `ta_sessions/Exercises/05 - Transformers/1. Transformer.ipynb` (§2). In practice the heads aren't separate modules. The input is projected **once** to $d$ dimensions and the result is **cut into $H$ chunks** of size $d/H$: `(batch, seq, d)` → `(batch, seq, H, d/H)` → `(batch, H, seq, d/H)`, so the heads act as an extra batch dimension and the same 4-line attention runs on all of them at once. Then the reshape is undone (**concatenation**) and $W^O$ mixes the heads.
>
> Consequence, from the lab's parameter count: `MultiHeadAttention(d=64, H=4)` has $4\cdot(64\cdot64+64) = 16{,}640$ parameters ($W^Q, W^K, W^V, W^O$ with biases), and **the same with 1 or 8 heads**. The compute is the same too. More heads = **more attention patterns, each in a smaller space** ($d/H$ dimensions per head).
>
> On the tiny date task, the 4 heads overlap a lot: an easy task gives little pressure to specialise.

### Menti: the two advantages of self-attention over recurrent models

The word cloud had parallelisation, better long-range dependencies, interpretability, no vanishing gradients, and also computational cost on long documents. The lecturer said most were correct, and stressed these two:

1. **Parallelisation.** Each position's attention is independent of the others, so the whole input is processed **at once**, not token by token. This is why the architecture fits **GPUs** so well. GPUs were designed for games and graphics, not for training models, but they happen to be built for exactly this kind of massively parallel computation. Transformers and GPUs "helped each other" (see [2:30:28](https://au.cloud.panopto.eu/Panopto/Pages/Viewer.aspx?id=fa3c84d3-b9d6-47b6-b8bd-b4cd0076eb7a&start=9028)).
2. **Better long-range dependencies.** Any two positions interact **directly, in one layer**. There is no chain of recurrent state updates in between. Attention with an RNN encoder only fixed the decoder↔encoder link. Self-attention removes recurrence from the encoder too.

> About "computational cost on long documents": the lecturer didn't expand on it. It's a real *disadvantage*. Every position attends to every other, so a layer costs $O(T^2)$ in the sequence length. Keep it in mind as the price of advantage 2.

### 4.5 The transformer block

![[appunti/assets/2026-09-22/transformers-082.png]]

**Multi-head attention is the main innovation.** Around it, each block has three "practical tricks":

- **Layer normalization**: normalises the outputs, for training **stability**.
- A **feed-forward network**, applied to each position separately.
- **Residual connections** (the "Add" in "Add & Norm"): add a module's input to its output. This creates "shortcut" paths for information, preventing vanishing gradients and the loss of crucial information. These are the same residuals as in [[2026-09-15 - Language Modeling (Part 2)#11. Recap from 22 Sep: gates as learned skip connections]].

So one block computes: $x' = \text{Norm}(x + \text{MultiHeadAttn}(x))$, then $x'' = \text{Norm}(x' + \text{FFN}(x'))$. Stack $N$ of these.

"Applied to each position separately" (**position-wise**) means **one and the same** FFN (`Linear → ReLU → Linear`, same weights) is applied to every position, not a different network per position. Attention is the only place where positions exchange information. (The lab's MCQ 8.4 uses exactly this as a trap, together with the order: attention first, then FFN.)

### Transformers and vanishing gradients

![[appunti/assets/2026-09-22/transformers-083.png]]

Transformers combine the two ideas from the start of the lecture:

1. **Self-attention.** Think of it as a skip connection from **every token to every other token**, a fully connected set of shortcuts across positions. The path between any two positions is **length 1**, whatever their distance, and there is no long recurrent chain over time.
2. **Residual blocks**, $x' = x + \text{Attention}(x)$ and $x'' = x' + \text{FFN}(x')$. These keep gradients flowing through **many stacked layers**.

> **Attention handles sequence distance; residuals handle network depth.**

(See [2:35:37](https://au.cloud.panopto.eu/Panopto/Pages/Viewer.aspx?id=fa3c84d3-b9d6-47b6-b8bd-b4cd0076eb7a&start=9337) for the "self-attention as fully connected skip connections" framing.)

### Summary: fighting vanishing gradients

![[appunti/assets/2026-09-22/transformers-084.png]]

| method | what to do | why it helps |
|---|---|---|
| change activation | use less saturating activations (ReLU instead of sigmoid) | larger derivatives |
| skip connections | add a shorter, **weighted** shortcut path | less repeated shrinking |
| residual blocks | $x + \mathcal F(x)$, an **identity** shortcut | direct gradient highway |
| LSTM / GRU | gated additive state path | preserves information over time |
| Transformers | attention + residual blocks | short token-to-token paths and easy deep training |

> **Vanishing gradients are solved by making learning paths shorter, additive, or both.** Skip = weighted shortcut; residual = identity shortcut; LSTM/GRU = gated additive shortcut.

The lecturer said this slide is a good recap of the last two lectures.

### 4.6 The decoder: masked self-attention and cross-attention

![[appunti/assets/2026-09-22/transformers-090.png]]

The full encoder is a stack of transformer blocks. The **decoder** block is similar, with **three** sub-layers instead of two:

1. **Masked** multi-head self-attention, over the tokens generated so far.
2. Multi-head **cross-attention** over the **final-layer encoder outputs**.
3. A feed-forward network.

(Each has Add & Norm.) On top of the decoder, a linear layer + softmax gives the next-token distribution.

#### Menti: what's the issue with self-attention in the decoder?

**It can see the future.** Self-attention lets every token attend to *every* other token. In the encoder that's fine, because the whole input is given. The decoder **generates**, and at generation time the future tokens don't exist yet.

At **training** time you do have the whole target sentence (e.g. the parallel Danish–English pair). If the decoder could attend to it, it would learn to copy the next token instead of predicting it. It would look great on the training data and then fail in production, where the future isn't there. The lecturer's analogy: giving someone the question *and* the answer. If they can look at the answer, they don't learn (see [2:43:30](https://au.cloud.panopto.eu/Panopto/Pages/Viewer.aspx?id=fa3c84d3-b9d6-47b6-b8bd-b4cd0076eb7a&start=9810)). This is the same point as "bidirectional is useless for generation" in [[2026-09-15 - Language Modeling (Part 2)#5. Bidirectional RNNs]].

#### Masked multi-head attention

![[appunti/assets/2026-09-22/transformers-089.png]]

**Mask the attention scores of future tokens so their attention is 0.** Before the softmax, set the score to $-\infty$ for every key that comes *after* the query:

$$a_{st} = \frac{(W^Q h^\ell_s)^\top (W^K h^\ell_t)}{\sqrt d} \;\;\Longrightarrow\;\; a_{st} := -\infty \text{ for } t > s \;\;\Longrightarrow\;\; \alpha_{st} = \frac{e^{a_{st}}}{\sum_j e^{a_{sj}}} = 0$$

Because $e^{-\infty} = 0$, future positions get exactly zero weight, and the remaining weights still sum to 1. In practice you use a large negative number instead of $-\infty$.

**Why this matters:** you **keep the parallelism**. The mask is just a fixed (triangular) matrix applied to the score matrix. All positions are still computed at once during training, but each one only "sees" its past. You get causality without going back to sequential computation.

> ⚠️ **Exam (2025 Q6 = mock exam, rated easy):** *In a Transformer encoder-decoder model, each decoder block utilizes masked self-attention followed by encoder-decoder cross-attention. Which of the following correctly explains the role of masked self-attention in the decoder?*
> - ❌ **A.** It enables the decoder to attend to all encoder outputs, aligning the input and output sequences. That's **cross-attention** (below), not masked self-attention.
> - ❌ **B.** It randomly masks part of the encoder outputs to act as a regularization mechanism similar to dropout, preventing overfitting. The mask is a **fixed triangular** matrix, not random, and it hides the decoder's *own* future tokens, not encoder outputs.
> - ❌ **C.** It allows the decoder to attend to all positions in the output sequence during training to improve parallelism. The opposite. Parallelism comes from computing every position in one pass (teacher forcing), and the mask is what makes that safe.
> - ✅ **D.** It ensures the decoder does not attend to future tokens during training to maintain autoregressive property.

> [!example] Lab — causal mask, padding mask, and why before the softmax
> `ta_sessions/Exercises/05 - Transformers/0. Attention.ipynb` (§3) and `1. Transformer.ipynb` (§4–5). There are **two** masks, built the same way (`True` = may attend, forbidden scores set to $-\infty$):
> - **Causal mask** = `torch.tril(ones(L, L))`: query $i$ sees keys $0..i$. Only in the **decoder's self-attention**.
> - **Padding mask**: sentences in a batch are padded to the same length, and no query may attend to a `<pad>` **key**. One entry per key, broadcast over all queries. Used in the encoder's self-attention *and* in cross-attention (both have source positions as keys). The lecture didn't mention this one.
>
> **Why $-\infty$ on the scores rather than 0 on the weights?** Zeroing weights after the softmax leaves rows that **no longer sum to 1**, each scaled down by a different amount; you'd have to renormalise by hand. $-\infty$ before the softmax removes the key *and* renormalises over what's left in one go.
>
> **The `<pad>` rows as *queries* aren't masked**, and that's fine: their output is garbage, but nothing reads it (no real token attends to them, and the loss ignores those positions with `ignore_index=PAD`). Masking a whole row would even break things: a row that is $-\infty$ everywhere makes the softmax return `NaN`.
>
> **What goes wrong without the causal mask** (lab 1, Q5.4): with teacher forcing, position $t$'s decoder input at $t+1$ *is* the label it has to predict. The model learns to **copy from the future**: training loss drops to ≈ 0 almost immediately, then at inference there's no future and the output is garbage. Nothing crashes and no shape is wrong. That's why the lab tests masks by **behaviour**: changing target tokens from position 6 on must not change the logits at positions 0–5, and appending `<pad>` to the source must change nothing.

#### Cross-attention

This is **the same classical attention as in the RNN encoder–decoder** (§3), inside a block:

- The **query** is the output of the decoder's masked self-attention (a decoder state).
- The **keys and values** are the outputs of the **final** encoder block.

Again, a representation from the decoder is used to **attend** to the encoder outputs.

Shapes make the difference concrete (lab 1, Q3.2): a weight matrix has one row per query and one column per key, so decoder self-attention is `tgt_len × tgt_len`, cross-attention is `tgt_len × src_len`. **Cross-attention needs no causal mask**: the source is fully known before the first output token is written, at training *and* at inference. The only thing to hide there is padding. Cross-attention is also **the only place where information flows from source to target**.

> ⚠️ **Exam (2025 Q9 = mock exam, rated medium):** *In a BART/T5 model, what are the keys and queries for the cross-attention module in the decoder?*
> - ❌ **A.** Decoder's token embeddings as keys and encoder outputs as queries. Backwards: the queries come from the decoder.
> - ❌ **B.** Decoder hidden states as keys and output of the final encoder layer as queries. Backwards too.
> - ❌ **C.** Encoder hidden states as keys and decoder hidden states as queries. Differs from D by a few words: the keys are the output of the *final* encoder layer, not "encoder hidden states" in general.
> - ✅ **D.** The output of the final encoder layer as keys and decoder hidden states as queries (and the same encoder output as values).
>
> The options are easy to misread: check *which side* gives the keys, then *which layer*.

### 4.7 Position embeddings (preview — covered next lecture)

> **Recurrent models provided word-order information. Does self-attention?**

**No.** Self-attention is a weighted sum over a *set*. Shuffle the input tokens and each token's output is the same, just in a different order. There is nothing in it that knows which position is first. So Transformers **add a position embedding** $p_i$ to each input embedding (the ⊕ with the wave symbol in the architecture diagram).

![[appunti/assets/2026-09-22/transformers-093.png]]

- **Early position embeddings (the original paper)** were **sinusoids offset by a phase shift proportional to the position**:
  $$p_i = \big[\sin(i/10000^{2\cdot1/d}),\ \cos(i/10000^{2\cdot1/d}),\ \dots,\ \sin(i/10000^{2\cdot\frac d2/d}),\ \cos(i/10000^{2\cdot\frac d2/d})\big]$$
  Each pair of dimensions oscillates at a different frequency.
- **In practice, the easiest thing is to learn position embeddings from scratch.**
- **Menti: what's a disadvantage of learned position embeddings?** **Poor generalization to sequences longer than the maximum position embedding you've learned.** If you trained positions 1–512, there's simply no vector for position 513.

The next lecture finished this topic: see [[2026-09-29 - Pre-trained Language Models#1. Position embeddings]] (addition vs concatenation, the clock intuition, and the lab on RoPE, ALiBi and NoPE).

> [!example] Lab — order-blindness, shown and fixed (lab material, not yet lecture)
> `ta_sessions/Exercises/05 - Transformers/0. Attention.ipynb` (§4).
> - **The experiment:** shuffle the input rows, run self-attention, and the output is the original output shuffled the same way (self-attention is **permutation-equivariant**). So *dog bites man* and *man bites dog* give every word **the same vector**, and pooled into one sentence vector they're indistinguishable. Same limitation as averaging word vectors ([[2026-09-01 - Neural Networks and Word Embeddings]]).
> - **More of the same doesn't help.** $W^Q, W^K, W^V$, more heads, the FFN and layer norm are all applied to every position with the same weights, so they're permutation-equivariant too, and so is any stack of them. The order information isn't in the input; it has to be **put there**. (The causal mask is an exception: it treats positions differently, so a decoder leaks some order. Encoders have no such mask.)
> - **Sinusoidal encoding**, in the paper's indexing with $i = 0,\dots,d/2-1$ (the slide's formula above counts the pairs from 1 instead, which shifts every frequency by one step; the idea is identical):
>   $$PE_{(pos,\,2i)} = \sin\!\left(\frac{pos}{10000^{2i/d}}\right),\qquad PE_{(pos,\,2i+1)} = \cos\!\left(\frac{pos}{10000^{2i/d}}\right)$$
>   It's **added** to the embeddings and is **not trainable** (a fixed buffer).
> - **Why many frequencies:** like the hands of a clock. Low dimensions oscillate fast (dim 0 = $\sin(pos)$) and tell neighbours apart precisely but repeat; high dimensions have wavelengths up to $2\pi\cdot10000$ and never repeat within a sentence but barely move from one position to the next. Together they give each position a **unique fingerprint** at fine and coarse resolution.
>
> `1. Transformer.ipynb` uses the other option, **learned** position embeddings (`nn.Embedding` indexed by `0, 1, 2, …`). Its optional §7 retrains the same model with `nn.Transformer` + sinusoids, and it reached ~83% exact match after 10 epochs vs 99.7%. The lab explicitly does **not** blame the sinusoids: several details differ (shared embeddings, initialisation, extra layer norms) and it catches up with more epochs. The lesson it draws: check initialisation, optimiser and training budget before crediting or blaming an architecture.

### 4.8 From the lab: a real pre-trained translator, BLEU, and fine-tuning

Not in the lecture. `ta_sessions/Exercises/05 - Transformers/2. Pretrained Translation.ipynb` takes `Helsinki-NLP/opus-mt-en-fr`: the same encoder–decoder architecture as above (6+6 layers, `self_attn` with `q/k/v/out_proj`, `encoder_attn` = cross-attention), at full size.

- **Where the parameters are.** 75M in total, and **~41% in the token embedding** ($59{,}514$ subwords $\times$ 512). The attention + FFN layers of all 12 layers together are ~44M. That's already the economical version: one matrix is shared by the encoder input, the decoder input and the output layer (**weight tying**). With three separate matrices the embeddings alone would be ~91M, two thirds of the model. In the toy date model (40 characters × 64) embeddings were negligible.
- **Why subwords.** Characters make sequences long, and attention is $O(T^2)$. Whole words make the vocabulary huge and still miss every rare word and name. Subwords sit in between: frequent words are one token, rare ones split (`▁` marks a word start). How the vocabulary is built comes later in the course.
- **BLEU** counts how many of the output's word **n-grams (n = 1–4)** also occur in the human **reference** translation, with a penalty for outputs that are too short. Range 0–100. The model scored **~20** on sentences from novels, yet the translations were fluent and mostly correct. Why so low: literary references are free translations; there's **one** reference, and every legitimate alternative counts as an error; typographic conventions differ (« » and dashes vs `"`), and each mismatch also breaks the longer n-grams around it; the automatic sentence alignment is noisy. **BLEU has no absolute meaning.** It's for **comparing systems on the same test set**, read alongside actual outputs.
- **Fine-tuning** on 1,000 in-domain sentence pairs for one epoch: BLEU 20.2 → 22.0. What changed is mostly **style/domain conventions** (dashes for dialogue, literary tenses like the passé simple), not translation ability, which all comes from pre-training. Caveats from the lab: ±2 BLEU on 200 sentences is at the edge of measurable, and fine-tuning longer risks **forgetting** the old domain.
- **Generalisation.** The toy date model hit 99.7% exact match on a test set from the **same generator** as the training data, then failed on anything outside it: `2026 september 12` → `2012-09-20`, `12 september 2250` → `2052-09-12`, and it happily outputs `2026-02-31`. It fails **silently**: a wrong output is a well-formed date produced with the same confidence. An in-distribution test set says nothing about inputs from anywhere else.

---

## Further resources (from the slides)

- *The Annotated Transformer*: https://nlp.seas.harvard.edu/2018/04/03/attention.html
- *The Illustrated Transformer*: https://jalammar.github.io/illustrated-transformer/. The lecturer recommended it for its visual explanations.
- "Only basics presented here today! Many modifications to the initial transformer exist."

---

## Da ricordare

- **Encoder–decoder:** encoder (can be **bidirectional**) → representation → **autoregressive, unidirectional** decoder. Trained **end-to-end** with cross-entropy $\sum_t -\log P(y_t\mid y_{<t}, x)$. **Teacher forcing** at training (gold target shifted right, all positions in one pass); **greedy decoding** at inference (one forward pass per token). It needs **paired data**, which is hard to find. Bidirectional $2d$ state → decoder: **project** with $W_{dec}[h^{fwd};h^{bwd}]+b$, so no need to double the decoder's size.
- **Temporal bottleneck:** the whole input is squeezed into **one vector** (⚠️ 2025 Q5: *the* key RNN disadvantage; RNNs are linear in $T$, share weights across steps, and are nonlinear). Attention fixes it with **direct connections from decoder states to all encoder states**. This also helps vanishing gradients and gives **interpretability** (alignment for free).
- **Attention recipe:** score $a_n = f(h^e_n, h^d_t)$ → **softmax** $\alpha_n$ → weighted sum $\sum_n \alpha_n h^e_n$. **Query** = decoder state, **keys/values** = encoder states. It's recomputed at **every** decoder step. Output layer gets $[\tilde h^d_t; h^d_t]$ (**concatenated**, not added). Weights are in $[0,1]$ **and** sum to 1.
- **Scaled dot product:** $\text{softmax}(QK^\top/\sqrt d)\,V$ with learned $W^Q, W^K, W^V$. **Self-attention:** Q, K and V all come from the same sequence. Every element is a key and a value, one is the query, and **the position attends to itself too**. The weight matrix is **not symmetric** (row-wise softmax).
- **Why $\sqrt{d_k}$** (⚠️ 2025 Q8)**:** $\text{Var}(q\cdot k) = d_k$ → large scores → **saturated softmax → ≈ zero gradients** (at init, by chance). It's the **scores** that are divided, not the keys, and it's not about cost. **Multi-head:** heads are **concatenated** + $W^O$ (**not averaged**); #parameters **doesn't depend on #heads** ($d$ is split into $H$ chunks of $d/H$).
- **Two advantages over RNNs:** **parallelisation** (no recurrence, fits GPUs) and **direct long-range interactions** (path length 1). Price: $O(T^2)$ in the sequence length (more precisely $O(T^2 d_k)$: **a larger $d_k$ does make the attention matrix costlier**, mock-exam trap). An RNN is *linear* in $T$; its problem is the lack of parallelism.
- **Transformer block** = multi-head attention + **Add & Norm** + FFN + **Add & Norm**. **Attention handles sequence distance; residuals handle network depth.** The FFN is **position-wise** = the *same* network at every position. Decoder block = **masked** self-attention ($a_{st} := -\infty$ for $t>s$) → **cross-attention** (query from decoder, keys/values from the **final** encoder layer; ⚠️ exam: the options differ by one word) → FFN.
- **Masks** (⚠️ 2025 Q6: masked self-attention = no attending to future tokens)**:** set **scores** to $-\infty$ **before** the softmax (so rows still sum to 1). **Causal** mask only in **decoder self-attention**; the encoder and cross-attention mask only **padding** keys. Not random (≠ dropout). Without the causal mask + teacher forcing, the model copies the next token: loss ≈ 0, garbage at inference.
- **Self-attention has no notion of order** (permutation-equivariant; extra heads/layers/projections **don't** fix it), so add **position embeddings**: sinusoidal (original) or learned (easiest in practice, but **can't generalise past the max trained length**).
