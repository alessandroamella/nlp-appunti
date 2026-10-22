Continuation of [[2026-09-08 - Language Modeling (Part 1)]], which ended with the Elman RNN. This lecture covers three things. First, **how an RNN is trained**: backpropagation through time. Second, **why training breaks down on long sequences**: vanishing gradients. Third, **the fix**: gated RNNs (LSTM, GRU).

The deck opens with a recap of Part 1 (n-grams, perplexity, fixed-context NLMs, the Elman network, slides 6–20). It's not repeated here. The Elman slides weren't a recap in practice: the 08 Sep lecture ran out of time before them, so they were presented here for the first time. See [[2026-09-08 - Language Modeling (Part 1)#16. Recurrent neural networks (RNNs) — introduction]].

One loose end from Part 1 was closed in this lecture: **what should $h_0$ be?** The class's answers were all accepted. (a) **Zeros**, because at the start there is no history. (b) Something **task-dependent/pre-trained**. (c) A **learnable** parameter, trained like the others (see [24:38](https://au.cloud.panopto.eu/Panopto/Pages/Viewer.aspx?id=201e3530-b7b1-48c4-ac4a-b4c6007554f3&start=1478)).

---

## 1. An RNN is a feedforward network with dynamic depth

![[appunti/assets/2026-09-15/lm-pt2-021.png]]
_Unrolling the RNN across all time steps gives the full computation graph._

> **RNN is a FFN with dynamic depth. Depth = length of the input sequence.**

If you unroll the loop, nothing recurrent is left. You have a plain feedforward chain where information flows left to right, one "layer" per token. A 5-token input gives a 5-layer network, and a 50-token input gives a 50-layer one.

The only real difference from an ordinary deep FFN is **weight sharing**. In a normal FFN every layer has its **own** weights, because each layer is meant to learn a different transformation. In the unrolled RNN, every "layer" uses the **same** $W_{hx}, W_{hh}, W_{zh}$. We want to learn one transformation, "absorb a token into the state", and apply it again and again over time (see [29:06](https://au.cloud.panopto.eu/Panopto/Pages/Viewer.aspx?id=201e3530-b7b1-48c4-ac4a-b4c6007554f3&start=1746)).

This framing matters because of what follows. **If an RNN is just a deep FFN, you train it with ordinary backpropagation.** And it inherits the usual problem of very deep networks, where gradients die on the way back (§6).

---

## 2. Backpropagation review (FFNs)

> ⚠️ **Exam:** you will **not** be asked to compute gradients by hand in the exam. You do need to understand *how* backprop works, but the arithmetic is not tested (see [38:05](https://au.cloud.panopto.eu/Panopto/Pages/Viewer.aspx?id=201e3530-b7b1-48c4-ac4a-b4c6007554f3&start=2285); repeated in the 22 Sep lecture).

The deck works through a toy network in full: input $x \in \mathbb{R}^3$, two hidden layers of 3 neurons, and one output neuron.

- Each neuron $\phi_{ij}$ computes a weighted sum of its inputs and applies an activation: $h_1 = \phi_1(W_0 x)$, $h_2 = \phi_2(W_1 h_1)$.
- Output: $u = W_{out} h_2 = w^o_1\phi_{12} + w^o_2\phi_{22} + w^o_3\phi_{32}$, then $P(\hat y) = \phi_o(u)$.
- Loss: **cross-entropy**, $\mathcal{L}(\hat y, y) = -y \log P(\hat y)$.

To find how the loss depends on a hidden neuron, apply the **chain rule** one step at a time:

$$\frac{\partial \mathcal{L}}{\partial \phi_{12}} = \frac{\partial \mathcal{L}}{\partial \hat y}\,\frac{\partial \hat y}{\partial u}\,\frac{\partial u}{\partial \phi_{12}} = \underbrace{\frac{\partial \mathcal{L}}{\partial \hat y}}_{\text{depends on label } y}\ \underbrace{\frac{\partial \phi_o(u)}{\partial u}}_{\text{depends on } \phi_o}\ w^o_1$$

One layer further back, the same pattern repeats with one more factor. With $v = w^{1}_{11}\phi_{11} + w^{1}_{21}\phi_{21} + w^{1}_{31}\phi_{31}$ as the input to $\phi_{12}$:

$$\frac{\partial \mathcal{L}}{\partial \phi_{11}} = \underbrace{\frac{\partial \mathcal{L}}{\partial \hat y}\frac{\partial \phi_o(u)}{\partial u}w^o_1}_{\text{already computed}}\ \frac{\partial \phi_{12}(v)}{\partial v}\ w^{1}_{11}$$

![[appunti/assets/2026-09-15/lm-pt2-038.png]]
_The "colour game". The green block is identical to the one already computed one layer up. Only the last factors change._

**The point of the colour-coding.** Compare the derivative for $\phi_{12}$ with the one for $\phi_{22}$, or the one for $\phi_{11}$ with the one for $\phi_{21}$. Most factors are **the same**. Only the last one or two differ. So backprop **caches** the shared upstream part and reuses it, instead of recomputing the whole chain for every weight. That is why it's cheap. Done naively, it would be enormously expensive. Hence "backward": you start at the loss and walk back one layer at a time, and each layer reuses what the layer above computed (see [38:37](https://au.cloud.panopto.eu/Panopto/Pages/Viewer.aspx?id=201e3530-b7b1-48c4-ac4a-b4c6007554f3&start=2317)).

(The slides differentiate with respect to the neuron *outputs* $\phi$. The gradient with respect to a *weight* is one more chain-rule factor on top, with the same logic.)

---

## 3. Backpropagation through time (BPTT)

Since an RNN unrolls into an FFN (§1), backprop carries over directly. You just run it through the unrolled graph. The one new thing is that you go **back through time** as well as down to the output.

Notation from the slides (the Elman network, with pre-activations named):

$$u_t = W_{hx}x_t + W_{hh}h_{t-1} + b_h,\quad h_t = \sigma(u_t) \qquad v_t = W_{zh}h_t + b_z,\quad z_t = \sigma(v_t)$$

Step by step:

- **Output → state at the same step:** $\dfrac{\partial z_t}{\partial h_t} = \dfrac{\partial\sigma(v_t)}{\partial v_t}W_{zh}$
- **State → previous state** (the new, "through time" step): $\dfrac{\partial h_t}{\partial h_{t-1}} = \dfrac{\partial\sigma(u_t)}{\partial u_t}W_{hh}$
- **Chain them** to reach further back:

$$\frac{\partial z_t}{\partial h_{t-2}} = \frac{\partial z_t}{\partial h_t}\,\frac{\partial h_t}{\partial h_{t-1}}\,\frac{\partial h_{t-1}}{\partial h_{t-2}} = \frac{\partial\sigma(v_t)}{\partial v_t}W_{zh}\;\frac{\partial\sigma(u_t)}{\partial u_t}W_{hh}\;\frac{\partial\sigma(u_{t-1})}{\partial u_{t-1}}W_{hh}$$

![[appunti/assets/2026-09-15/lm-pt2-047.png]]
_"Note that these are actually the same matrix". Every step back in time multiplies by the same $W_{hh}$._

**Keep that red note in mind.** Weight sharing is what makes RNNs cheap (the model doesn't grow with length). But it also means that going $T$ steps back multiplies by **the same matrix $T$ times**. That is the root cause of §6.

At full scale, the gradient flows back from each output $z_t$, horizontally through the chain of states, and down into each input $x_t$. Because the weights are shared, the gradient contributions for $W_{hh}$ from every time step add up.

### RNNs in practice

- Computing gradients by hand is hard. It's still a good way to **sanity-check** that a network behaves as expected.
- Modern DL libraries use **automatic differentiation**. You define only the **forward pass**, and the gradients come for free.
- The course uses **PyTorch**, so you won't compute gradients by hand in the labs either.

> [!example] Lab — truncated BPTT
> `ta_sessions/Exercises/04 - RNNs/0. RNNs.ipynb`, MCQ 6.4. Not in the lecture, but asked in the lab. Full BPTT over a sequence of thousands of steps needs every step's activations in memory and a chain of thousands of Jacobians. **Truncated BPTT** unrolls only a fixed window (e.g. 20 or 50 steps), processes the sequence in chunks, and backpropagates only inside each window. **Why:** it bounds the cost per update and the length of the Jacobian chain, so it also limits vanishing/exploding. **Price:** a dependency longer than the window can't be learned through the gradient.
>
> The MCQ: *RNNs are often trained with truncated BPTT: the network is unrolled for only a fixed number of time steps, the sequence is processed in chunks, and gradients are backpropagated only within these windows. The main reason for doing this is:*
> - ❌ **A.** To avoid overfitting by randomly dropping parts of the sequence. Nothing is dropped, and nothing is random.
> - ✅ **B.** To limit the computational cost and mitigate exploding/vanishing gradients over very long sequences.
> - ❌ **C.** To enforce fixed-length sequences as input. The input can be any length; only the gradient window is fixed.
> - ❌ **D.** To increase the parallelism of the recurrent connections. The recurrence is still sequential inside each window.

---

## 4. Deep (stacked) RNNs

![[appunti/assets/2026-09-15/lm-pt2-052.png]]
_Three stacked recurrent layers. The output $z^1_t$ of layer 1 is the input to layer 2 at the same time step. Dashed arrows show the gradient flowing back both through time and down through layers._

Nothing forces you to go only horizontally. You can also **stack** recurrent layers vertically, like adding layers to an MLP. The output sequence of layer $\ell$ becomes the input sequence of layer $\ell+1$.

The two "depths" behave differently, and the lecturer insisted on this:

| | horizontal depth | vertical depth |
|---|---|---|
| what it is | number of time steps | number of stacked layers |
| set by | the **input length**, so it **varies** per input | **you** (a hyperparameter), so it's **fixed** |
| typical value | whatever the text is | **3–4 layers** |

BPTT works the same way. Start from the top-right output, then walk back in time *and* down through the layers, step by step.

---

## 5. Bidirectional RNNs

A left-to-right RNN has a blind spot. The state at step $t$ has seen only the **prefix**. In long sequences, information from the start has also faded by the end (this is §6 in advance).

The fix is to run **two** RNNs, one left→right ($h^f$) and one right→left ($h^b$). Then combine them at each step:

![[appunti/assets/2026-09-15/lm-pt2-055.png]]

- **Concatenate** the output states at each step, $[z^f_t; z^b_t]$. You can also take the mean or max.
- **Separate parameters** for the forward and backward RNNs.
- **Important note: only useful for applications where you can use the future text for the task.**

**Where can you use it?** (Menti poll.)

- ✅ **Sequence classification** (sentiment, topic): the whole document is available before you have to output anything.
- ✅ **Sequence labelling** (POS tagging, per-token sentiment): each token's label can depend on what comes after it.
- ❌ **Next-token generation / language modelling**: the whole task is predicting the future, so the future isn't available.

There was a long discussion in class about the ❌ case. The key refinement is "*can* ≠ *should*". You *can* bolt a backward RNN onto a generator, but it runs over tokens that don't exist yet. At the position you're predicting, the backward direction has nothing to read, so it adds no information. You're "throwing compute at it without any expected gain" (see [1:20:42](https://au.cloud.panopto.eu/Panopto/Pages/Viewer.aspx?id=201e3530-b7b1-48c4-ac4a-b4c6007554f3&start=4842) for the discussion and [1:29:10](https://au.cloud.panopto.eu/Panopto/Pages/Viewer.aspx?id=201e3530-b7b1-48c4-ac4a-b4c6007554f3&start=5350) for the conclusion). The lecturer also said there is a setup where this *is* legitimate, "without cheating", and that **machine translation** is still done that way, to be shown "maybe 3 weeks from now". My reading: it's the bidirectional encoder of the encoder–decoder model, since the source sentence is fully known. That model came the very next week, see [[2026-09-22 - Seq2Seq Models and Transformers#Bidirectional encoders]].

### Wrap-up of the RNN half

- RNNs can **theoretically** model unbounded context with no growth in model size.
- **Problem:** many steps of **state overwriting** make it hard to learn **long-range dependencies**.

---

## 6. Vanishing gradients

### The symptom

![[appunti/assets/2026-09-15/lm-pt2-063.png]]

> Nearby words should affect each other more than farther ones, but RNNs make it challenging to learn **any** long-range interactions.

The deck's example is a paragraph that ends with *"Aside from writing, I've always loved **dancing**."* The clues that predict *dancing* are *jig*, *dancing in circles*, *swinging*, *spinning* and *dance steps*. They are all dozens of tokens back. By the time the RNN reaches the blank, their contribution to $h_t$ is close to zero.

### The cause

Generalise the BPTT chain from §3 to $T$ steps:

$$\frac{\partial h_t}{\partial h_{t-T}} = \prod_{i=t-T+1}^{t}\frac{\partial h_i}{\partial h_{i-1}} = \prod_{i=t-T+1}^{t}\frac{\partial\sigma(u_i)}{\partial u_i}\,W_{hh}$$

*(The deck writes the product from $i = t-T$. With $T$ steps back there are $T$ factors, which is the range above. It doesn't change anything.)*

Both factors tend to be small:

- $\dfrac{\partial\sigma(u)}{\partial u}$ is **< 1 for many activation functions**. For the sigmoid it is **at most 0.25**, and it's close to 0 whenever the unit saturates.
- $W_{hh}$ is **typically small** because of **regularisation**.

![[appunti/assets/2026-09-15/lm-pt2-064.png]]
_The sigmoid (blue) and its derivative (dashed red). The derivative peaks at 0.25 and vanishes in both tails._

A product of $T$ numbers below 1 shrinks **exponentially** in $T$. The 22 Sep recap put a number on it: $0.9^{100} \approx 0.000027$. **Small terms × many steps = vanishing gradient.** The gradient signal from a distant output barely reaches the early tokens, so the network gets almost no signal telling it to learn long-range dependencies.

> **Learning problem:** long unrolled networks will crush/squash gradients that backpropagate to earlier time steps. A typical problem in many RNNs, **especially Elman (vanilla) RNNs, due to the sigmoid activation.**

This is the irony the lecture was built around. Weight sharing is the RNN's big **advantage** (unbounded length at no extra model cost), and it's also the **curse**. The same $W_{hh}$ multiplied over and over is exactly what kills the gradient.

*(Not covered in the lecture, but the other side of the same coin: if the factors are **> 1**, the product **explodes** instead. The usual remedy there is gradient clipping. The lecture only discussed vanishing; the lab covers exploding, see below.)*

> [!example] Lab — exploding gradients and gradient clipping
> `ta_sessions/Exercises/04 - RNNs/0. RNNs.ipynb`, §4 and MCQ 6.3. The lab's training loop has one line the earlier labs didn't: `torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)`, between `backward()` and `step()`. **Gradient clipping:** if the norm of the whole gradient exceeds a threshold, rescale it down to that threshold before the update. The direction is kept and only the step size is capped. **Why:** backprop through 50 steps multiplies 50 Jacobians, and for a vanilla RNN the product occasionally **explodes**, and a single huge step throws the weights off. The notebook says clipping "makes the difference between an RNN that trains and one that does not". It fixes **exploding** only: it can't make a vanishing gradient bigger.
>
> MCQ 6.3: *In BPTT for RNNs, which factor is primarily responsible for the vanishing gradient problem?*
> - ✅ **A.** The gradient repeatedly involves products of Jacobians $\partial h_t/\partial h_{t-1}$ of the recurrent transition across many time steps. That is the formula above. Norms below 1 give vanishing, norms above 1 give exploding.
> - ❌ **B.** The gradient becomes dominated by the output layer's softmax cross-entropy derivative when the sequence length grows.
> - ❌ **C.** The gradient decays mainly because the bias terms accumulate through the recurrent connections.
> - ❌ **D.** The gradient is suppressed because the input-to-hidden transformation has fewer parameters than the hidden-to-hidden recurrence.
>
> B, C and D each blame something that doesn't get *multiplied once per step*, which is what makes the decay exponential in the distance.
>
> *Small difference from the slides:* PyTorch's `nn.RNN` uses $\tanh$, not the sigmoid, so the lab writes the Jacobian as $\text{diag}(1-h_k^2)\,W_{hh}$. The derivative of $\tanh$ peaks at 1 instead of 0.25, which is gentler, but it's still ≤ 1 and still near 0 when the unit saturates, so the argument is the same.

### Menti: how would you fix it?

The class suggested **skip/residual connections**, **ReLU** (or other activations), and **normalisation layers**. The lecturer's take:

- **Different activation (ReLU):** its derivative isn't forced below 1, so the systematic shrinking goes away. Reasonable.
- **Skip connections:** give the gradient a shorter path. They help, but "skip connections is not the way we solve this problem in language models or in recurrent neural networks" (see [1:46:24](https://au.cloud.panopto.eu/Panopto/Pages/Viewer.aspx?id=201e3530-b7b1-48c4-ac4a-b4c6007554f3&start=6384)). The actual solution is **gating**, which is closely related (§11).

---

## 7. Gated RNNs: the idea

![[appunti/assets/2026-09-15/lm-pt2-068.png]]

$$\underbrace{h_t = \sigma(W_{hx}x_t + W_{hh}h_{t-1} + b_h)}_{\text{Elman}} \qquad \underbrace{h_t = h_{t-1}\odot f + \text{func}(x_t)}_{\text{gated network (abstraction)}}$$

- A **gate** value $f$, with $0 < f < 1$, decides **how much of the previous state moves on** to the next step. $\odot$ is element-wise multiplication (Hadamard product).
- In the Elman update, $h_{t-1}$ sits **inside** the squashing activation. So every step back in time costs a factor of $\sigma'(\cdot) \le 0.25$ (§6).
- In the gated update, $h_{t-1}$ is **outside** any activation. It's carried forward by an **additive** path, scaled only by $f$. So the gradient along that path is simply $f$, which is not automatically squashed. This reduces vanishing gradients.

Intuition from the lecturer: this works "something like a skip connection, but maybe stronger", because $f$ is **learned** and **input-dependent**. The network can decide, per token, which past information to keep and which to drop (see [1:48:32](https://au.cloud.panopto.eu/Panopto/Pages/Viewer.aspx?id=201e3530-b7b1-48c4-ac4a-b4c6007554f3&start=6512)).

---

## 8. Long Short-Term Memory (LSTM)

![[appunti/assets/2026-09-15/lm-pt2-069.png]]

**Gates**, each a sigmoid so each lies in $(0,1)$, each computed from the current input and the previous hidden state:

$$f_t = \sigma(W_{fx}x_t + W_{fh}h_{t-1} + b_f) \quad\text{(forget)}$$
$$i_t = \sigma(W_{ix}x_t + W_{ih}h_{t-1} + b_i) \quad\text{(input)}$$
$$o_t = \sigma(W_{ox}x_t + W_{oh}h_{t-1} + b_o) \quad\text{(output)}$$

**State updates** ($\phi$ is the activation, typically $\tanh$):

$$\tilde c_t = \phi(W_{cx}x_t + W_{ch}h_{t-1} + b_c) \quad\text{(candidate update)}$$
$$c_t = i_t \times \tilde c_t + f_t \times c_{t-1} \quad\text{(cell state)}$$
$$h_t = o_t \times \phi(c_t) \quad\text{(hidden state)}$$

($\times$ here is element-wise, the same as $\odot$ above.)

### Two memories

The LSTM keeps **two** states:

- **Hidden state $h_t$** = **short-term** memory. It's what gets exposed as output and fed into the next step's gates.
- **Cell state $c_t$** = **long-term** memory. It's the additive "conveyor belt". Hence the name: a *long* short-term memory.

The cell state update is the gated abstraction of §7 made concrete. $f_t$ decides how much of the old $c_{t-1}$ survives, and $i_t$ decides how much of the new candidate $\tilde c_t$ gets written.

LSTMs date from the 1990s (Hochreiter & Schmidhuber, 1997). The lecturer's anecdote: it took **almost 13 years** before people could use them meaningfully in practice. The concept looked nice in theory, but for those years it was "a bit unusable".

### The three gates, one at a time

The deck's running example is *"I went to **the lecture**"*: blue marks the past context, red the current token.

| gate | controls | at 1 | at 0 |
|---|---|---|---|
| **forget** $f_t$ | how much of the **old cell state** is kept | remember the past | forget everything up to now |
| **input** $i_t$ | how much of the **new candidate** is written into the cell | take in the current step | ignore the current time step |
| **output** $o_t$ | how much of the cell state is exposed in $h_t$, and so influences the gates at the next step | expose everything | expose nothing |

The slide asks **"What might be ignored?"** for the input gate. The answer given was words that carry almost no meaning, such as *the*. In *"I went to the lecture"*, the input gate on *the* would plausibly be driven close to 0. This isn't hard-coded: the gate *learns* this from data during training (see [2:28:45](https://au.cloud.panopto.eu/Panopto/Pages/Viewer.aspx?id=201e3530-b7b1-48c4-ac4a-b4c6007554f3&start=8925)).

### What does the cell state track?

Take a trained LSTM, pick **one dimension** of $c$, and colour each input character by that dimension's activation ($\tanh$, so from −1 blue to +1 red). Some dimensions have a recognisable meaning. The lecturer said only that the examples come from "a very nice blog post". The figures look like Karpathy's post on character-level RNNs, but that's my guess.

- **Source code (Linux kernel):** the answers pointed to a dimension tracking **scope/indentation depth**, which changes as you enter `if`/`for` blocks.
- **Online news / novel text:** a dimension that switches on **inside quotation marks**. Some people answered "the speaker", but a speaker-tracking unit would have to react to the speaker's name specifically, and it doesn't. It flips exactly at the quote marks.

> ⚠️ **Exam:** you won't get interpretation questions this under-specified. On 15 Sep: the teachers will be in the exam room, and an under-specified question would be removed. On 22 Sep: a question like this would come with more information, you can ask clarification questions in the room, and a clarification that applies to everyone goes on the whiteboard/projector (see [2:25:18](https://au.cloud.panopto.eu/Panopto/Pages/Viewer.aspx?id=201e3530-b7b1-48c4-ac4a-b4c6007554f3&start=8718); repeated at the start of the 22 Sep lecture, [13:46](https://au.cloud.panopto.eu/Panopto/Pages/Viewer.aspx?id=fa3c84d3-b9d6-47b6-b8bd-b4cd0076eb7a&start=826)).

### Quick comprehension questions (Menti)

Worked out from $c_t = i_t\tilde c_t + f_t c_{t-1}$:

- **When would the input gate learn to be 0?** For inputs that carry nothing worth storing, such as stopwords (see above), padding, or noise.
- **Forget gate = 0:** $c_t = i_t\tilde c_t$. The whole past is erased, and the cell holds only what the current step wrote.
- **Forget = 0 and input = 0:** $c_t = 0$. The past is gone *and* nothing new is written, so the cell **resets and starts from scratch**. The lecturer compared it to the GRU's *reset* idea (§9).
- **Forget = 1 and input = 1:** $c_t = c_{t-1} + \tilde c_t$. Nothing is ever forgotten, and every update is added on top. The class debated whether this is "a vanilla RNN". It isn't, because the output gate still chooses what reaches $h_t$. A student suggested it looks like a **skip connection**. The lecturer agreed it was an interesting angle and resolved it the following week (§11).

---

## 9. Gated Recurrent Unit (GRU)

![[appunti/assets/2026-09-15/lm-pt2-080.png]]

The same goal as the LSTM (a gated path that doesn't damp the gradient every step), but a lighter architecture:

$$\underbrace{h_t = (1-\mathbf z)\odot h_{t-1} + \mathbf z\odot \text{func}(x_t, h_{t-1})}_{\text{GRU}} \qquad \underbrace{h_t = h_{t-1}\odot \mathbf f + \text{func}(x_t)}_{\text{LSTM (abstraction)}}$$

In full:

$$z_t = \sigma(W_z x_t + U_z h_{t-1} + b_z) \quad\text{(update gate)}$$
$$r_t = \sigma(W_r x_t + U_r h_{t-1} + b_r) \quad\text{(reset gate)}$$
$$h_t = (1-z_t)\circ h_{t-1} + z_t \circ \tanh\!\big(W_h x_t + U_h(r_t \circ h_{t-1}) + b_h\big)$$

- **Update gate $z$**: how much the hidden state gets updated. It's **one** gate that plays the role of both the forget and the input gate. What you keep ($1-z$) and what you write ($z$) are tied together and always sum to 1.
- **Reset gate $r$**: how much of the previous state is used when computing the candidate. $r \to 0$ means "compute the new content as if there were no history".
- **Single hidden state**, with no separate cell state and no output gate. That makes the mixing rule simpler than in LSTMs.
- The difference from the LSTM that the lecturer pointed out: the GRU still uses the previous state ($r_t \circ h_{t-1}$) **inside** the activation that computes the candidate.

**Trade-offs** (slide):

- **Pro:** typically **faster to train**, and sometimes works **better** than LSTMs. The lecturer's point was that the most complex thing is not automatically the best.
- **Con:** **theoretically less powerful**, e.g. **it can't count**. *(Not explained in class. My understanding, from Weiss et al. 2018: the LSTM cell state is additive and not bounded, so it can act as a counter. The GRU's state is always a convex mix of the old state and a $\tanh$ output, which keeps it bounded, so it can't grow like a counter.)*

*(Heads-up on convention: the deck puts $z$ on the **new** content and $1-z$ on the old state. Some sources, including Cho et al.'s original paper, swap the roles. If you meet the other form, it's the same model with $z \leftrightarrow 1-z$. **PyTorch's `nn.GRU` uses the other form**, $h_t = (1-z_t)\odot\tilde h_t + z_t\odot h_{t-1}$, so there $z_t\approx1$ means *keep*. The Week-4 lab points this out. It matters as soon as you touch the gate's bias by hand, see §10.)*

---

## 10. Comparisons

### RNN vs LSTM/GRU

![[appunti/assets/2026-09-15/lm-pt2-083.png]]

| | Vanilla RNN | LSTM |
|---|---|---|
| state kept by | hidden-state feedback, $h_t = \sigma(W_{hx}x_t + W_{hh}h_{t-1} + b_h)$ | cell value, $c_t = i_t\tilde c_t + f_t c_{t-1}$ |
| gradient step back in time | **systematically squashed by the sigmoid** | $\dfrac{\partial c_t}{\partial c_{t-1}} = f_t$, **set by the forget gate** |
| can it still vanish? | yes, by construction | yes, but **only if the forget gate closes** |

That last line is the whole argument. In a vanilla RNN, vanishing is **built in**. In an LSTM, it happens only when the network has **learned** to forget.

*(Caveat, also made explicit in the 22 Sep recap: $\partial c_t/\partial c_{t-1} = f_t$ is the gradient along the **direct** cell-state path only. Gradients also flow through the gates and the candidate, because those depend on $h_{t-1}$. Those are extra paths and don't contradict the point above.)*

> [!example] Lab — the same comparison on real data (AG News)
> `ta_sessions/Exercises/04 - RNNs/0. RNNs.ipynb`. Topic classification of news articles (4 classes), 50 tokens each: embedding → recurrent layer → linear layer on the **final hidden state**. Typical macro-F1 after 3 epochs: **RNN ≈ 0.86, LSTM ≈ 0.91, GRU ≈ 0.91**. The gated models are ahead already after the first epoch.
> - **It's the gating, not the parameters.** The three models have 3.94M, 4.04M and 4.01M parameters. About 99% of that is the embedding matrix ($30{,}522 \times 128$), so making the recurrent layer 4× bigger adds only 2.5%. The GRU has fewer parameters than the LSTM and does just as well.
> - **Padding shows "state overwriting" concretely** (mechanism in [[2026-09-08 - Language Modeling (Part 1)]], §16): pad steps aren't skipped, so with right padding the vanilla RNN drifts towards an article-independent state and the gradient has to cross the useless steps to reach a real token. An LSTM/GRU can learn to close its gates on padding and just carry its state. The lab pads on the **left** to sidestep it.
> - **Why the vanilla RNN still reaches 0.86:** the task is easy. A few topical words near the end of the window are often enough, and news puts the key information first.

> [!example] Lab — at initialisation, the forget gate is half closed
> `ta_sessions/Exercises/04 - RNNs/1. LSTM vs GRU.ipynb`. This refines the table above. "It only vanishes if the forget gate closes" assumes the forget gate *starts* open. It doesn't. PyTorch initialises all weights and biases near 0, so at initialisation $f_t \approx \sigma(0) = 0.5$: **the cell state is halved at every step**. After $T$ steps, $0.5^{T}$ of the first token's trace is left, and $\partial c_T/\partial c_1 = \prod f_t$ is just as small. That's $\approx 2\cdot10^{-6}$ after 19 steps and $\approx 2\cdot10^{-12}$ after 39.
>
> So the LSTM can **represent** the solution (with $f \approx 1$ it would carry information across any distance), but gradient descent can't **find** it from there. The gradient that would say "open the forget gate" has to travel back through the half-closed forget gate itself. It's a vanishing-gradient problem created by the initialisation, in the architecture designed to avoid it.
>
> **The fix: forget-gate bias initialisation** (Gers et al. 2000; Jozefowicz et al. 2015). Set the forget-gate bias to a positive value, e.g. $b_f = 3$, so $f \approx \sigma(3) \approx 0.95$ at initialisation. The LSTM then **remembers by default and learns when to forget**, instead of the reverse. Some libraries do this by default. PyTorch doesn't.

### Which is better, LSTM or GRU?

The slide shows the results of Chung et al. (2014) on speech and music modelling. **Neither wins across the board.** Which one is better depends on the data, the task and the activation, so you have to try both.

> [!example] Lab — LSTM vs GRU: the gap was the initialisation
> `ta_sessions/Exercises/04 - RNNs/1. LSTM vs GRU.ipynb`. On AG News the two tie (≈0.91 macro-F1 each, see above). To isolate long-term memory, the follow-up uses a synthetic **recall task**: the first token is one of 8 symbols, all the others are noise, and the model must output the first symbol at the end. Chance level is $1/8 = 12.5\%$. Sequence length = dependency distance.
>
> | model (PyTorch defaults unless stated) | solves distance… |
> |---|---|
> | LSTM | 10 only. At 20, 30 and 40 it sits at 12.5% for all 2,500 steps. |
> | GRU | 10, 20 and 30 (needing ~2,250 steps for 30). Fails at 40. |
> | **LSTM, forget bias = 3** | **all of them, in ~50 steps each** |
> | GRU, update-gate bias = 3 | all of them, in ~100 steps each |
>
> - Failure is **all-or-nothing**: accuracy is either ≈100% or exactly chance. The model doesn't remember "a bit less". Either it finds the solution or it never leaves the plateau (loss ≈ $\ln 8$).
> - The first two rows make the GRU look "better at long-term memory". The ranking **flips** after changing 32 numbers at initialisation (the explanation is in the callout above). With both models initialised to remember, it's a **tie** again. **The gap was two default initialisations, not two architectures.** PyTorch's GRU just starts out leaking a bit less.
> - The GRU fix uses a **positive** bias only because of PyTorch's convention ($z\approx1$ = keep, see §9). With the slides' convention you'd need a **negative** bias. A sign error silently makes the model forget *faster*.
> - **Lesson for reading benchmark tables:** "LSTM x%, GRU y%" compares whole *recipes* (architecture + initialisation + optimiser + learning rate + budget + seed). This fits the lecture's "neither dominates", and it's what Jozefowicz et al. (2015) found at scale.

### Advantages and disadvantages of LSTMs/GRUs

- **Advantage:** they fix vanishing gradients through a gated, additive state update, so they can actually **learn long-range dependencies**. Vanilla RNNs can only do this in theory.
- **Disadvantage: more parameters.** An LSTM has **four** sets of $(W_{\cdot x}, W_{\cdot h}, b_\cdot)$: $f$, $i$, $o$ and $\tilde c$. A vanilla RNN has one. So an LSTM is about **4×** the recurrent parameters, and a GRU about 3×. Parameters are **shared across time**, but **not across gates**. Each gate has its own.

> [!example] Lab — where 4× and 3× come from, and what they don't tell you
> `ta_sessions/Exercises/04 - RNNs/0. RNNs.ipynb` §5.3 and `1. LSTM vs GRU.ipynb` §6. One "RNN-sized block" is $E\cdot H + H\cdot H + 2H$: input matrix, recurrent matrix, and PyTorch's **two** bias vectors. With $E = H = 128$ that's 33,024. The LSTM has 4 blocks ($f, i, o, \tilde c$) = 132,096, and the GRU has 3 ($r, z, \tilde h$) = 99,072.
>
> **"Faster to train" ≠ fewer parameters.** On the TA's CPU, the GRU (25% fewer parameters) was about **2.5× *slower*** per training step than the LSTM. At this tiny size, per-step overhead and how well each layer's kernel is optimised dominate, not the arithmetic. The results vary with the hardware. Parameter counts and FLOPs are a poor proxy for speed: **measure it**. This qualifies the slide's "GRU: typically faster to train" (§9): it's a statement about the architecture, not a guarantee for a given implementation.

### Could there be better architectures?

Jozefowicz et al. (2015) searched automatically over thousands of recurrent cell variants (MUT1–3 on the slide). Some beat the LSTM/GRU on particular tasks, but none did so consistently. The lecturer's comment: at that point it "started becoming more like a hack or an engineering exercise". It's for background only.

### Wrap-up

- RNNs can **theoretically** model unbounded context.
- **In practice, vanishing gradients stop vanilla RNNs from learning useful long-range dependencies.**
- **LSTMs and GRUs** are recurrent variants that **mitigate** the vanishing gradient problem.

---

## 11. Recap from 22 Sep: gates as learned skip connections

The "both gates = 1 → skip connection?" question from §8 was left open. The next lecture opened by settling it, with three slides that tie the vanishing-gradient fixes together (see [19:38](https://au.cloud.panopto.eu/Panopto/Pages/Viewer.aspx?id=fa3c84d3-b9d6-47b6-b8bd-b4cd0076eb7a&start=1178)).

![[appunti/assets/2026-09-22/transformers-014.png]]

**Skip connection in an RNN.** Add a shortcut from a few steps back, e.g. $h_t = \sigma(W_1h_{t-1} + W_5h_{t-5} + W_xx_t)$. During backprop the gradient uses **both** routes, and the contributions add. The shortcut crosses fewer multiplications, so it shrinks less. But the shortcut is **still weighted** ($W_5$), so it can still attenuate the gradient. It helps but doesn't solve the problem.

![[appunti/assets/2026-09-22/transformers-015.png]]

**Residual block** (from computer vision, ResNets, used to train very deep FFNs): $x_{l+1} = x_l + \mathcal F(x_l)$, so $\dfrac{\partial x_{l+1}}{\partial x_l} = I + \dfrac{\partial \mathcal F}{\partial x_l}$. The shortcut is the **identity**, not a weight matrix. It always contributes an exact $I$ to the gradient, and nothing can shrink that term. It creates a **gradient highway through depth**. Highway networks are the gated version of the same idea.

![[appunti/assets/2026-09-22/transformers-016.png]]

**LSTM and GRU are the residual idea with learned gates:**

| model | carry coefficient on the old state | interpretation |
|---|---|---|
| pure residual RNN, $h_t = h_{t-1} + F(h_{t-1}, x_t)$ | **1** (fixed) | exact identity shortcut |
| LSTM, $c_t = f_t\odot c_{t-1} + i_t\odot\tilde c_t$ | $f_t$ | gated residual-like carry; $f_t \approx 1$ means nearly identity |
| GRU, $h_t = (1-z_t)\odot h_{t-1} + z_t\odot\text{func}$ | $1-z_t$ | gated residual-like carry; $z_t \approx 0$ means nearly identity |

So the answer to §8's question: **with the forget gate fixed at 1, the LSTM cell update *is* a residual connection through time.** Why not just use that? A residual path **always passes everything** from the past. You'd never be able to **forget**, and language modelling needs that: sometimes old context must be dropped. **Learnable** gates keep the gradient highway where it's useful and still let the network learn what to throw away. That's why they beat a plain residual RNN (see [25:30](https://au.cloud.panopto.eu/Panopto/Pages/Viewer.aspx?id=fa3c84d3-b9d6-47b6-b8bd-b4cd0076eb7a&start=1530)).

> **Takeaway (slide):** LSTM/GRU = residual idea + learned gates. Residual RNN = fixed carry 1. LSTM/GRU = gated carry + extra nonlinear paths.

---

## Da ricordare

- **RNN = FFN with dynamic depth** (depth = sequence length), with the **same weights at every step**. It's trained with **BPTT**, which is ordinary backprop on the unrolled graph. Backprop is cheap because shared upstream chain-rule factors are **cached**. ⚠️ **No hand-computed gradients in the exam.**
- **Stacked RNNs:** horizontal depth varies with the input, vertical depth is a fixed hyperparameter (typically **3–4** layers). **Bidirectional:** concatenate $[z^f_t; z^b_t]$, with **separate** parameters for each direction. It's **only useful when the future text is available** (classification, tagging). It's pointless for generation/LM.
- **Vanishing gradient:** $\dfrac{\partial h_t}{\partial h_{t-T}} = \prod \sigma'(u_i)\,W_{hh}$. Here $\sigma' \le 0.25$ and $W_{hh}$ is small (regularised), and it's the **same matrix multiplied $T$ times**. Small terms × many steps → exponentially small ($0.9^{100} \approx 2.7\times10^{-5}$). It's worst in **Elman RNNs because of the sigmoid**. Factors > 1 give **exploding** gradients instead → **gradient clipping** (rescale the gradient norm to a threshold; it fixes exploding only). **Truncated BPTT** backprops only within fixed windows: it bounds the cost and the chain length, but dependencies longer than the window can't be learned.
- **LSTM:** $f, i, o = \sigma(W_{\cdot x}x_t + W_{\cdot h}h_{t-1} + b_\cdot)$, $\tilde c_t = \tanh(\ldots)$, $c_t = i_t\tilde c_t + f_tc_{t-1}$, $h_t = o_t\tanh(c_t)$. **$c$ = long-term memory, $h$ = short-term memory.**
- **Why the LSTM fixes vanishing:** $\partial c_t/\partial c_{t-1} = f_t$ (the direct path). It **only vanishes if the forget gate closes**, whereas in a vanilla RNN it's squashed by construction. It's *mitigated*, not eliminated: still trained with BPTT, more parameters, still nonlinear. ⚠️ **At initialisation PyTorch's $f \approx \sigma(0) = 0.5$**, so the cell halves every step. **Forget-bias init** ($b_f \approx 3$, $f\approx0.95$) makes it remember by default (lab).
- **Gate edge cases:** $f=0$ erases the past. $f=i=0$ resets the cell to 0. $f=i=1$ gives $c_t = c_{t-1}+\tilde c_t$, which is **a residual connection through time**. The input gate goes to ≈0 on uninformative tokens ("the").
- **GRU:** update gate $z$ (forget and input in one, $1-z$ vs $z$) plus reset gate $r$, with a single hidden state. It's faster and sometimes better, but **theoretically less powerful (can't count)**. **Neither LSTM nor GRU dominates.** In the lab, the apparent gap came from the *default initialisation*, not the architecture. Fewer parameters ≠ faster, so measure it. The cost of both is **more parameters** (LSTM ≈ 4× a vanilla RNN, shared across time but **not across gates**).
- **Skip vs residual vs gates:** a weighted skip shortens the path but can still shrink the gradient. A residual adds an exact $I$ (gradient highway). **LSTM/GRU = residual + learned gates**, so they can also *forget*.
