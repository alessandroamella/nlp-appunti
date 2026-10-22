Every model so far generates text the same way: one token at a time, each fed back as input. This lecture looks at that process **independently of the architecture** ("you can also feed an n-gram language model, and the framework still remains"), in three parts:

1. **The framework:** what a single generation step computes, and how the model is trained (MLE / teacher forcing).
2. **Decoding:** the algorithm that turns a probability distribution into an actual token. **Greedy** methods (argmax, beam search) vs **sampling** (top-k, top-p, temperature).
3. **Training challenges:** why MLE training makes fancy decoders necessary (**lack of diversity**, **exposure bias**), and **reinforcement learning** as the fix.

> **Coverage.** The deck (`08 - NLG.pdf`) opens with a recap of last week (slides 5–19). Its last part, **PPO and DPO**, finishes the 6 Oct lecture and is written up there: [[2026-10-06 - From GPT to ChatGPT#7.6 PPO (Proximal Policy Optimization)]] and [[2026-10-06 - From GPT to ChatGPT#7.8 DPO (Direct Preference Optimization)]].

---

## 1. What is natural language generation?

> **Any task involving text production for human consumption requires natural language generation.**

Applications: **machine translation** (still hard for languages with little data, much improved since ~2019 thanks to LMs); **summarisation** (documents, emails, meetings); **data-to-text** (describe a table or spreadsheet; the input isn't language at all); **visual description** (image/video captioning); **creative generation** (stories from an outline, poems); and the **all-in-one** assistants (trip plans, code, syllabi) and **agents** that plan and act.

The lecturer's point: NLG used to be one corner of NLP (MT, summarisation). Since every task can be posed as text generation (the T5 idea, [[2026-09-29 - Pre-trained Language Models#T5: Text-to-Text Transfer Transformer]]), **NLG has become the whole of NLP**. Classification and regression too: the model "is still just generating the next token".

All the models in this course are **autoregressive**. They can be *trained* in parallel (Transformers), but they *generate* one token at a time. **Diffusion models** are non-autoregressive generators, used for language a bit too, but they are **not part of this course** (see [32:58](https://au.cloud.panopto.eu/Panopto/Pages/Viewer.aspx?id=3afaa41e-2164-4a9a-ad0a-b4e90076fde1&start=1978)).

---

## 2. The framework

### A single step

At each time step $t$, the model takes the tokens so far $\{y\}_{<t}$ and:

1. computes a **vector of scores**, one per vocabulary token: $S = f(\{y_{<t}\};\ \theta) \in \mathbb R^{|V|}$, where $f$ is your model;
2. turns them into a **probability distribution** with a softmax:
   $$P(y_t = w \mid \{y_{<t}\}) = \frac{\exp(S_w)}{\sum_{w' \in V}\exp(S_{w'})}$$
3. **selects a token** from that distribution with a **decoding algorithm** $g$:
   $$\hat y_t = g\big(P(y_t \mid \{y_{<t}\})\big)$$

![[appunti/assets/2026-10-20/nlg-046.png]]

The model outputs **a distribution, not a token** ("He wanted to go to the" → restroom, grocery, store, airport…). To keep the autoregressive loop going, something has to pick one token, and that something is $g$. Then $\hat y_t$ is fed back as input at $t+1$, until `<END>` is produced or a length limit is reached.

### Training: maximum likelihood (teacher forcing)

$$\mathcal L = -\sum_{t=1}^{T} \log P\big(y^*_t \mid \{y^*\}_{<t}\big)$$

![[appunti/assets/2026-10-20/nlg-053.png]]

- Each step is a **multi-class classification task**: each $w \in V$ is a class, and the label is the actual next word $y^*_t$ of the training sequence, called the **gold** or **ground-truth** token.
- The input at every step is the **gold** prefix $y^*_0, \dots, y^*_{t-1}$, not the model's own predictions. That's **teacher forcing**, the same thing seen for RNNs, Transformers and GPT ([[2026-09-22 - Seq2Seq Models and Transformers#Training]], [[2026-09-29 - Pre-trained Language Models#Pretraining]]). Pretraining is literally this objective.

**Wrap-up of the framework:**

- Text generation is the foundation of many NLP applications (MT, summarisation, dialogue).
- Autoregressive NLG produces one token at a time, using the context and the previously generated tokens.
- The model produces a score for every vocabulary token, turned into a distribution by the softmax.
- **MLE training is what makes that distribution calibrated.** The decoder doesn't have to worry about it. Its job is only to pick a token.

---

## 3. Greedy decoding: argmax and beam search

### Argmax

$$\hat y_t = \arg\max_{w\in V} P(y_t = w \mid \{y\}_{<t})$$

**Select the highest-scoring token** ("restroom").

**Menti: what's an inherent problem with argmax decoding?** A student said "creativity might be limited", which is fair. The lecturer's main answer is the problem of every greedy algorithm: **it's only locally optimal.** Each step maximises the probability *at that step*, without seeing what happens 10 tokens later. The best choice at each step need not give the best sequence (see [46:19](https://au.cloud.panopto.eu/Panopto/Pages/Viewer.aspx?id=3afaa41e-2164-4a9a-ad0a-b4e90076fde1&start=2779)).

### Beam search

> **In greedy decoding, we cannot revise prior decisions.**

"les pauvres sont démunis" (the poor don't have any money): commit to "the" → "the poor" → "the poor are"… If one choice was bad, you're stuck with it.

**Beam search: explore several different hypotheses instead of just one.** Keep track of the **$b$ most probable sequences** at each decoder step instead of one. $b$ is the **beam size**, and $b = 1$ is greedy decoding.

![[appunti/assets/2026-10-20/nlg-067.png]]

Example with $b = 2$. The numbers are the **log-probability of the whole sequence so far** (sum of the token log-probs), not of the last token:

1. From `<START>`: *the* (−1.05) and *a* (−1.39). **Which is more likely? *the*.** These are log-probabilities (all negative), so **closer to 0 = more likely**. Many in class got this backwards.
2. Extend each beam by its best tokens: *the poor* (−1.90), *the people* (−2.3), *a poor* (−1.54), *a person* (−3.2). Keep the best 2 **across all beams**: *a poor*, *the poor*.
3. Extend again: *the poor are* (−2.42), *the poor don't* (−2.13), *a poor person* (−3.12), *a poor but* (−3.53). Keep *the poor don't*, *the poor are*. The *a* branch is **pruned**.
4. Continue until `<END>` or a length limit, and return the best finished sequence: *the poor don't have any money*.

A bigger beam explores more but costs more compute. It's still a greedy method, just less myopic. (Ties are an edge case: "it's still greedy… generally it will work".)

### What's the problem with greedy decoding, even with beams?

> **Works well for MT or summarisation, not for tasks that don't have a strong correspondence between input and output.**

MT is close to one-to-one: syntax and length change, but the content is fixed by the input. With a good model and a big enough beam, greedy methods do fine there. They break down for **open-ended** generation: creative writing, dialogue, continuing a story, tasks with **many valid answers** and **no strong input–output correspondence**. In class, students suggested different lengths, multiple valid answers and "generating out of thin air". The lecturer argued that a large enough beam handles the first two, so the real issue is the next point (see [1:12:11](https://au.cloud.panopto.eu/Panopto/Pages/Viewer.aspx?id=3afaa41e-2164-4a9a-ad0a-b4e90076fde1&start=4331)).

**Why?** Two reasons came up: the process is still essentially **deterministic**, and above all:

> **They maximise the likelihood of the sequence.** Humans don't speak by always picking the most likely next word. If they did, predicting what someone will say would be easy. There's surprise and serendipity in human language.

---

## 4. Why greedy text degenerates

**What do maximum-likelihood sequences look like?**

- Early dialogue systems answered **"I don't know."** all the time. It's a very common, safe continuation in the training data. (The irony noted in class: today the complaint is the opposite, that models *don't* say "I don't know" and hallucinate.)
- Feed the model **"I don't know. I don't know. I don't know…"** and look at the NLL of each step: **the negative log-likelihood decreases over time.** The more the model has repeated something, the more confident it gets that the repetition continues.

![[appunti/assets/2026-10-20/nlg-073.png]]

- **Worse for Transformer LMs.** On "dogs dogs dogs…", the NLL of the LSTM **saturates** after a few steps, but the Transformer's (OpenAI GPT) **keeps dropping** (see [1:17:33](https://au.cloud.panopto.eu/Panopto/Pages/Viewer.aspx?id=3afaa41e-2164-4a9a-ad0a-b4e90076fde1&start=4653)). My reading of why: attention can copy from any earlier position directly, so each repetition gives one more "matching" past token to copy from.

![[appunti/assets/2026-10-20/nlg-074.png]]

- **And it keeps going: the longer it goes, the worse it gets.** This is a self-reinforcing loop: repeat → more confident → repeat. It's the "Universidad Nacional Autónoma de México" continuation from last week.

> **Repetition is a big problem in text generation.**

**Whose fault is it?** Not MLE alone: "maximum likelihood is still the best way of training these models". The problem appears when you **pair MLE training with a greedy-style decoder, which exploits this vulnerability** (see [1:19:13](https://au.cloud.panopto.eu/Panopto/Pages/Viewer.aspx?id=3afaa41e-2164-4a9a-ad0a-b4e90076fde1&start=4753)). Better decoding is the first fix (§5). Training beyond MLE (rewards and RL, §8; the RLHF steps of [[2026-10-06 - From GPT to ChatGPT#7. From GPT-3 to ChatGPT: instruction tuning and RLHF]]) is the other.

### Are greedy methods reasonable?

![[appunti/assets/2026-10-20/nlg-077.png]]

The probability the model assigns to each token of a 100-token text: **beam search** stays near the top, almost always ~1. **Human** text jumps all over the place, from very unlikely to very likely tokens. Human language has **higher entropy**: it's less predictable. Greedy decoding produces a *distribution* of token probabilities that looks nothing like human writing. So no, greedy methods aren't reasonable for open-ended text.

(Student objection: humans also write predictable stop words. True, but the model also gives stop words high probability when appropriate, and over 100 tokens, not all of them are stop words. "Humans are not as predictable as things should be, and that's what makes us humans.")

---

## 5. Sampling

### Pure sampling: "time to get random"

Instead of taking the top token, **sample** from the distribution:

$$\hat y_t \sim P(y_t = w \mid \{y\}_{<t})$$

It's random, so **any** token can come out. But it's **not uniform**: likely tokens are still picked more often. It respects the model's probabilities while introducing human-like unpredictability.

**What's the problem with sampling?** You can get **nonsense**, lose **factuality**, or derail into something incoherent. The structural reason is the **long tail**:

- With a vocabulary of 50,000 tokens, say 50 tokens are good and hold 80% of the mass. The other **49,950 are individually very unlikely, but together they still hold ~20%** (see [1:27:40](https://au.cloud.panopto.eu/Panopto/Pages/Viewer.aspx?id=3afaa41e-2164-4a9a-ad0a-b4e90076fde1&start=5260)).
- So at every step there's a real chance of sampling from the tail. Over many steps it *will* happen, and one bad token derails everything after it. It's the same long-tail shape as word frequencies in the BoW lecture ([[2026-08-25 - Introduction to NLP]]).

> **Why are we giving them individually a tiny chance, and as a group a high chance, to be selected?**

### Top-k sampling

**Only sample from the top $k$ tokens** of the distribution: set the rest to zero, **renormalise**, sample. Common values are $k = 5, 10, 20$ ("but it's up to you"); GPT-2 used 40.

- **Increase $k$** → more **diverse/risky** outputs.
- **Decrease $k$** → more **generic/safe** outputs. As $k \to 1$ it becomes greedy again, with all its repetition.

**What's the problem with top-k?** One fixed $k$ for every step, but **the distributions are dynamic**:

![[appunti/assets/2026-10-20/nlg-086.png]]

- **Flat** distribution ("She said, 'I never ___'": thought, knew, had, saw, did, said, wanted, told, liked… all plausible): a small $k$ **cuts off too quickly** and removes viable options.
- **Peaky** distribution ("I ate the pizza while it was still ___": *hot*, then a tail): a large $k$ **cuts off too slowly** and lets tail tokens in. That's the same issue as vanilla sampling.

### Top-p (nucleus) sampling

A student proposed exactly this in class: make the cutoff **distribution-aware**.

> **Sample from all tokens in the top-$p$ cumulative probability mass** (where the mass is concentrated). **$k$ varies with the uniformity of $P_t$.**

![[appunti/assets/2026-10-20/nlg-089.png]]

With $p$ fixed (say 0.8): a **flat** distribution needs **many** tokens to reach 80%, so it acts like a large $k$. A **peaky** one reaches it with **2**, so it acts like a small $k$. One hyperparameter now adapts to every step. **Top-p is the sampling method used in most LLMs today.**

### Temperature

All of the above trusts the distribution as it comes. The **softmax temperature** $\tau$ reshapes it first:

$$P_t(y_t = w) = \frac{\exp(S_w/\tau)}{\sum_{w'\in V}\exp(S_{w'}/\tau)}$$

![[appunti/assets/2026-10-20/nlg-092.png]]

Dividing by $\tau > 1$ **shrinks the differences** between scores. With scores 10 and 5, the gap is 5. Divide by 2 and you get 5 and 2.5, a gap of 2.5. The softmax only sees differences, so the distribution flattens.

| | effect on $P_t$ | output |
|---|---|---|
| **raise** $\tau > 1$ | more **uniform** | **more diverse**: probability spread over the vocabulary |
| **lower** $\tau < 1$ | more **spiky** | **less diverse**: probability concentrated on top words |
| $\tau \to 0$ | all mass on the highest-probability token | **greedy decoding**, whatever sampling method you use |
| $\tau \to \infty$ | **uniform** | random tokens |

Temperature is a **hyperparameter applied at decoding time**. It changes no weights. It can be combined with any sampling method (top-p 0.9–0.95 with $\tau \approx$ 0.6–1.0 is the common choice for open-ended text). The two extremes are both useless.

---

## 6. Improving decoding further

### Re-balancing the distribution

**Problem: what if I don't trust how well my model's distribution is calibrated?** Temperature reshapes it blindly. Instead, **don't rely only on the model's distribution**.

![[appunti/assets/2026-10-20/nlg-097.png]]

**Re-balance $P_t$ using retrieval from n-gram phrase statistics** (kNN-LM, Khandelwal et al. 2020):

1. Cache a database of phrases from your training corpus (or another corpus), with the word that follows each.
2. At decoding time, search for the most similar phrases. For "Obama's birthplace is ___": "Obama was born in → Hawaii", "Obama is a native of → Hawaii", "Obama was senator for → Illinois".
3. Turn their next words into a distribution $P_{phrase}$ (Hawaii 0.8, Illinois 0.2) and **interpolate** it with the model's own (Hawaii 0.2, Illinois 0.2…).

You don't need the math. The idea is **adding external information at test time** to recalibrate. The lecturer tied it to **retrieval-augmented generation (RAG)**, next week's topic: don't rely only on what's encoded in the parameters, retrieve extra context (see [1:39:41](https://au.cloud.panopto.eu/Panopto/Pages/Viewer.aspx?id=3afaa41e-2164-4a9a-ad0a-b4e90076fde1&start=5981)). A use case: a domain the model hasn't seen much (e.g. molecular biology), where you inform the distribution with domain-specific examples.

### Re-ranking

**Problem: what if I decode a bad sequence?** **Decode a bunch** (~10 candidates is common, but up to you), **define a score that approximates quality, and re-rank by it.**

- The simplest score is **perplexity**. ⚠️ **Careful: repetitive sequences can get low perplexity.** That's §4 again.
- Re-rankers can score many properties: **style**, **discourse**, **entailment/factuality**, **logical consistency**… ⚠️ **Beware of poorly calibrated re-rankers.**
- You can use **several re-rankers in parallel** and combine them.

### Wrap-up: decoding

- **Decoding is still a challenging problem** in NLG.
- **Human language is not well approximated by probability maximisation.**
- Different decoding algorithms inject biases that encourage different properties of coherent, aligned generation.
- Some of the **most impactful advances** in NLG of the last years came from **simple but effective changes to decoding**. Top-k (Fan et al., 2018) and top-p (Holtzman et al., 2020) are both recent. (The lecturer said 2019 for top-k; the paper is ACL 2018.) There's a lot more work to be done.

> [!example] Lab — decoding strategies on GPT-2 (`08 - Generation/`; solutions not released yet, so the MCQ answers below are mine)
> - **Two metrics** to compare strategies: the average **log-probability per token** (how likely the text is to the model; $e^{-\text{avg}}$ is the perplexity) and **distinct-n** (unique n-grams / total n-grams; 1 = no repetition, near 0 = loops). The trade-off: greedy and beam search are high-likelihood and low-diversity; pure sampling is the reverse; **human text sits in between**, which is where top-p with a moderate temperature lands.
> - **Greedy misses likely sequences:** "The nice woman" (0.5 × 0.4 = 0.20) vs "The dog has" (0.4 × 0.9 = 0.36). Greedy takes *nice* (0.5 > 0.4) and never sees *has*. Beam search with 2 beams finds it. **Beam search with `num_beams=1` = greedy.**
> - **Beam scores** are usually **length-normalised** (sum of log-probs ÷ length, `length_penalty`), because each extra token adds a negative log-prob, so raw sums favour short outputs. **Beam search is not exhaustive:** finding the true most likely sequence of 20 tokens would require scoring $|V|^{20}$ sequences. A wider beam usually, but not always, finds a more likely text.
> - **`no_repeat_ngram_size`** removes loops by forbidding any repeated n-gram. It breaks text that *should* repeat: names ("New York" twice), refrains, code, legal text.
> - **Min-p sampling** (newer): keep tokens with $p \ge$ `min_p` × $p_{max}$. With top token 0.4 and `min_p` 0.1, the threshold is 0.04. Like top-p it adapts to the shape, but relative to the top token.
> - **KV cache:** in a causal model, earlier tokens' keys and values never change (a token attends only to the past), so store them and feed only the new token at each step. Without the cache, generating $n$ tokens costs ~$O(n^2)$ token passes; with it, ~$O(n)$. The price is memory, which grows with layers × context length. (This is the "cache the keys/values of earlier positions" from [[2026-09-22 - Seq2Seq Models and Transformers#Training]].)
> - **Logits processors** (repetition penalty, banned tokens, the "no letter *e*" lipogram) act on the scores *before* the choice at every step. Setting a score to $-\infty$ makes the token **impossible** under any strategy. That's more reliable than asking in the prompt, and it's how constrained decoding (valid JSON, grammar-constrained output) works.
> - Three of the lab MCQs, with my answers:
>   - *(Decoding from Scratch, 6.3) The next-token probabilities are 0.6, 0.25, 0.1, 0.05. Which tokens does top-p sampling with $p = 0.8$ keep?* A. Only the first · **B. The first two** ✅ · C. The first three · D. All four. Add tokens from the top until the mass reaches $p$: 0.6 < 0.8, then 0.85 ≥ 0.8, stop.
>   - *(Decoding from Scratch, 6.2) Beam search with `num_beams=4` generates 10 tokens. How many candidate sequences are scored at each step (after the first), with a vocabulary of size $|V|$?* A. 4 · **B. $4\cdot|V|$** ✅ · C. $|V|^4$ · D. $4^{10}$. Each of the 4 beams is extended by every token; then the best 4 are kept. 4 is what *survives*, not what's scored.
>   - *(Decoding Strategies, 8.5) For which task is beam search the most suitable decoding strategy?* A. Writing creative short stories · B. Brainstorming many different product names · **C. Translating sentences from English to French** ✅ · D. Generating diverse dialogue for characters in a game. Translation has (roughly) one right output, so you want the most likely one; the other three want diversity, i.e. sampling.

---

## 7. Training challenges: why do we need fancy decoders?

> **Shouldn't our models just be better calibrated?**

The models *are* trained well and the probabilities *are* calibrated. The issues come from **how** they're trained: **teacher forcing**.

### Issue #1: MLE discourages diversity

MLE pushes up the probability of continuations that are **frequent** in the training data, for a given context. It's like class imbalance: the model optimises for the dominant part of the data. But at test time, different users may want different continuations. The lecturer's example: models trained mostly on English drift back to their dominant language and its patterns. Combined with greedy decoding, you get the repetition of §4. **MLE-trained models are biased towards generic, frequent continuations.**

**Menti: how can we increase the diversity of the sequences seen during training?**

- **More data! Pretraining!** More *diverse* data, or **rebalancing**. You can't add more of the rarer languages (the internet is used up), but you can **down-weight** dominant ones like English in the mix (see [2:09:55](https://au.cloud.panopto.eu/Panopto/Pages/Viewer.aspx?id=3afaa41e-2164-4a9a-ad0a-b4e90076fde1&start=7795)).
- **Learn from the sequences that the model generates itself!** With top-p sampling a model can produce diverse text, which can be used as **synthetic data** / data augmentation (paraphrases too). This matters because "the whole internet is gone": real data has run out. (Distillation is a different thing: matching a big model's distribution with a smaller one, [[2026-09-29 - Pre-trained Language Models#Variants]].)

### Issue #2: exposure bias

> **Training with teacher forcing leads to exposure bias at generation time.**

- During **training**, the model's inputs are **gold** context tokens from real, human-generated text: $\mathcal L_{MLE} = -\log P(y^*_t \mid \{y^*\}_{<t})$.
- At **generation** time, its inputs are its **previously decoded** tokens: $\mathcal L_{dec} = -\log P(\hat y_t \mid \{\hat y\}_{<t})$.

The model was never trained to continue *its own* text. As soon as its outputs differ from human text (and they do: compare the beam vs human probability plot in §4), it is conditioning on inputs from a distribution it never saw, and it can **slowly deviate** further (see [2:13:39](https://au.cloud.panopto.eu/Panopto/Pages/Viewer.aspx?id=3afaa41e-2164-4a9a-ad0a-b4e90076fde1&start=8019)). It's worst with greedy methods but happens with any decoder. It's the same "shortcoming of only using SFT" as last week ([[2026-10-06 - From GPT to ChatGPT#7.2 What's wrong with only SFT?]]): SFT is teacher forcing on demonstrations.

> In class, "learn from the model's own sequences" and "this leads to exposure bias" came out tangled together. The slides' wrap-up (below) is the clean version. **Teacher forcing causes exposure bias; the remedy is to make models learn to recover from their own (bad) samples**, e.g. scheduled sampling (Bengio et al., 2015) or RL. Training on self-generated text is part of the *fix*.

---

## 8. Text generation as reinforcement learning

**Solution: cast your text generation model as a Markov decision process.** In RL an agent in a **state** takes **actions** following a **policy**, gets **rewards**, and learns the policy that maximises reward. The lecturer's analogy is a video game (Mario): levels are states, moves are actions, and you learn to take the moves that get you to the end.

| RL | text generation |
|---|---|
| **state** $s$ | the model's representation of the preceding context |
| **actions** $a$ | the words (tokens) that can be generated: the whole vocabulary |
| **policy** $\pi$ | the decoder (the model + decoding), which picks the next token |
| **rewards** $r$ | provided by an **external score** |

**Learn behaviours by rewarding the model when it exhibits them.**

### REINFORCE (recap)

$$\mathcal L_{RL} = -R(\hat Y)\sum_{t=1}^{T}\log P\big(\hat y_t \mid \{y^*\};\ \{\hat y\}_{<t}\big)$$

**Sample a sequence from your model** given the gold context $\{y^*\}$ (the prompt/instruction). Next time, **increase the probability of the sampled tokens in the same context**, and **do it more if the reward is high**. The full treatment, with baseline and coherence term, is in [[2026-10-06 - From GPT to ChatGPT#7.5 REINFORCE]].

**Credit assignment:** ideally you'd reward each *token*, but feedback comes for whole sequences (a thumbs-up rates the whole answer), and deciding which token deserved it is the hard **credit assignment** problem of RL. So in practice **an aggregate reward is used for the sequence: every token gets the same reward** (see [2:27:36](https://au.cloud.panopto.eu/Panopto/Pages/Viewer.aspx?id=3afaa41e-2164-4a9a-ad0a-b4e90076fde1&start=8856)).

### Reward estimation

**How should we define a reward? Just use your evaluation metric!**

| metric | task |
|---|---|
| **BLEU** | machine translation |
| **ROUGE** | summarisation |
| CIDEr, SPIDEr | image captioning |

Here you optimise the metric *directly*, on top of MLE (which still provides coherence) and whatever decoding you use. For example: "for MT, instead of only minimising perplexity, give me higher BLEU".

> ⚠️ **Be careful about optimising for the task as opposed to "gaming" the reward!** Evaluation metrics are merely **proxies** for generation quality. «Even though RL refinement can achieve better BLEU scores, it barely improves the human impression of the translation quality» (Wu et al., 2016, Google NMT).

That's why today rewards come from **human preference data** ([[2026-10-06 - From GPT to ChatGPT#7.3 Step 2: the reward model]]). It's also noisy (a linguist's or native speaker's judgement of a translation is worth more than a non-speaker's), but it targets what we actually care about. Rewards can capture much more than correctness: for proofs, a mathematician can say which of 5 proofs is right *and* which they prefer, and why.

**What behaviours can we tie to rewards?** Cross-modality consistency (captioning), sentence simplicity, temporal consistency, utterance politeness, paraphrasing, sentiment, formality… If you can formalise a behaviour as a reward function, or **train a neural network to approximate it**, you can train a model to exhibit it.

**Isn't training the reward model harder than training the LM?** The lecturer said usually not. The LM has to encode the whole internet. The bottleneck for reward models used to be *collecting* preferences (crowdsourcing, expert annotation), and today billions of users provide them. An aside from the lecturer: as soon as you give feedback (thumbs up/down) to a chat model, that conversation goes into the training data, even if you opted out of training on your data (see [2:37:30](https://au.cloud.panopto.eu/Panopto/Pages/Viewer.aspx?id=3afaa41e-2164-4a9a-ad0a-b4e90076fde1&start=9450)).

### Implementation tricks (recap) and reward hacking

- **Start from the SFT model.**
- **Baseline** $b$: otherwise, with rewards always $> 0$ (metrics and RMs in $[0,1]$), you **never penalise** a generation, even utter garbage. The lecturer compared it to a regularisation term.
- **Keep the MLE term:** $\mathcal L = \mathcal L_{MLE} + \alpha\,\mathcal L_{RL}$. RL alone can pull the model away from fluent language.

> **Can I use a "politeness classifier" as a reward function? Yes!** Reward sequences that the model classifies as polite… and it **learns to add "please" 20× to the end of any utterance.**

The flip side works too: an exclamation-mark-based "rudeness" reward gets you 20 "!". Simple rewards are easy to define and easy to **hack**. This is the case for learning rewards from preference data.

### Wrap-up: training

- **Teacher forcing is still the premier algorithm** for training text generation models.
- **Diversity** is an issue with sequences generated from teacher-forced models.
- **Exposure bias** makes text generation models **lose coherence** easily. Models must learn to **recover from their own bad samples** (scheduled sampling, RL).
- Training with **RL** lets models learn behaviours that are hard to formalise, but **learning can be very unstable**. That's the motivation for PPO's clipping and DPO's simplicity: [[2026-10-06 - From GPT to ChatGPT#7.6 PPO (Proximal Policy Optimization)]].

---

## Exam info given in this lecture

- **Midterm: Fri 6 Nov 2026, 9:00–12:00.** It **starts at 9:00, not 9:15**. Two rooms: **5344-101 "Benjamin"** (the big hall behind the auditorium, also used for the hackathon) and **5123-122 "Seminarrum"** (the exercise room). Room assignments and final instructions via Brightspace by Fri 30 Oct. Two staff per room (see [3:16](https://au.cloud.panopto.eu/Panopto/Pages/Viewer.aspx?id=3afaa41e-2164-4a9a-ad0a-b4e90076fde1&start=196)).
- The exam is already written and is being **dry-run by a TA** for timing and clarity. If a question turns out to be really confusing, you get **a week to raise issues** afterwards, and a question with a major issue **can be discounted completely** (see [5:08](https://au.cloud.panopto.eu/Panopto/Pages/Viewer.aspx?id=3afaa41e-2164-4a9a-ad0a-b4e90076fde1&start=308)).
- Not asked: the proof that RLHF and DPO optimise the same objective; diffusion models (not part of the course).

---

## Practice questions

> Not from the course: written in the style of the 2025 paper to test this lecture, which the past paper doesn't cover. One correct option each. Answers are folded.

**P1. Which statement about softmax temperature $\tau$ is correct?**

A. $\tau > 1$ makes the distribution more peaked, so outputs are less diverse.

B. As $\tau \to 0$, sampling approaches greedy decoding.

C. Temperature is a hyperparameter of fine-tuning that rescales the model's weights.

D. Temperature can be combined with top-k but not with top-p.

> [!success]- Answer
> **B.** A is backwards: $\tau>1$ shrinks score differences → **flatter**, more diverse. C: it's applied **at decoding time** and changes no weights. D: it reshapes the distribution *before* any sampling method, so it combines with all of them.

**P2. A model's next-token distribution, sorted, is $(0.5, 0.2, 0.15, 0.1, 0.05)$. With top-p sampling at $p = 0.8$, how many tokens are kept, and what is the top token's probability after renormalising?**

A. 2 tokens; ≈ 0.71

B. 3 tokens; ≈ 0.59

C. 3 tokens; 0.5

D. 4 tokens; ≈ 0.53

> [!success]- Answer
> **B.** Cumulative mass: 0.5, 0.7 (< 0.8), **0.85** (≥ 0.8) → the smallest set reaching $p$ is 3 tokens. Renormalise: $0.5/0.85 \approx 0.59$. A stops before reaching $p$. C forgets to renormalise. D keeps one token too many.

**P3. After `<START>`, a model gives $P(\text{a}) = 0.6$, $P(\text{the}) = 0.4$. The best continuation of "a" is "cat" with $P = 0.3$; the best continuation of "the" is "dog" with $P = 0.8$. Which two-token output do greedy decoding and beam search with $b = 2$ return?**

A. Both return "a cat".

B. Greedy returns "a cat" (0.18); beam search returns "the dog" (0.32).

C. Greedy returns "the dog"; beam search returns "a cat".

D. Both return "the dog".

> [!success]- Answer
> **B.** Greedy commits to "a" (0.6 > 0.4) and can't revise it: $0.6 \times 0.3 = 0.18$. Beam search keeps both prefixes and finds $0.4 \times 0.8 = 0.32$. Beam search still isn't guaranteed to find the most likely sequence in general (it's not exhaustive).

**P4. Why does beam search work well for machine translation but poorly for open-ended story generation?**

A. Beam search is too slow for long outputs.

B. It maximises sequence likelihood, but human open-ended text isn't maximum-likelihood text, so it produces generic and repetitive output.

C. Translation models are trained without teacher forcing.

D. Beam search can't produce the `<END>` token.

> [!success]- Answer
> **B.** MT has a strong input–output correspondence, so the most likely output is a good one. Open-ended text has many valid answers and human token probabilities are low and variable.

**P5. Which statement about repetition in generated text is correct?**

A. The NLL of a repeated phrase increases with each repetition, so models naturally escape loops.

B. Re-ranking candidates by perplexity reliably removes repetitive outputs.

C. The NLL of a repeated phrase decreases with each repetition, and greedy-style decoding exploits this; the effect is stronger for Transformers than for LSTMs.

D. It's caused by MLE training alone and appears with any decoding algorithm.

> [!success]- Answer
> **C.** A is backwards. B is a trap: repetitive text gets **low** perplexity, so it would be ranked *higher*. D: the lecturer blamed **MLE + greedy decoding** together; sampling methods mitigate it.

**P6. Why was top-p (nucleus) sampling proposed over top-k?**

A. Top-k can't be combined with temperature.

B. A fixed $k$ is too small for flat distributions and too large for peaky ones; top-p adapts the number of candidates to the distribution.

C. Top-k is deterministic, while top-p is random.

D. Top-p removes exposure bias.

> [!success]- Answer
> **B.** C: both are sampling methods. D: exposure bias is a training problem; no decoder fixes it.

**P7. Consider these statements about text generation as reinforcement learning:**

1. The policy is the decoder, and the actions are the vocabulary tokens.
2. Using BLEU as the reward reliably improves human-judged translation quality.
3. With REINFORCE, every token of a sampled sequence typically gets the same sequence-level reward.

**Which are TRUE?** A. 1 · B. 1, 3 · C. 2, 3 · D. 1, 2, 3

> [!success]- Answer
> **B.** Statement 2 is false: metrics are proxies and get **gamed** (Google NMT: better BLEU, barely better to humans). That's why rewards now come from human preferences.

---

## Da ricordare

- **One step:** scores $S = f(y_{<t};\theta)$ → softmax → $P$ → **decoding algorithm $g$** picks $\hat y_t$, which is fed back. **Training = MLE / teacher forcing** (gold prefixes, multi-class classification over $V$). **Decoding changes no weights.**
- **Argmax** = locally optimal, can't revise decisions. **Beam search** keeps the $b$ best **sequences** (sum of log-probs, **closer to 0 = more likely**), with $b=1$ being greedy. It's not exhaustive. It's good for **MT/summarisation** (strong input–output correspondence), bad for **open-ended** generation, because it **maximises likelihood and human text isn't max-likelihood** (human token probabilities are low and variable).
- **Repetition:** the NLL of a repeated phrase **decreases** with each repetition (**worse for Transformers** than LSTMs, which saturate). It's a self-reinforcing loop. Caused by **MLE + greedy decoding** together.
- **Sampling** respects $P$ but the **long tail** (thousands of individually unlikely tokens, a large total mass) derails the text. **Top-k:** fixed $k$, cuts too quickly on **flat** distributions and too slowly on **peaky** ones. **Top-p (nucleus):** smallest set with cumulative mass ≥ $p$, so **$k$ adapts** to the distribution. It's the standard in LLMs.
- **Temperature** $\exp(S_w/\tau)$: $\tau>1$ **flatter, more diverse**; $\tau<1$ **spikier, less diverse**; $\tau\to0$ = **greedy**; $\tau\to\infty$ = uniform. It's applied at decoding time.
- **Re-balancing** (kNN-LM: interpolate with the next-word distribution of retrieved similar phrases, ≈ RAG) and **re-ranking** (decode ~10 candidates, score them; ⚠️ perplexity favours repetitive text).
- **Issue #1:** MLE **discourages diversity** (favours frequent continuations) → more/rebalanced data, self-generated data. **Issue #2: exposure bias** = trained on **gold** prefixes, generates from **its own** → drift and loss of coherence → learn to recover from own samples (scheduled sampling, **RL**).
- **NLG as RL:** state = context representation, action = next token, **policy = decoder**, reward = external score. One reward per sequence (credit assignment is hard). **Metric rewards (BLEU, ROUGE) get gamed** (better BLEU, no better for humans; "please" ×20) → use **human preferences**. RL training is **unstable** → PPO/DPO.
