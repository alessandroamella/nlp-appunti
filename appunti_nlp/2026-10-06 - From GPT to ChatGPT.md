[[2026-09-29 - Pre-trained Language Models]] ended in 2019–20: pretrain a Transformer, then **fine-tune** it for each task. This lecture covers what changed when models got much bigger, and the recipe that turned a big pretrained model (GPT-3) into an assistant (ChatGPT):

1. **Scaling** LMs into LLMs, and the abilities that "emerge" with scale.
2. **In-context learning and prompting**: solving tasks with no gradient updates at all, just by writing the right input.
3. **Instruction fine-tuning and RLHF**: SFT → reward model → reinforcement learning (REINFORCE, PPO), and DPO as the shortcut.

> **Coverage.** The lecture was given over Zoom; the lecturer was sick and stopped ~15 minutes early, right before **PPO**. PPO and **DPO** were covered at the end of the 20 Oct lecture. They're written up here (§7.6–7.8) with links to that recording. The deck (`07 - LLMs.pdf`) opens with a long recap of the PLM lecture (slides 4–26), not repeated here. In the 6 Oct recording, the explanation of the TriviaQA slides fell into the gap around the break.
>
> **Links:** this recording is a Brightspace `.mp4`, not Panopto, so the timestamp links use `#t=<seconds>`. They open the video file at that point (in most browsers).

Two recap Menti questions are worth remembering, because they're exam-shaped:

- **Main improvement of BERT-style models over GPT?** **Bidirectionality**: one model that sees both sides learns more expressive representations.
- **What can BERT *not* do because of its MLM objective?** **Generate text.** It can still classify and tag. It was never trained left to right (no causal mask), so it can't produce text one token at a time.

---

## 1. Scaling LMs into LLMs

![[assets/2026-10-06/llms-034.png]]

GPT → GPT-2 → GPT-3 → GPT-4: 117M → 1.5B (~10×) → 175B (~100× again) → 1.8T (rumoured). Model size, data and compute all grew together:

- **Larger models**: from ELMo (94M) and BERT-Large (340M) in 2018 to Megatron-Turing NLG (530B) in 2022, on a straight line in log scale.
- **More data** and longer training. RoBERTa beat BERT with the same architecture *by training longer on more data*. GPT-3 was trained far longer than anything before it.
- **More compute**, and therefore **more money**. Note the **log scale** on the y-axes.

> **Training costs were doubling every nine months.**

The lecturer said this plot "gets old every 2–3 weeks". Today it's more like every 3 months, or even every month.

---

## 2. "Emergence"

> «Emergence is when quantitative changes in a system result in qualitative changes in behavior.» — Philip Anderson, 1972
>
> For LLMs: «an ability is emergent if it is **not present in smaller models** but **is present in larger models**.»

The **quantitative** change is more parameters, data and compute. The **qualitative** change is new behaviours: first **zero-shot** tasks, then **in-context learning**. ⚠️ **It's a controversial term.** Many researchers dislike it because it sounds unscientific, as if something mysterious appears. The observations themselves are real.

---

## 3. Three ways to use a pretrained model

| paradigm | what you give the model | weights updated? | era |
|---|---|---|---|
| **fine-tuning** | many labelled examples, one gradient update each | **yes**, all of them | GPT, BERT |
| **zero-shot** | a **task description** ("Translate English to French:") + the **prompt** ("cheese =>") | no | GPT-2 |
| **in-context learning (few-shot)** | task description + **a few examples** (*demonstrations*: "sea otter => loutre de mer", …) + prompt, **all in the same input sequence** | **no** | GPT-3 |

![[assets/2026-10-06/llms-040.png]]

> **There is no more learning in the classical sense.** The model "learns" by **attending to the tokens of the in-context examples**: no gradients, no backpropagation, no parameter updates.

Remove the examples from the prompt and the "learning" is gone. GPT had to be fine-tuned for each task. GPT-2 could do some tasks zero-shot. GPT-3 could learn from examples in its context. The architecture was the same each time and only the scale changed, which is why this counts as emergence.

> [!example] Lab — the "old paradigm" measured (`07 - Finetuning and Prompting/0. Finetuning.ipynb`)
> DistilBERT on IMDB sentiment, 2,000 training reviews:
> - **Before training:** ~50%. The classification head on `[CLS]` is **new and randomly initialised**. The pretrained knowledge is in the `[CLS]` vector, but nothing tells the head how to read it. Per-class accuracies are lopsided (all reviews have similar `[CLS]` vectors, so a random head predicts the same class for nearly all).
> - **Full fine-tuning** (all 67M parameters, **learning rate 2e-5**, 50× smaller than for training from scratch): ~85%. The small learning rate (and warm-up) avoids **catastrophic forgetting**, i.e. overwriting the pretrained weights with the large, random early gradients from the new head.
> - **Frozen features + logistic regression** on `[CLS]`: ~80%. MLM never asked `[CLS]` to summarise sentiment, and a linear classifier can't change the representation. Fine-tuning can. Frozen features still make sense with little compute, many tasks (one shared encoder, a 769-parameter classifier each), or very little labelled data. This is the same lesson as ELMo vs BERT in [[2026-09-29 - Pre-trained Language Models#Fine-tuning vs embeddings: ELMo is not BERT]].
> - **Dynamic padding:** pad each batch only to its longest sequence. It saves compute, and thanks to the attention mask it gives the same predictions.

---

## 4. In-context learning: results and limits

### What it can do

- **TriviaQA** (closed-book QA: "Miami Beach in Florida borders which ocean?"): GPT-3's accuracy rises steadily with size. Few-shot GPT-3 175B matches the fine-tuned SOTA. The slide's warning: **"No access to supporting documents! Only memorisation!!"** The model answers from what it stored during pretraining. Retrieval comes later in the course.
- **Machine translation:** the model was **not explicitly trained to translate**, and **not purposefully trained on multilingual data** (only ~7% of GPT-3's data isn't English). It still learns *some* translation ability. Few-shot GPT-3 is below supervised systems **into** French/German/Romanian, but **beats or matches the supervised SOTA translating into English** (Fr→En 39.2 vs 35.0, De→En 40.6 vs 40.2, Ro→En 39.5 vs 39.9). **The model is much better at translating *into* English** than out of it, because English is what it saw most.

### Necessity of scale

![[assets/2026-10-06/llms-050.png]]

On the same task, adding examples in context helps **a lot at 175B** (GPT-3, ~10× T5 scale), **a little at 13B** (~T5 scale, "the largest model we normally fine-tune"), and **barely at 1.3B** (GPT-2 scale).

> **In-context learning only works well with VERY large models.**

### It's sensitive

![[assets/2026-10-06/llms-052.png]]

- **Example selection matters:** GPT-3 on SST-2 sentiment with 5 random demonstrations from the training set, 5 trials: accuracy from **86.9% to 95.8%**.
- **Example order matters:** the *same* examples in 24 different orders give a large spread. The variance is **largest for small models** and shrinks, but doesn't vanish, at 175B.

ICL is not a silver bullet ("not a panacea").

### Why does it work?

**There is no clear, unifying theory yet.** The evidence on the slide: models do better at in-context reasoning on terms that were **more frequent in the pretraining corpus**. The lecturer's picture: think of the model as a huge storage system into which a lot of information has been crammed, where the right prompt is the **trigger** that reaches the relevant region (see [52:38](https://brightspace.au.dk/content/enforced/219812-LR60743/recordings_2026/video1764972533.mp4#t=3158)).

---

## 5. Making in-context learning more effective

### Prompt engineering

**Patterns** for expressing the input, plus a **verbalizer** that maps labels to words. For Yelp ratings (1–5 stars):

- patterns: $P_1(a)$ = "It was ___. $a$", $P_3(a)$ = "$a$. All in all, it was ___.", $P_4(a)$ = "$a$ ‖ In summary, the restaurant is ___."
- verbalizer: $v(1)$ = terrible, $v(2)$ = bad, $v(3)$ = okay, $v(4)$ = good, $v(5)$ = great.

Why it helps: the model is still a **text generator**. Constraining how the task is presented and what the answer words are makes the generation more controlled. **Good prompts can even improve performance at small scale.**

It's mostly **trial and error**, with no principled theory. Folk findings mentioned in class: models do better when information is given in **Markdown** (they've seen a lot of it from GitHub), or even in **uppercase** (see [54:13](https://brightspace.au.dk/content/enforced/219812-LR60743/recordings_2026/video1764972533.mp4#t=3253)).

### Searching for the prompt: AutoPrompt

**Gradient-guided search:** initialise a set of **trigger tokens** in the prompt template, estimate how the loss changes if each trigger is replaced by another vocabulary token, and keep the best ones.

![[assets/2026-10-06/llms-058.png]]

> **The optimal prompts found are sometimes sequences of random words.** For sentiment: "atmosphere alot dialogue Clone totally [MASK]".

They work because they were selected for their effect on the loss, but they mean nothing to a human. **This is the foundation of "jailbreaking"**: searching for token sequences that make a model **bypass its guardrails** and do what it isn't meant to (e.g. give a recipe for a bomb), by analogy with jailbreaking a phone (see [59:45](https://brightspace.au.dk/content/enforced/219812-LR60743/recordings_2026/video1764972533.mp4#t=3585)). It comes back in the responsible-AI lecture.

### Learning the prompt: soft prompts / prompt tuning

If the best discrete prompt is gibberish anyway, why insist on real tokens? **Learn the prompt.** For each task:

1. Initialise a special input **prompt vector** (or several).
2. **Prepend** it to every example of the task.
3. Compute the gradients of the loss **with respect to the prompt parameters**.
4. **Only the prompt vectors are updated. The rest of the model is frozen.**

![[assets/2026-10-06/llms-062.png]]

**Menti: (dis)advantages of prompt tuning?** (see [1:08:01](https://brightspace.au.dk/content/enforced/219812-LR60743/recordings_2026/video1764972533.mp4#t=4081))

- ✅ **More efficient training**: only the prompt vectors are trained (82K parameters per task vs 11B for a fine-tuned copy). It's not "less efficient", which some answered.
- ✅ **Separable multi-task learning**: one frozen model, one small prompt per task, and you can mix tasks in the same batch. With model tuning you need a separate fine-tuned copy per task.
- ❌ **Not as interpretable as discrete prompts**: you get dense vectors, not text you can read.

![[assets/2026-10-06/llms-063.png]]

**As scale increases** (SuperGLUE, T5 models):

- **prompt tuning becomes competitive with full model fine-tuning**, while hand-designed prompts (GPT-3-style "prompt design") stay far below;
- **the length of the prompt matters less** (1 token is bad, beyond ~5–20 little changes);
- **the initialisation stops mattering** (random, sampled vocabulary or class labels converge).

> **You need large models for these benefits to emerge.** That's emergence again.

---

## 6. Chain-of-thought prompting

![[assets/2026-10-06/llms-070.png]]

**Standard prompting** shows the question and the answer ("The answer is 11."). For a multi-step problem the model often gets it wrong (27 instead of 9).

**Chain-of-thought (CoT):** for complex problems, don't just show the example prompt and the answer. **Demonstrate the reasoning process** in the in-context example ("Roger started with 5 balls. 2 cans of 3 tennis balls each is 6 tennis balls. 5 + 6 = 11."). The model **learns to produce an explanation as it decodes the answer**, and mimics the pattern on the new question ("23 − 20 = 3 … 3 + 6 = 9").

![[assets/2026-10-06/llms-071.png]]

> **CoT needs scale to work.** On GSM8K, SVAMP and MAWPS, CoT only beats standard prompting for the largest models (LaMDA/GPT/PaLM at ~100B+). Below that it doesn't help, or it hurts.

⚠️ **Does the model "reason"?** The lecturer asked you to accept the word "reasoning" *for now*. A whole lecture on whether models can reason comes after the midterm (see [1:18:43](https://brightspace.au.dk/content/enforced/219812-LR60743/recordings_2026/video1764972533.mp4#t=4723)). On 20 Oct the stance was explicit: "there is no thinking going on… the model is still just mimicking a pattern, but mimicking that pattern actually allows the model to perform better".

> [!example] Lab — prompting a 27B model on GSM8K (`07 - Finetuning and Prompting/1. Prompting.ipynb`)
> - **Chat format:** messages with roles `system` / `user` / `assistant`. The API is **stateless**: to continue a conversation you resend all of it. Behind the API the messages are joined with special tokens (the **chat template**) and the model just *continues the text* as the assistant. Few-shot prompts are built by writing the `assistant` turns yourself.
> - **Direct answers, 0- vs 5-shot:** demonstrations change little. Question→number pairs teach the **format**, not how to do math. **In-context examples are good at showing format, label set and style, and weak at adding new abilities.**
> - **CoT** raises accuracy a lot (to ~90%) at several times the token cost. **Why it works:** a Transformer does a **fixed amount of computation per generated token** (one pass through the layers). Answering directly forces the whole calculation into that one pass. Writing the steps lets each intermediate result become **input for later tokens**, like a scratch pad. (Lab MCQ: CoT replies get *longer*, not shorter.)
> - **Temperature:** use **0** to *measure* accuracy (reproducible, best guess) and higher values for variety.
> - **Self-consistency:** sample several CoT paths at temperature > 0 and take a **majority vote** on the final answers. Right paths agree, wrong ones scatter. Agreement is also a **confidence signal**.
> - **Evaluation hygiene:** with 20 test problems each one is 5 points, and the standard error is ~9 points. Small differences are noise.

### Wrap-up of this half

- **Abilities emerge at scale:** in-context learning, chain-of-thought.
- They can be improved with **prompt engineering** (manual design, mining/paraphrasing, gradient-based search, prompts generated by LMs) and **prompt tuning** (soft prompts).
- Both exploit that models are big and trained on lots of data, and that **every task can be mapped to a natural-language description**: predicting a label = producing the word for it.

> **Guide the model to perform tasks using natural-language descriptions.**

---

## 7. From GPT-3 to ChatGPT: instruction tuning and RLHF

ChatGPT (late 2022) **follows instructions and explains its answers**. The recipe has three steps (from InstructGPT). The same flavour is behind every chat model today, with different RL algorithms and data:

![[assets/2026-10-06/llms-081.png]]

| step | data humans provide | what's trained |
|---|---|---|
| **1. SFT** | a prompt is sampled; a labeller **writes the desired output** ("Explain reinforcement learning to a 6-year-old" → their explanation) | GPT-3.5 fine-tuned with supervised learning |
| **2. Reward model** | several model outputs for the same prompt; a labeller **ranks them** best → worst | a **reward model** (RM) that scores outputs |
| **3. RL** | new prompts; **no human in the loop**: the RM gives the reward | the policy, optimised with **PPO** against the RM |

**Data collected:** ~75k task demonstrations in total, split 13k for SFT, 31k for the reward model, and 33k for PPO (slide numbers; the InstructGPT paper lists 33k for the RM and 31k for PPO).

Why the split matters: **writing** a good answer (step 1) is slow and expensive, so that's the smallest set. **Ranking** answers (step 2) is much easier, so you can collect more of it.

### 7.1 Step 1: supervised fine-tuning (SFT)

Train the model to generate the next word of the **demonstration** $y^*$ given the preceding words of the demonstration and the **instruction** $x^*$:

$$\mathcal L_{SFT} = -\sum_{t=1}^{T} \log P\big(y^*_t \mid \{y^*\}_{<t};\ \{x^*\}\big)$$

It's exactly next-token prediction with teacher forcing, as in pretraining ([[2026-09-29 - Pre-trained Language Models#Pretraining]]). Only the names changed: the input is called the *instruction* and the output the *demonstration*.

### 7.2 What's wrong with only SFT?

![[assets/2026-10-06/llms-084.png]]

1. **Expensive to collect lots of data.** Every example is a human writing a full answer. (Class answers: "we'd need a lot of answers written by hand", "with little data a lot of probabilities will be 0 or 1".) Fine-tuning *compute* isn't the issue; the pretrained model already exists.
2. **Exposure bias.** **During training**, the model's inputs are **gold** context tokens from real human-written text: $\mathcal L_{MLE} = -\log P(y^*_t \mid \{y^*\}_{<t})$. **At generation time**, its inputs are its **own previously decoded tokens**: $\mathcal L_{dec} = -\log P(\hat y_t \mid \{\hat y\}_{<t})$. The model is never trained on the situation it's in at test time. Once its own outputs drift from human text, it is on unfamiliar ground. The slide's example: a continuation that starts fine and degenerates into repeating "Universidad Nacional Autónoma de México" again and again (see [1:37:49](https://brightspace.au.dk/content/enforced/219812-LR60743/recordings_2026/video1764972533.mp4#t=5869)).

> The repetition in that example has a second cause, covered on 20 Oct: MLE-trained models become *more* confident each time they repeat something, and greedy-style decoding exploits that. See [[2026-10-20 - Natural Language Generation#4. Why greedy text degenerates]]. Exposure bias explains why the model drifts. The confidence loop explains why it gets stuck once it does.

Steps 2 and 3 address both problems: they need only cheap comparison data, and in step 3 the model is trained **on its own generations**.

### 7.3 Step 2: the reward model

We can't have humans judge every output during training. So:

- **Initialise a reward model to predict a score** for any demonstration (prompt + output). The score represents its **quality** given the instruction.
- **Train it on human comparisons:** for two demonstrations $Y_+$ and $Y_-$ (you know one is better than the other),

$$\mathcal L_{RM} = -\log \sigma\big(R(Y_+) - R(Y_-)\big)$$

![[assets/2026-10-06/llms-086.png]]

As $R(Y_+) - R(Y_-) \to \infty$, $\sigma(\cdot) \to 1$ and $\log(\cdot) \to 0$. The loss pushes the **gap** between the good and the bad score up, which puts them in "far-apart regions". The lecturer compared it to a margin loss, the same spirit as an SVM. (The standard name is the Bradley–Terry pairwise loss.) Only the **difference** matters, so the absolute scale of $R$ is arbitrary. That's relevant for the baseline below.

**Outcome:** for any demonstration $Y$, we can now get a score $R(Y)$, with no human needed.

### 7.4 Step 3: optimising with RL

![[assets/2026-10-06/llms-087.png]]

- **Rollout:** the LM (the *policy*) generates a response to a query ("This movie is" → "really great!").
- **Evaluation:** the reward model scores query + response (e.g. 1.0). The slide notes the reward could also come from a classifier, a rule, or a human.
- **Optimisation:** the **active** model's log-probabilities are compared with a frozen **reference** copy's (a **KL-divergence** term keeps them close), and the reward is used to update the active model with **PPO**.

In words: step 1 shows the model good outputs, step 2 learns what a good output looks like, and step 3 trains the model, at every iteration, to generate outputs with higher reward. How generation can be seen as RL at all (state = context, action = next token, policy = the model/decoder) is spelled out in [[2026-10-20 - Natural Language Generation#8. Text generation as reinforcement learning]].

> ⚠️ **Exam:** "we will never ask you about reinforcement learning directly in the exam… we will just ask you what we have taught you in the lectures" (see [1:29:22](https://brightspace.au.dk/content/enforced/219812-LR60743/recordings_2026/video1764972533.mp4#t=5362)). So no RL theory beyond these slides, but what *is* on them is fair game: the 2025 exam had a question on PPO (below).

### 7.5 REINFORCE

![[assets/2026-10-06/llms-089.png]]

Given an instruction, **sample a demonstration from the model** ($\hat y_1, \hat y_2, \dots$, each fed back in), then:

$$\mathcal L_{RL} = -R(\hat Y) \sum_{t=1}^{T} \log P\big(\hat y_t \mid \{x^*\};\ \{\hat y\}_{<t}\big)$$

- The $\sum \log P$ part: **next time, increase the probability of this sampled token in the same context**.
- The $R(\hat Y)$ factor: **…but do it more if the reward was high.** It's the MLE loss with each sequence weighted by its reward.
- The reward is given to the **whole sequence**, and every token gets the same weight. Assigning credit to individual tokens is the **credit assignment** problem, which is hard. (20 Oct; see [[2026-10-20 - Natural Language Generation#8. Text generation as reinforcement learning]].)

> ⚠️ One correction to the 6 Oct explanation. The lecturer described the sampled token as "the right token… we know this from our supervised training data". In REINFORCE, $\hat y$ (with a hat) is **sampled from the model itself**. There is no gold token here. Only the reward says whether the sample was good. That's the whole difference from SFT, where $y^*$ is gold.

**Implementation tricks:**

1. **Start from the SFT model**, not the raw pretrained one.
2. **Subtract a baseline** $b$: $\ \mathcal L_{RL} = -(R(\hat Y) - b)\sum_t \log P(\hat y_t \mid \dots)$. **Why?** If the reward model always gives positive scores (say in $[0,1]$), then *every* sampled generation gets reinforced, just some more than others. You **never penalise a generation**, even garbage. With a baseline, below-average generations get a negative weight and are pushed *down* (see [1:51:05](https://brightspace.au.dk/content/enforced/219812-LR60743/recordings_2026/video1764972533.mp4#t=6665)).
3. **Keep a term that encourages coherence:** $\mathcal L = \mathcal L_{MLE} + \alpha\, \mathcal L_{RL}$. Chasing reward alone can drift away from fluent language. The MLE term keeps the model a good language model.

### 7.6 PPO (Proximal Policy Optimization)

Covered on 20 Oct (see [2:42:18](https://au.cloud.panopto.eu/Panopto/Pages/Viewer.aspx?id=3afaa41e-2164-4a9a-ad0a-b4e90076fde1&start=9738)). **PPO is what was actually used for ChatGPT.** It's REINFORCE plus "hooks" that stop the model from moving too far from where it started.

![[assets/2026-10-06/llms-094.png]]

$$\mathcal L_{PPO} = -\min\!\left( R(\hat Y)\sum_{t=1}^{T} \log \frac{P_{curr}(\hat y_t \mid \{x^*\};\{\hat y\}_{<t})}{P_{base}(\hat y_t \mid \{x^*\};\{\hat y\}_{<t})},\ \ g(\epsilon, R(\hat Y)) \right) \qquad g = \begin{cases}(1+\epsilon)R(\hat Y) & R(\hat Y) \ge 0\\ (1-\epsilon)R(\hat Y) & R(\hat Y) < 0\end{cases}$$

- **Two models:** $P_{curr}$ is being optimised; $P_{base}$ is the model **before RL** (the SFT model) and stays fixed.
- **The ratio** $P_{curr}/P_{base}$ replaces the plain probability. It gets **extreme values when $P_{curr}$ and $P_{base}$ deviate**.
- **The min** limits how much your policy can deviate from the original base policy in **every step**. If the reward is positive, the objective is **capped at $(1+\epsilon)R$**: no extra credit for moving further. That's the "proximal" in the name.
- In the pipeline there's also the **KL-divergence** term between the active and the reference model (slide 87). The result stays coherent, while generations that humans prefer get rewarded.

> Standard PPO (my addition, for context): the clip is on the probability **ratio** $r_t = P_{curr}/P_{base}$ itself, $\ \min\big(r_t A,\ \text{clip}(r_t, 1-\epsilon, 1+\epsilon)\,A\big)$, with $A$ an advantage (reward minus baseline). The slide's version is a simplified rendering of the same idea: **cap the update once the new policy has moved more than $\epsilon$ from the old one.** For the exam, the idea is what matters.

> ⚠️ **Exam (2025 Q3 = mock exam):** *Why does RLHF use PPO instead of REINFORCE?* → **PPO stabilises updates by limiting policy drift.** The distractors: it does **not** avoid catastrophic forgetting (that's about old tasks), does **not** manage exploration (that's reward design/sampling), and does **not** reduce the *reward* variance (PPO doesn't touch the reward model, only the policy).

### 7.7 Outcome

![[assets/2026-10-06/llms-097.png]]

- **Instruction-tuned smaller models are better at human tasks than GPT-3.** Win rate against SFT-175B: PPO > SFT > prompted GPT > GPT at every size. A 1.3B PPO model beats 175B GPT-3. SFT alone already helps, and PPO on top helps much more.
- **Multi-task abilities**: planning a two-day Paris trip, writing attention-mechanism code…
- **Personalisation**: "explain the causes of WWI" for an expert vs for a 10-year-old gives different, appropriate answers.

### 7.8 DPO (Direct Preference Optimization)

Covered on 20 Oct (see [2:47:33](https://au.cloud.panopto.eu/Panopto/Pages/Viewer.aspx?id=3afaa41e-2164-4a9a-ad0a-b4e90076fde1&start=10053)).

![[assets/2026-10-06/llms-101.png]]

RLHF needs a separate reward model trained from the preferences, *then* RL against it. **DPO** asks: we already have the preference data, so why not optimise for human preferences **directly**?

$$\mathcal L_{DPO} = -\log \sigma\!\left( \beta \log \frac{P_{curr}(Y_+ \mid x^*)}{P_{base}(Y_+ \mid x^*)} - \beta \log \frac{P_{curr}(Y_- \mid x^*)}{P_{base}(Y_- \mid x^*)} \right)$$

- It's the **reward-model loss** $-\log\sigma(R(Y_+) - R(Y_-))$ with $R(Y) = \beta\log\frac{P_{curr}(Y\mid x)}{P_{base}(Y\mid x)}$ plugged in. The policy's own probability ratio *is* the reward, which is why it's called the **implicit reward**. Both objectives are crammed into one equation.
- **The objective pushes the ratio $P_{curr}/P_{base}$ to be higher for the preferred sample $Y_+$ than for $Y_-$.** Like PPO, it keeps the model anchored to the base policy.
- **No separate reward model, no RL loop:** a simple, classification-style loss on preference pairs.

> **Less flexible than RLHF in theory, but works well in practice.**

Two lines to take away, in the lecturer's words: "DPO makes the preferred response more likely and the rejected response less likely relative to a frozen base model. It turns preference learning into a simple classification-style objective instead of training a separate reward model and doing RL on top of it." There's a proof that, after a transformation, RLHF and DPO optimise the same objective. ⚠️ "I'm not going to ask you about that proof in the midterm" (see [2:50:42](https://au.cloud.panopto.eu/Panopto/Pages/Viewer.aspx?id=3afaa41e-2164-4a9a-ad0a-b4e90076fde1&start=10242)).

> [!example] Lab — SFT, LoRA and DPO on a 0.5B base model (`07 - Finetuning and Prompting/2. Preference Alignment.ipynb`)
> - **Base model** (Qwen2.5-0.5B): treats a question as the start of a web document. It may answer, but it adds more questions, uses no consistent format, and **doesn't stop** (it never learned that `<|im_end|>` ends its turn).
> - **SFT** = next-token cross-entropy on prompt→completion conversations, with the **loss only on the completion** (prompt tokens get label `-100`). We want it to learn to *answer*, not to write questions. Learning to predict the end-of-turn token = learning **when to stop**. SFT on 2,000 examples mainly teaches **format and behaviour**. The knowledge comes from pretraining (the **superficial alignment hypothesis**, LIMA).
> - **LoRA:** freeze $W$, learn a low-rank update $h = Wx + \frac{\alpha}{r}BAx$ with $B$ initialised to **0**, so training starts exactly at the pretrained model. With $r=16$ on all projections that's 8.8M trainable parameters (~2%), a 35 MB adapter instead of a 2 GB model, and ~2.1 GB instead of ~7.9 GB of training memory (AdamW needs ~16 bytes per *trained* parameter). It works because fine-tuning updates have a low intrinsic rank. ⚠️ Lab MCQ: LoRA ≠ storing weights in 4 bits (that's **Q**LoRA) ≠ training only biases (BitFit).
> - **DPO:** data = (prompt, **chosen**, **rejected**). **Larger β = held closer to the reference model** (β plays the role of RLHF's KL penalty). The reference is a **frozen** copy of the starting (SFT) model, so DPO *does* need a reference model, but no reward model. Tokens shared by chosen and rejected **cancel out** in the loss, so the model learns only from where they differ (that's Step-DPO's point). A known quirk: the margin grows, but often the chosen response's probability goes **down** too, just less than the rejected one's.
> - In practice preference alignment shapes **behaviour that's easy to compare but hard to demonstrate**: helpfulness, harmlessness, tone, refusing unsafe requests.

---

## Wrap-up

- **Larger scale** (more parameters, data, compute) → **emergent** abilities: **in-context learning**, **chain-of-thought**.
- Emergent abilities enable new, efficient ways to adapt models: **prompt engineering**, **prompt tuning**.
- **Instruction tuning and RLHF** make models adaptable to new, never-seen tasks, as long as the task can be stated in natural language.

---

## Da ricordare

- **Emergence** = quantitative change (scale) → qualitative change (new abilities): an ability present in large models but absent in small ones. Controversial term. **ICL and CoT only work with very large models.**
- **In-context learning** = task description + **demonstrations** + prompt in **one input sequence**. **No gradient updates.** The model "learns" by **attending** to the examples. Zero-shot = description only (GPT-2); fine-tuning = weights updated (GPT/BERT). It's **sensitive to which examples and in what order** (variance larger for small models). There's **no theory** of why it works. GPT-3 translates **into English** better than out of it.
- **Prompt engineering** (patterns + verbalizers) is trial and error. **Gradient search** (AutoPrompt) finds gibberish prompts → the foundation of **jailbreaking**. **Prompt tuning** = learn soft prompt vectors with the model **frozen**: efficient, multi-task, **not interpretable**. With scale it **matches full fine-tuning**, and length and initialisation stop mattering.
- **CoT** = demonstrate the **reasoning steps** in the examples. Why it works (lab): **fixed compute per token**, so written steps become input for later tokens. It **needs scale**. Self-consistency = majority vote over sampled CoT paths.
- **GPT-3 → ChatGPT:** (1) **SFT** on human-written demonstrations (cross-entropy, teacher forcing); (2) **reward model** on human **rankings**, $-\log\sigma(R(Y_+)-R(Y_-))$; (3) **RL (PPO)** against the RM, with a KL term to a reference model. Writing is expensive, ranking is cheap.
- **SFT alone:** expensive data + **exposure bias** (trained on gold prefixes, generates from its own → drift).
- **REINFORCE:** $-R(\hat Y)\sum\log P(\hat y_t\mid\dots)$ on the model's **own samples**. **Baseline** $b$, otherwise with always-positive rewards you **never penalise** a generation. Add $\mathcal L_{MLE}$ for coherence. Start from the SFT model.
- **PPO** = REINFORCE + ratio $P_{curr}/P_{base}$ + **cap at $(1\pm\epsilon)R$** → ⚠️ **stabilises updates by limiting policy drift** (2025 exam Q3; not forgetting, not exploration, not reward variance). **DPO** = the RM loss with $R = \beta\log\frac{P_{curr}}{P_{base}}$: **no reward model, no RL**, still a **reference model**; larger β = closer to it. "Less flexible in theory, works well in practice."
