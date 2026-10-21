<!-- Note: 2026 exam will only have a MCQ part, which will be longer -->

# MCQs (3 Points * 10 = 30 Points)

Each question has only **one** correct option, and there are no partial points.

---

### Question 1

**What is the primary purpose of backpropagation in neural networks?**

A. To compute gradients of the loss function with respect to the input data

B. To adjust the learning rate dynamically during training

C. To propagate error values backward through the network to refine predictions

D. To compute gradients of the loss function with respect to the network’s parameters

**[Your Answer]:**

**D**

* **A:** Even though sometimes this gradient is computed, it's very rare (e.g., adversarial attacks). Thus it is not the primary purpose of backpropagation.
* **B:** Backpropagation doesn't handle the learning rate adjustments. That is the optimizer's job.
* **C:** Error values are only to compute gradients, and have nothing to do with refining predictions.

---

### Question 2

**A model has a perplexity of 20 on dataset A and 40 on dataset B. Which conclusion is correct?**

A. The model performs twice as well on A as on B

B. The model is twice as confident on A as on B

C. The model’s vocabulary coverage is higher on dataset B than on dataset A.

D. Perplexity cannot be compared across datasets.

**[Your Answer]:**

**D. Perplexity cannot be compared across datasets.**

---

### Question 3

**Why does RLHF use PPO (Proximal Policy Optimization) instead of standard policy gradient methods like REINFORCE?**

A. PPO avoids catastrophic forgetting

B. PPO allows the model to explore unseen tokens safely

C. PPO reduces the reward variance

D. PPO stabilizes updates by limiting policy drift

**[Your Answer]:**

**D**

* **A:** Catastrophic forgetting refers to a model forgetting old tasks when learning new ones. PPO doesn’t address this; its goal is stability in updates, not memory retention.
* **B:** PPO doesn’t directly manage exploration. That’s influenced by reward design, entropy bonuses, or sampling strategies.
* **C:** PPO does not affect the reward model in any way, just the policy.

---

### Question 4

Consider the following statements about Word2Vec:

**Statements**

1. Word2Vec usually can generalize better to rare and unseen words than those using one-hot vectors.
2. Both CBOW and Skip-gram algorithms do not take into account the order of words that appear in the context window.
3. Shuffling the order of words within each sentence does not affect the Word2Vec vector representations.

**Which of the above statements are TRUE?**

A. 2

B. 1, 3

C. 1, 2

D. 2, 3

**[Your Answer]:**

**A**

* **1:** Word2Vec can generalize better to rare words (if seen) than one-hot vectors, because it learns distributed representations. However, for truly unseen words, Word2Vec cannot generalize at all.
* **2:** In both CBOW and Skip-gram, the context is treated as a bag of words. The order of words within the context window is ignored.
* **3:** Shuffling sentences changes which words co-occur within each context window (e.g., A[CBD]E -> A[DBE]C).

---

### Question 5

**Which of the following accurately describes a key disadvantage of using recurrent neural networks (RNNs) for modeling long sequences?**

A. RNNs compress all historical context into a single fixed-size state vector, causing information loss.

B. RNNs always have quadratic computational complexity with respect to sequence length, making them impractical for long sequences.

C. RNNs require extremely large datasets for training because each timestep uses different parameter sets.

D. RNNs cannot learn nonlinear representations because their hidden states must remain linear combinations of past inputs.

**[Your Answer]:**

**A**

* **B:** RNNs have linear computational complexity with respect to sequence length.
* **C:** RNNs reuse the same parameters at each timestep.
* **D:** RNNs are nonlinear models; their hidden states are updated via nonlinear activation functions.

---

### Question 6

**In a Transformer encoder-decoder model, each decoder block utilizes masked self-attention followed by encoder-decoder cross-attention. Which of the following correctly explains the role of masked self-attention in the decoder?**

A. It enables the decoder to attend to all encoder outputs, aligning the input and output sequences.

B. It randomly masks part of the encoder outputs to act as a regularization mechanism similar to dropout, preventing overfitting.

C. It allows the decoder to attend to all positions in the output sequence during training to improve parallelism.

D. It ensures the decoder does not attend to future tokens during training to maintain autoregressive property.

**[Your Answer]:**

**D**

* **A:** This describes encoder-decoder cross-attention, not masked self-attention.
* **B:** The mask in masked self-attention is not random.
* **C:** It cannot attend to all positions in the output sequence because of the mask.

---

### Question 7

Given a training set containing the following sequences:

> "the cat sat on the mat"
> "the dog chased after the cat"
> "a cat and a dog ran"
> "she sat on the sofa"
> "they bought a new mat"
> "the sofa was comfortable"

The training set includes **18 unique tokens** and **32 tokens in total**. We consider the words split by whitespace as the tokens in the vocabulary. We don’t consider adding any special tokens (e.g., `<UNK>` or `<stop>`) unless specifically stated.

Please answer the following questions in the context of **N-gram language modeling**:

To avoid the Out of Vocabulary problem, we apply the add-one Laplace with smoothing, where the probability of each bigram can be computed as:

$$P(A\vert{}B) = \frac{\text{count}(B, A) + \alpha}{\text{count}(B) + \alpha \cdot \vert{}V\vert{}}$$

After applying add-one Laplace smoothing $\alpha = 1$ to the bigram model, what are the probabilities of $P(\text{sat}\vert{}\text{on})$ and $P(\text{on}\vert{}\text{sat})$?

---

A. None of the other options is correct.

B. $P(\text{sat}\vert{}\text{on}) = \frac{1}{20}$, $P(\text{on}\vert{}\text{sat}) = \frac{3}{20}$.

C. $P(\text{sat}\vert{}\text{on}) = 1$, $P(\text{on}\vert{}\text{sat}) = 0$.

D. $P(\text{sat}\vert{}\text{on}) = \frac{3}{20}$, $P(\text{on}\vert{}\text{sat}) = \frac{1}{20}$.

**[Your Answer]:**

**B**

Bigrams in the corpus include sat $\rightarrow$ on twice and on $\rightarrow$ sat zero times.

Counts: $\text{count}(\text{sat})=2$, $\text{count}(\text{on})=2$, vocabulary $\vert{}V\vert{}=18$, $\alpha=1$.

---

### Question 8

**In a Transformer’s self-attention layer, which one of the following statements is correct?**

A. The self-attention mechanism ensures that each token attends only to previous tokens, which allows the model to preserve causal order.

B. The attention weights are computed by applying a softmax over the Query matrix, normalizing the importance of each input dimension within a single token representation.

C. Increasing the dimension of the Query and Key vectors does not change the computational complexity of the attention matrix.

D. The scaled dot-product attention divides the dot product of Query and Key by $\sqrt{d_k}$ to prevent the softmax function from entering regions with extremely small gradients when the vector dimension is large.

**[Your Answer]:**

**D**

* **A:** It describes only masked self-attention, not every self-attention layer.
* **B:** Softmax is applied over the dot product of Query and Key.
* **C:** Increasing the dimension of the Query and Key vectors increases the parameter count and computational complexity.

---

### Question 9

**In a BART/T5 model, what is the keys and queries for the cross-attention module in the decoder?**

A. Decoder’s token embeddings as keys and encoder outputs as queries.

B. Decoder hidden states as keys and output of the final encoder layer as queries.

C. Encoder hidden states as keys and decoder hidden states as queries.

D. The output of the final encoder layer as keys and decoder hidden states as queries.

**[Your Answer]:**

**D**

BART/T5 are Encoder-Decoder models. The output of the final encoder layer acts as keys, and decoder hidden states act as queries in the cross-attention module in the decoder.

---

### Question 10

**The following statements about model applicability are correct:**

A. T5 can only perform generation tasks without any modification and cannot perform text classification tasks.

B. BERT cannot perform text generation tasks without any modification.

C. A fixed-window language model using only average pooling can perform generation tasks.

D. ELMo can perform both generation and classification tasks without any modification.

**[Your Answer]:**

**B**

* **A:** T5 is designed so that every NLP task is framed as text-to-text generation.
* **C:** A fixed-window language model with average pooling captures only local context and lacks an autoregressive or sequential prediction mechanism.
* **D:** ELMo only provides contextual word embeddings trained with bidirectional LSTMs.
