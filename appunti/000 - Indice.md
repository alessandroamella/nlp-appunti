# Index — NLP

Map of the course. One line per topic: the link to the note plus a sentence on
what it covers.

## Lecture notes

1. [[2026-08-25 - Introduction to NLP]] — what NLP is, the major processing tasks, historical framing, BoW and TF-IDF.
2. [[2026-09-01 - Neural Networks and Word Embeddings]] — row/column normalization of TF-IDF, neural network primer, word embeddings.
3. [[2026-09-08 - Language Modeling (Part 1)]] — what a language model is, chain rule, n-grams and the Markov assumption, perplexity, smoothing, fixed-context neural LMs, intro to RNNs.
4. [[2026-09-15 - Language Modeling (Part 2)]] — BPTT, deep and bidirectional RNNs, vanishing gradients, LSTM and GRU, gates as learned residual connections.
5. [[2026-09-22 - Seq2Seq Models and Transformers]] — encoder–decoder models, the temporal bottleneck, attention (query/keys/values), self-attention, multi-head attention, the transformer block, masking and cross-attention, position embeddings (preview).
6. [[2026-09-29 - Pre-trained Language Models]] — position embeddings (sinusoidal vs learned; RoPE/ALiBi in the lab), the quadratic cost and sparse attention, GPT (causal LM), ELMo (contextual embeddings from biLSTMs), BERT (masked LM, [CLS], fine-tuning vs frozen), RoBERTa/DistilBERT/ELECTRA, BART and T5.
7. [[2026-10-06 - From GPT to ChatGPT]] — scaling and emergence, in-context learning, prompt engineering and prompt tuning, chain-of-thought, SFT → reward model → RL (REINFORCE, PPO), DPO.
8. [[2026-10-20 - Natural Language Generation]] — the generation step and teacher forcing, argmax and beam search, why greedy text repeats, sampling (top-k, top-p, temperature), re-balancing and re-ranking, MLE's lack of diversity, exposure bias, generation as RL and reward hacking.

## Scratch notes

Not lecture notes — different conventions apply, don't reformat them.

- [[2026-09-01 - Robe miste]] — esempio numerico svolto di TF-IDF e normalizzazione di riga/colonna.

## To review

<!-- Topics that don't click yet. Delete the line when they do. -->

## Exam

- Format: **midterm Fri 06 Nov 2026, 9:00–12:00** (starts at 9:00, not 9:15), rooms 5344-101 "Benjamin" and 5123-122 "Seminarrum"; room assignments via Brightspace by Fri 30 Oct. MCQ only. A confusing question can be raised within a week and discounted. See [[2026-10-20 - Natural Language Generation#Exam info given in this lecture]].
- Out of syllabus:
  - State-space models / Mamba (mentioned only). See [[2026-09-29 - Pre-trained Language Models]].
  - Diffusion models for text. See [[2026-10-20 - Natural Language Generation]].
  - The proof that RLHF and DPO optimise the same objective. See [[2026-10-06 - From GPT to ChatGPT]].
- Said by the lecturer:
  - No hand-computed gradients (backprop/BPTT): understand the mechanism, don't do the arithmetic. See [[2026-09-15 - Language Modeling (Part 2)]].
  - No under-specified interpretation questions (e.g. "what does this LSTM cell track?"). If one appears, it comes with more information and you can ask for clarification. See [[2026-09-15 - Language Modeling (Part 2)]].
  - No reinforcement learning "directly": only what was taught in the lectures (REINFORCE, baseline, PPO, DPO at slide level). The 2025 exam asked why RLHF uses PPO. See [[2026-10-06 - From GPT to ChatGPT]].
