# AI Agents Workshop - Worldcoin Brazil HH 2025
This repo contains some useful resources for quick starting you journey to develop AI Agents.

created by 0xZapLabs with ♥
## 📕 List of contents
Contents that will be covered in this workshop:
- Quick Intro to LLM
- Most popular LLM models
- AI "beyond" LLM
- When AI meets the Real World
- Giving superpowers to LLM
- How to make an AI trade on your behalf? 
- An Intro to Agentic Systems
- Most common architectures
- Zap in a nutshell
- What is next?

## Introduction
As computers get more powerful, AI is getting smarter incredibly fast. Now we regularly see new models with billions of parameters doing things that used to be just science fiction. Characters like Jarvis, Joi, and Rachel from movies are close to becoming real.

You might be wondering: What are these models? How do they work? Can you build one? What can you actually do with them? And how do you turn a language model into a useful assistant like the ones in movies?

In this workshop, we'll answer these questions and more as we explore AI agents and how they work in the real world. We'll cover the basics and show you how to build practical systems so you can start creating your own AI agents.

## Quick Intro to LLM
Language models have long been a cornerstone of natural language processing (NLP), using mathematical techniques to capture the rules and patterns of language for prediction and generation. Decades of research have taken us from early statistical language models (SLMs)—simple systems counting word frequencies—to today’s large language models (LLMs), which boast billions of parameters and can produce human-like text. This swift evolution, especially in recent years, has turned science fiction into reality, with models now able to chat, reason, and assist in ways that rival characters like Jarvis from Iron Man.

But what’s behind this leap? LLMs rely on transformer architectures, a breakthrough in neural network design that excels at understanding context by processing entire sequences of words at once. Trained on massive datasets—think books, websites, and more—these models generalize language laws, predicting what comes next in a sentence or generating responses from scratch. In this workshop, we’ll peel back the layers of LLMs, showing you how they work and why they’re the backbone of modern AI agents. Whether you’re a coder, a dreamer, or just curious, you’ll leave with a clear grasp of their potential.

| Era            | Model Type          | Key Characteristics                                              | Examples             |
|----------------|---------------------|------------------------------------------------------------------|----------------------|
| 1990s         | Statistical LMs (SLMs) | Probability distributions for word sequences, Maximum Likelihood Estimation | Early NLP systems   |
| Early 2000s   | Neural LMs (NLMs)    | Neural networks, word vectors (Word2Vec), Softmax for probabilities | Word2Vec models     |
| 2010s         | Pre-trained LMs (PLMs) | Pre-training on large corpora, fine-tuning for tasks, transformer-based | BERT, GPT-2         |
| 2020s-Present | Large LMs (LLMs)     | Tens/hundreds of billions parameters, human-level text, in-context learning | GPT-3, ChatGPT, LLaMA |

### A little bit of history

#### Statistical Language Models (SLMs)
Born in the 1990s, SLMs were the first step in modeling language probabilistically. They aimed to calculate the likelihood of a sentence—like "I am very happy"—appearing in a text, expressed as \( P(S) = P(I, am, very, happy) \). Using conditional probabilities, this breaks down to \( P(I) \times P(am | I) \times P(very | I, am) \times P(happy | I, am, very) \). To estimate these, SLMs relied on Maximum Likelihood Estimation, substituting probabilities with word sequence frequencies from a training set, e.g., \( P(w_i | w_1 \ldots w_{i-1}) = \frac{C(w_1 \ldots w_i)}{C(w_1 \ldots w_{i-1})} \). The n-gram approach assumed each word depended only on the prior \( n-1 \) words, but with \( n \) typically limited to 2 or 3 due to storage constraints, accuracy suffered as vocabulary size grew exponentially.

#### Neural Language Models (NLMs)
By the early 2000s, NLMs emerged, swapping statistical counts for neural networks. They introduced word vectors—numerical representations where "cat" and "dog" are closer in vector space than "cat" and "tree," thanks to tools like Word2Vec. An NLM's structure includes an input layer (concatenating vectors of \( n-1 \) prior words), a hidden layer (applying non-linear functions like sigmoid), and an output layer (predicting the next word via Softmax, e.g., \( O_i' = \frac{\exp(O_i)}{\sum_{j=1}^{|V|} \exp(O_j)} \)). This allowed NLMs to handle longer dependencies and capture semantic relationships, overcoming SLMs' rigid n-gram limits.

#### Pre-Trained Language Models (PLMs)

The 2010s brought PLMs like BERT and GPT-2, pre-trained on vast unlabeled text to learn language fundamentals—vocabulary, syntax, semantics—before fine-tuning on specific tasks like translation or question-answering. Think of it as a martial artist building a strong foundation before mastering a sword. Pre-training on huge corpora, followed by fine-tuning on smaller, task-specific datasets, made PLMs versatile and powerful, driven by the rise of transformer architectures.

### Transformer: Attention Is All You Need
In this section, we will try to give you a grasp of how a transformer works. First of all, at a high level, transformers are a type of neural network architecture introduced in the 2017 paper *"Attention Is All You Need"* by Vaswani et al. Unlike earlier models like SLMs and NLMs, which processed words sequentially or struggled with long-range dependencies, transformers revolutionized NLP by processing entire sequences of words simultaneously. This parallel processing, powered by a mechanism called "attention," allows transformers to capture context—like understanding the difference between "bank" in "river bank" vs. "money bank"—far more effectively.

#### How Transformers Work: The Big Picture
At their core, transformers are built to handle sequences, whether it’s translating a sentence, generating text, or answering questions. They consist of two main parts: an **encoder** and a **decoder**. The encoder processes the input sequence (e.g., a sentence in English) and creates a rich representation of it, while the decoder generates the output sequence (e.g., the translated sentence in French). What makes transformers special is their use of **self-attention**, a mechanism that lets each word in the sequence "pay attention" to every other word, figuring out which ones are most relevant. For example, in the sentence "The cat, which slept all day, was tired," self-attention helps the model understand that "cat" and "tired" are related, even though they’re far apart.

#### Self-Attention: The Magic Ingredient
Self-attention is the heart of the transformer. Here’s how it works: for each word in the input, the model creates three vectors **Query (Q)**, **Key (K)**, and **Value (V)** by multiplying the word’s vector (like the ones from Word2Vec in NLMs) by three different weight matrices. Then, it computes an attention score by taking the dot product of the Query of one word with the Key of every other word, scaling it, and applying a Softmax function to get a probability distribution. Mathematically, the attention output for a word is:

Attention(Q, K, V) = Softmax(QK^T/sqrt{d_k})V \]

where \( d_k \) is the dimension of the Key vector, used for scaling to prevent large values from destabilizing the computation. This process lets the model weigh the importance of other words when understanding a given word, capturing relationships that SLMs’ n-grams or NLMs’ sequential processing couldn’t.

#### Multi-Head Attention and Layers
Transformers don’t stop at one round of attention—they use **multi-head attention**, running the attention mechanism multiple times in parallel with different weight matrices. This allows the model to focus on different aspects of the sequence simultaneously (e.g., syntax, semantics). The outputs of these "heads" are concatenated and transformed to produce the final representation. A transformer typically stacks multiple layers of this multi-head attention, along with feed-forward neural networks, normalization, and residual connections, to refine the representation at each step.

#### Why Transformers Win
Transformers outperform earlier models for several reasons:
- **Parallel Processing**: Unlike sequential models, transformers process all words at once, making them faster to train on GPUs.
- **Long-Range Dependencies**: Self-attention captures relationships between distant words, solving the limitations of SLMs’ small \( n \)-grams and NLMs’ weaker dependency handling.
- **Scalability**: Transformers scale well with more data and compute, paving the way for LLMs like GPT-3 and beyond.

In short, transformers are the backbone of modern NLP, enabling everything from chatbots to translation systems. Their ability to understand context through attention makes them a game-changer, and they’re the foundation for the LLMs we’ll explore next.

### Exercise: Let's code a basic transformer

### Factors Influencing LLM Development
What fueled this rise? Three big drivers stand out:

- **Compute Power**: As computers got faster (think GPUs and TPUs), training giant models became feasible.
- **Data Explosion**: The internet provided a flood of text—Wikipedia, forums, X posts—giving LLMs raw material to learn from.
- **Algorithmic Breakthroughs**: Transformers and techniques like fine-tuning made models smarter and more adaptable.

Today’s LLMs, like GPT-3 (175 billion parameters, 570GB of text) vs. GPT-2 (1.5 billion parameters, 40GB), showcase this scale. Their ability to align with human values and generalize across tasks marks a leap from specialized PLMs to all-purpose language wizards.

For a more detailed review of LLMs, please refer to: https://arxiv.org/html/2402.06853v3

## Most popular LLM models
There’s a growing lineup of LLM models, each with unique strengths. Here’s a quick rundown of some heavy hitters as of early 2025:

- GPT Series (OpenAI): From ChatGPT to GPT-4 and beyond, these models are known for their versatility and conversational prowess. They excel in natural language understanding and generation. (NOTE: really good for structured outputs, will get back to it later)
- LLaMA (Meta AI): Optimized for research, LLaMA models are lightweight yet powerful, often used as a base for custom fine-tuning.
- Grok (xAI): Designed by X to be truthful and helpful, it was built to assist and provide clear answers, often with a dash of outside perspective on humanity.
Gemini (Google): A multimodal contender, blending text, images, and more, with a focus on integration into Google’s ecosystem.
- Claude (Anthropic): Prioritizing safety and interpretability, Claude is a favorite for applications where alignment with human values is key.
Each model has trade-offs—some are open-source, others proprietary; some shine in reasoning, others in creative tasks. We’ll explore what makes these models tick and how to pick the right one for your AI agent.