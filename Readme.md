# Mechanistic Interpretability Workshop: Part 1
**Uncovering the Hidden Knowledge of Toy Transformers**

Welcome to the Mechanistic Interpretability workshop! In this first session, we will dive into the internal representations of a "Toy Transformer." Even though small language models often generate incoherent text, they frequently learn sophisticated internal features about language structure (syntax, grammar, and boundaries).

We will use **Linear Probes** and **PyTorch Hooks** to surgically extract these activations and prove what the model knows, layer by layer.

## 🧠 What is a Linear Probe?

A linear probe is a diagnostic tool used to understand deep learning models. Think of it like a "mind-reading" device for neural networks.

1.  **The Premise:** As data moves through the layers of a Transformer, the model transforms it into high-dimensional vectors (embeddings).
2.  **The Hypothesis:** If the model understands a concept (like "the current word is a verb"), that information must be encoded linearly within those vectors.
3.  **The Method:** We freeze the model. We extract the internal activation vectors and train a simple linear classifier (like Logistic Regression) to predict a specific feature (e.g., "Is this a space?").
4.  **The Result:** If the simple classifier achieves high accuracy, we know the model has explicitly learned and represented that feature at that specific layer.

## 🛠️ Installation & Setup

### 1. Prerequisites
You will need:
* **Python 3.8+**
* A code editor that supports **Interactive Python Cells** (highly recommended).
    * **VS Code** (install the Python extension)
    * **Spyder** or **JupyterLab**
    * *Note: The script uses `# %%` cell markers, which allows you to run code in blocks like a notebook.*

### 2. Installation
Install the required libraries using pip:

```bash
pip install -r requirements.txt
```

## 🚀 How to Follow the Workshop
Open ```Linear_Probes.py```. This file is designed to be run **interactively**.

### Recommended Workflow (VS Code):

1. Open ```Linear_Probes.py```.
2. You will see "Run Cell" | "Run Below" Options appear above the ```# %%``` comments.
3. Click "**Run Cell** to execute one block at a time.
4. Watch the output in the Interactive Window on the right.

### The Workshop Activities
The script guides you through four main stages:

1.  **The Hook:** We will write a custom PyTorch Hook to intercept the data flowing through the model's layers without modifying the model source code.
2. **Task 1 - Space Detection:** We will extract activations and train a probe to see if the model knows where spaces belong.
3. **Task 2 - Capitalization:** We will repeat the process for capital letters.
4. **Diagnostics:** We will challenge our findings. Is the probe actually learning, or it is just guessing? We will compare our results against a "Random Noise" baseline.

## 🧪 Exercises

At the bottom of the script, there is an "Exercises" section. Try to complete at least:

* **Exercise 1 (Diagnostics):** Validate the capitalization probe

* **Exercise 2 (New Task):** Write a probe for Vowels vs. Consonants.

---

# Mechanistic Interpretability Workshop: Part 2
**Probing GPT-2 with TransformerLens**

In Part 2, we scale up! We move from toy transformers to **GPT-2** (124M parameters) and use **TransformerLens**, a library specifically designed for mechanistic interpretability research.

## 🎯 What You'll Learn

In this session, you'll discover:

1. **TransformerLens Basics**: How to access any layer's activations with simple, clean code
2. **Factual Recall Probing**: Does GPT-2 know when sentences are factually plausible?
3. **Negation Understanding**: Can GPT-2 track whether statements are affirmative or negated?
4. **Contrastive Learning**: How to properly create positive/negative examples for probing
5. **Hierarchical Processing**: Different features emerge at different layers!

## 🔬 Key Discoveries

By the end of Part 2, you'll see:

- **Factual plausibility** peaks in middle layers (5-6) - requires semantic processing
- **Negation detection** emerges early (layers 1-2) - it's a surface lexical feature!
- **Different capabilities, different layers** - reveals the computational structure of transformers

## 📁 Files for Part 2

* **`workshop_part2_gpt2_probing.py`** - Main workshop script with two probing tasks
* Uses the same `# %%` cell markers for interactive execution

## 🛠️ Additional Requirements for Part 2

Install TransformerLens:

```bash
pip install transformer-lens
```

Note: This will download GPT-2 small (~500MB) on first run.

## 🚀 How to Follow Part 2

1. Open `workshop_part2_gpt2_probing.py`
2. Run cells interactively (same as Part 1)
3. The script has two main sections:
   - **Task 1: Factual Recall** - "Paris is the capital of France" vs "Paris is the capital of Germany"
   - **Task 2: Negation Understanding** - "Paris is in France" vs "Paris is not in France"
4. Watch how different features emerge at different layers!

## 🎓 What Makes Part 2 Different?

### Conceptual Advances:
- **Sentence-level probing**: Run complete sentences, not just prompts
- **Contrastive examples**: Compare correct vs incorrect completions
- **Balanced datasets**: Avoid the "majority class trap"
- **Diagnostic rigor**: Always compare to base rate and random noise

### Technical Advances:
- **TransformerLens**: Much cleaner than manual PyTorch hooks
- **Larger models**: GPT-2 (124M) vs toy models (~1M)
- **Token-level**: Work with subword tokens, not characters
- **Layer patterns**: Discover hierarchical processing

## 🧪 Exercises for Part 2

The script includes 8 exercises:

1. **Analyze Layer Patterns** - Why does negation emerge earlier than facts?
2. **Test Different Facts** - Do mathematical facts peak at the same layers as geography?
3. **Sentiment Detection** - Create a positive/negative sentiment probe
4. **Increase Dataset Size** - Does more data change the patterns?
5. **Probe MLP Outputs** - What does each component (attention vs MLP) compute?
6. **Analyze Failure Cases** - What does the probe get wrong?
7. **Cross-Task Generalization** - Do probes transfer between tasks?
8. **Temporal Understanding** - Create a past/present tense probe

## 🔑 Key Takeaways

### Part 1 Taught You:
- How to use PyTorch hooks
- How to train linear probes
- How to validate probes with diagnostics
- Character-level features in toy models

### Part 2 Teaches You:
- How to use TransformerLens (much easier!)
- How to probe large models (GPT-2)
- How to design proper contrastive datasets
- **That transformers are hierarchical**: syntax → semantics → output

### The Big Picture:
Transformers aren't black boxes! Different layers specialize in different computations:
- **Early layers**: Surface features (negation markers, basic syntax)
- **Middle layers**: Semantic processing (factual plausibility, meaning)
- **Later layers**: Output preparation (next token prediction)

Linear probes let us **measure** this hierarchy scientifically!

## 📚 Next Steps

After completing both parts, you can:

- Try other features: sentiment, tense, grammaticality
- Explore attention patterns with TransformerLens visualization tools
- Try activation patching to test **causal** importance
- Read papers from Redwood Research, Anthropic, and EleutherAI on mech interp
- Join the mech interp community and share your findings!

## 🙏 Acknowledgments

This workshop builds on techniques from:
- The TransformerLens library by Neel Nanda
- Probing work by Belinkov, Hewitt, and many others
- The broader mechanistic interpretability research community


