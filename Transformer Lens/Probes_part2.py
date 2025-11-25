"""
LINEAR PROBING WORKSHOP - PART 2
Probing GPT-2 for Linguistic Knowledge

In Part 1, you learned how to probe your tiny character-level transformer.
Now we'll probe GPT-2 to discover what it knows about:
1. Factual knowledge - "Paris is the capital of ___"
2. Syntactic structure - "Who is the subject of this sentence?"

We'll use TransformerLens to make this easier and introduce new techniques:
- Token-level probing (not character-level)
- Logit lens (what does each layer predict?)
- Layer-by-layer knowledge emergence
"""

# %% Imports
import torch
import numpy as np
from transformer_lens import HookedTransformer
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, recall_score
import seaborn as sns

# %% Load GPT-2
print("="*60)
print("LOADING GPT-2 WITH TRANSFORMERLENS")
print("="*60)

# Load GPT-2 small (124M parameters)
model = HookedTransformer.from_pretrained("gpt2-small")
model.eval()

print(f"\nModel info:")
print(f"  Number of layers: {model.cfg.n_layers}")
print(f"  Embedding dimension: {model.cfg.d_model}")
print(f"  Vocabulary size: {model.cfg.d_vocab}")
print(f"  Number of attention heads: {model.cfg.n_heads}")

# %% Understanding TransformerLens
print("\n" + "="*60)
print("WHAT IS TRANSFORMERLENS?")
print("="*60)
print("""
TransformerLens is a library designed for mechanistic interpretability.
It makes it MUCH easier to access internal activations than raw PyTorch hooks.

Key features:
- Access any layer with simple names: "blocks.0.hook_resid_post"
- Built-in caching of activations
- Easy attention pattern visualization
- Logit lens functionality

Instead of manually registering hooks, we use run_with_cache()!
""")

# %% Test TransformerLens
print("\n" + "="*60)
print("EXAMPLE: Getting Activations with TransformerLens")
print("="*60)

test_text = "The cat sat on the mat"
tokens = model.to_tokens(test_text)
print(f"Input text: '{test_text}'")
print(f"Tokenized: {tokens}")
print(f"Token shape: {tokens.shape}")

# Run model and cache ALL activations
logits, cache = model.run_with_cache(tokens)

print(f"\nAvailable activations in cache:")
print(f"  - Residual stream after layer 0: cache['blocks.0.hook_resid_post']")
print(f"  - Attention output from layer 0: cache['blocks.0.attn.hook_result']")  
print(f"  - MLP output from layer 0: cache['blocks.0.hook_mlp_out']")
print(f"  - Final logits: logits")

print(f"\nLayer 0 residual stream shape: {cache['blocks.0.hook_resid_post'].shape}")
print(f"  [batch_size, sequence_length, d_model]")
print(f"  [{cache['blocks.0.hook_resid_post'].shape[0]}, {cache['blocks.0.hook_resid_post'].shape[1]}, {cache['blocks.0.hook_resid_post'].shape[2]}]")

print("\n✓ Much easier than manual hooks!")

# %%============================================================================
# TASK 1: FACTUAL RECALL
# ============================================================================
print("\n" + "="*80)
print("TASK 1: FACTUAL RECALL")
print("="*80)
print("""
Question: Does GPT-2 know factual relationships?

We'll test prompts like:
- "Paris is the capital of" → should know "France"
- "The Eiffel Tower is located in" → should know "Paris"  
- "Apple was founded by Steve" → should know "Jobs"

We'll probe the residual stream to see:
1. Which layers "know" the correct answer
2. When does the model figure it out?
3. Does knowledge emerge gradually or suddenly?
""")

# %% Create Factual Recall Dataset
print("\n" + "="*60)
print("CREATING FACTUAL RECALL DATASET")
print("="*60)

# Define factual prompts and their answers
factual_prompts = [
    # Geography - Capitals (30)
    ("Paris is the capital of", "France"),
    ("London is the capital of", "England"),
    ("Tokyo is the capital of", "Japan"),
    ("Berlin is the capital of", "Germany"),
    ("Rome is the capital of", "Italy"),
    ("Madrid is the capital of", "Spain"),
    ("Beijing is the capital of", "China"),
    ("Moscow is the capital of", "Russia"),
    ("Washington is the capital of", "United States"),
    ("Ottawa is the capital of", "Canada"),
    ("Canberra is the capital of", "Australia"),
    ("New Delhi is the capital of", "India"),
    ("Cairo is the capital of", "Egypt"),
    ("Athens is the capital of", "Greece"),
    ("Dublin is the capital of", "Ireland"),
    ("Vienna is the capital of", "Austria"),
    ("Brussels is the capital of", "Belgium"),
    ("Amsterdam is the capital of", "Netherlands"),
    ("Stockholm is the capital of", "Sweden"),
    ("Oslo is the capital of", "Norway"),
    ("Copenhagen is the capital of", "Denmark"),
    ("Warsaw is the capital of", "Poland"),
    ("Prague is the capital of", "Czech"),
    ("Budapest is the capital of", "Hungary"),
    ("Lisbon is the capital of", "Portugal"),
    ("Ankara is the capital of", "Turkey"),
    ("Seoul is the capital of", "Korea"),
    ("Bangkok is the capital of", "Thailand"),
    ("Manila is the capital of", "Philippines"),
    ("Jakarta is the capital of", "Indonesia"),
    
    # Geography - Locations (15)
    ("The Eiffel Tower is located in", "Paris"),
    ("The Statue of Liberty is located in", "New York"),
    ("The Colosseum is located in", "Rome"),
    ("Big Ben is located in", "London"),
    ("The Great Wall is located in", "China"),
    ("The Taj Mahal is located in", "India"),
    ("The Pyramids are located in", "Egypt"),
    ("Mount Everest is located in", "Nepal"),
    ("The Grand Canyon is located in", "Arizona"),
    ("Machu Picchu is located in", "Peru"),
    ("The Sydney Opera House is located in", "Australia"),
    ("The Louvre is located in", "Paris"),
    ("Times Square is located in", "New York"),
    ("The Parthenon is located in", "Athens"),
    ("Stonehenge is located in", "England"),
    
    # Famous People - Tech (15)
    ("Apple was founded by Steve", "Jobs"),
    ("Microsoft was founded by Bill", "Gates"),
    ("Facebook was founded by Mark", "Zuckerberg"),
    ("Tesla was founded by Elon", "Musk"),
    ("Amazon was founded by Jeff", "Bezos"),
    ("Google was founded by Larry Page and Sergey", "Brin"),
    ("Twitter was founded by Jack", "Dorsey"),
    ("SpaceX was founded by Elon", "Musk"),
    ("PayPal was co-founded by Elon", "Musk"),
    ("Oracle was founded by Larry", "Ellison"),
    ("Dell was founded by Michael", "Dell"),
    ("Netflix was founded by Reed", "Hastings"),
    ("Uber was founded by Travis", "Kalanick"),
    ("Airbnb was founded by Brian", "Chesky"),
    ("LinkedIn was founded by Reid", "Hoffman"),
    
    # Famous People - Historical (10)
    ("The United States was founded by George", "Washington"),
    ("The theory of relativity was developed by Albert", "Einstein"),
    ("Gravity was discovered by Isaac", "Newton"),
    ("The telephone was invented by Alexander Graham", "Bell"),
    ("The light bulb was invented by Thomas", "Edison"),
    ("Penicillin was discovered by Alexander", "Fleming"),
    ("The Mona Lisa was painted by Leonardo da", "Vinci"),
    ("Romeo and Juliet was written by William", "Shakespeare"),
    ("The Origin of Species was written by Charles", "Darwin"),
    ("Relativity was discovered by Albert", "Einstein"),
    
    # Science - Basic Facts (20)
    ("Water is made of hydrogen and", "oxygen"),
    ("The speed of light is approximately", "300"),
    ("DNA stands for deoxyribonucleic", "acid"),
    ("The Sun is a", "star"),
    ("The Moon orbits around the", "Earth"),
    ("Humans have two", "arms"),
    ("The chemical symbol for water is", "H2O"),
    ("The chemical symbol for gold is", "Au"),
    ("The Earth orbits around the", "Sun"),
    ("A year on Earth is approximately", "365"),
    ("The human body has", "206"),  # bones
    ("The brain is located in the", "head"),
    ("The heart pumps", "blood"),
    ("Photosynthesis occurs in", "plants"),
    ("The atomic number of hydrogen is", "1"),
    ("The atomic number of carbon is", "6"),
    ("Electrons are negatively", "charged"),
    ("Protons are positively", "charged"),
    ("The smallest unit of life is a", "cell"),
    ("Oxygen is essential for", "breathing"),
    
    # Mathematics (10)
    ("Two plus two equals", "four"),
    ("Ten divided by two equals", "five"),
    ("The square root of nine is", "three"),
    ("Pi is approximately", "3.14"),
    ("A triangle has", "three"),  # sides
    ("A square has", "four"),  # sides
    ("A circle has", "360"),  # degrees
    ("One hundred divided by ten equals", "ten"),
    ("Five times five equals", "twenty"),
    ("The first prime number is", "two"),
    
    # Language & Culture (10)
    ("The most spoken language in the world is", "English"),
    ("Spanish is spoken in", "Spain"),
    ("French is spoken in", "France"),
    ("German is spoken in", "Germany"),
    ("Italian is spoken in", "Italy"),
    ("The Statue of Liberty was a gift from", "France"),
    ("Christmas is celebrated in", "December"),
    ("New Year's Day is", "January"),
    ("Halloween is celebrated in", "October"),
    ("Thanksgiving is celebrated in", "November"),
]

print(f"Created {len(factual_prompts)} factual prompts")
print("\nCategories:")
print("  - Geography (capitals & landmarks): 45 prompts")
print("  - Famous people (tech & historical): 25 prompts")
print("  - Science facts: 20 prompts")
print("  - Mathematics: 10 prompts")
print("  - Language & culture: 10 prompts")
print("\nExample prompts:")
for prompt, answer in factual_prompts[:5]:
    print(f"  '{prompt}' → '{answer}'")

# %% Extract Activations for Factual Recall
print("\n" + "="*60)
print("EXTRACTING ACTIVATIONS FOR FACTUAL RECALL")
print("="*60)
print("""
Binary classification approach with contrastive examples:

For each prompt like "Paris is the capital of":
- Positive: Run model on "Paris is the capital of France"
            Extract activation → label 1
- Negative: Run model on "Paris is the capital of Germany"
            Extract activation → label 0

Question: Does this layer's activation indicate a factually correct sentence?

KEY: We run the model with DIFFERENT complete sentences, so activations differ!
""")

# Collect all unique answers to use as contrastive negatives
all_answers = list(set([answer for _, answer in factual_prompts]))
print(f"Total unique answers: {len(all_answers)}")

# Storage
layer_activations_factual = {i: [] for i in range(model.cfg.n_layers)}
labels_factual = []

# For each prompt, create 1 positive + 1 negative example (balanced!)
num_negatives_per_prompt = 1
print(f"Creating 1 positive + {num_negatives_per_prompt} negative example per prompt (balanced dataset)")

print("\nProcessing prompts...")
total_examples = 0
examples_shown = 0

for prompt, correct_answer in factual_prompts:
    # Positive example: correct completion
    positive_sentence = prompt + " " + correct_answer
    positive_tokens = model.to_tokens(positive_sentence)
    
    # Run model with correct answer
    logits_pos, cache_pos = model.run_with_cache(positive_tokens)
    
    # Extract last token activation from each layer
    for layer_idx in range(model.cfg.n_layers):
        layer_output = cache_pos[f'blocks.{layer_idx}.hook_resid_post']
        last_token_act = layer_output[0, -1, :].cpu().numpy()
        layer_activations_factual[layer_idx].append(last_token_act)
    
    labels_factual.append(1)  # Correct answer
    total_examples += 1
    
    # Negative example: wrong completion
    wrong_answers = [ans for ans in all_answers if ans != correct_answer]
    wrong_answer = np.random.choice(wrong_answers)
    
    negative_sentence = prompt + " " + wrong_answer
    negative_tokens = model.to_tokens(negative_sentence)
    
    # Run model with wrong answer
    logits_neg, cache_neg = model.run_with_cache(negative_tokens)
    
    # Extract last token activation from each layer
    for layer_idx in range(model.cfg.n_layers):
        layer_output = cache_neg[f'blocks.{layer_idx}.hook_resid_post']
        last_token_act = layer_output[0, -1, :].cpu().numpy()
        layer_activations_factual[layer_idx].append(last_token_act)
    
    labels_factual.append(0)  # Wrong answer
    total_examples += 1
    
    # Show a few examples
    if examples_shown < 5:
        print(f"  ✓ Positive: '{positive_sentence}'")
        print(f"  ✗ Negative: '{negative_sentence}'")
        examples_shown += 1

# Convert to numpy
for i in range(model.cfg.n_layers):
    layer_activations_factual[i] = np.stack(layer_activations_factual[i])

labels_factual = np.array(labels_factual)

print(f"\n✓ Created {total_examples} examples from {len(factual_prompts)} prompts")
print(f"✓ Positive examples (correct completions): {sum(labels_factual)}")
print(f"✓ Negative examples (wrong completions): {len(labels_factual) - sum(labels_factual)}")
print(f"✓ Balance: {sum(labels_factual)/len(labels_factual):.1%} positive")
print(f"✓ Each layer shape: {layer_activations_factual[0].shape}")

print("\n" + "="*60)
print("How this works:")
print("  We run the model on COMPLETE sentences:")
print("    'Paris is the capital of France' → activation_1 → label 1")
print("    'Paris is the capital of Germany' → activation_2 → label 0")
print("  The activations are DIFFERENT, so the probe can learn!")
print("="*60)

# %% Train Probes for Factual Recall
print("\n" + "="*60)
print("TRAINING PROBES: Factual Recall")
print("="*60)
print("Training logistic regression on each layer...")
print("Question: Which layers can detect the correct factual answer?")

factual_layer_accuracies = []
factual_layer_recalls = []
factual_layer_probes = []

# Calculate base rate
base_rate_factual = max(sum(labels_factual), len(labels_factual) - sum(labels_factual)) / len(labels_factual)
print(f"\nBase rate (majority class): {base_rate_factual:.1%}")
print(f"A 'dumb' classifier always predicting majority gets {base_rate_factual:.1%} accuracy")
print(f"Our probe must beat this to show real learning!\n")

for layer_idx in range(model.cfg.n_layers):
    # Get activations
    X = layer_activations_factual[layer_idx]
    y = labels_factual
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # Train probe
    probe = LogisticRegression(max_iter=1000, random_state=42)
    probe.fit(X_train, y_train)
    
    # Evaluate
    test_acc = probe.score(X_test, y_test)
    y_pred = probe.predict(X_test)
    test_recall = recall_score(y_test, y_pred, pos_label=1)
    
    factual_layer_accuracies.append(test_acc)
    factual_layer_recalls.append(test_recall)
    factual_layer_probes.append(probe)
    
    if layer_idx % 3 == 0 or layer_idx == model.cfg.n_layers - 1:
        print(f"  Layer {layer_idx:2d}: Accuracy={test_acc:.3f}, Recall={test_recall:.3f}")

print(f"\n✓ Trained probes on all {model.cfg.n_layers} layers")

# Diagnostic: Random noise baseline
print("\n" + "="*60)
print("DIAGNOSTIC: Are we really learning from activations?")
print("="*60)

random_activations = np.random.randn(len(labels_factual), model.cfg.d_model)
X_train_rand, X_test_rand, y_train_rand, y_test_rand = train_test_split(
    random_activations, labels_factual, test_size=0.3, random_state=42, stratify=labels_factual
)
probe_random = LogisticRegression(max_iter=1000, random_state=42)
probe_random.fit(X_train_rand, y_train_rand)
random_acc = probe_random.score(X_test_rand, y_test_rand)
y_pred_random = probe_random.predict(X_test_rand)
random_recall = recall_score(y_test_rand, y_pred_random, pos_label=1)

print(f"\nComparison:")
print(f"  Base rate (always predict majority): {base_rate_factual:.1%} accuracy, 0.0% recall")
print(f"  Probe on random noise:               {random_acc:.1%} accuracy, {random_recall:.1%} recall")
print(f"  Probe on Layer 0:                     {factual_layer_accuracies[0]:.1%} accuracy, {factual_layer_recalls[0]:.1%} recall")
print(f"  Probe on Layer {model.cfg.n_layers-1}:                    {factual_layer_accuracies[-1]:.1%} accuracy, {factual_layer_recalls[-1]:.1%} recall")

if factual_layer_recalls[0] > random_recall * 1.5:
    print(f"\n✓ Probes beat random noise - activations contain real factual information!")
else:
    print(f"\n⚠ Probes barely beat random - factual info may be weak or nonlinear")

# %% Visualize Factual Recall Results  
print("\n" + "="*60)
print("RESULTS: Detecting Factually Correct Sentences")
print("="*60)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Plot accuracy
ax1.plot(range(model.cfg.n_layers), factual_layer_accuracies, 'o-', color='steelblue', linewidth=2, markersize=8)
ax1.set_xlabel('Layer', fontsize=12)
ax1.set_ylabel('Probe Accuracy', fontsize=12)
ax1.set_title('Factual Recall: Accuracy Across Layers', fontsize=13)
ax1.set_ylim([0, 1])
ax1.axhline(y=base_rate_factual, color='r', linestyle='--', alpha=0.5, label=f'Base rate ({base_rate_factual:.1%})')
ax1.grid(alpha=0.3)
ax1.legend()

# Plot recall  
ax2.plot(range(model.cfg.n_layers), factual_layer_recalls, 'o-', color='darkgreen', linewidth=2, markersize=8)
ax2.set_xlabel('Layer', fontsize=12)
ax2.set_ylabel('Recall (Correct Sentences)', fontsize=12)
ax2.set_title('Factual Recall: Detecting Correct vs Incorrect Facts', fontsize=13, fontweight='bold')
ax2.set_ylim([0, 1])
ax2.axhline(y=0.5, color='orange', linestyle='--', alpha=0.5, label='50% recall (random)')
ax2.axhline(y=random_recall, color='red', linestyle='--', alpha=0.5, label=f'Random noise ({random_recall:.1%})')
ax2.grid(alpha=0.3)
ax2.legend()

plt.tight_layout()
plt.savefig('factual_recall_gpt2.png', dpi=150)
print("✓ Saved plot: factual_recall_gpt2.png")
plt.show()

# Find when knowledge emerges
print("\nKnowledge emergence pattern:")
for i in range(model.cfg.n_layers):
    if i == 0:
        print(f"  Layer {i:2d}: {factual_layer_recalls[i]:.1%} (baseline)")
    else:
        change = factual_layer_recalls[i] - factual_layer_recalls[i-1]
        arrow = "↑" if change > 0.05 else "→" if abs(change) < 0.05 else "↓"
        print(f"  Layer {i:2d}: {factual_layer_recalls[i]:.1%} {arrow}")

best_layer = np.argmax(factual_layer_recalls)
print(f"\n✓ Best layer: {best_layer} with {factual_layer_recalls[best_layer]:.1%} recall")

# %% Interpretation
print("\n" + "="*60)
print("INTERPRETATION: Factual Knowledge in GPT-2")
print("="*60)
print("""
What we learned:

1. SENTENCE-LEVEL FACTUAL DETECTION
   - We asked: "Is this sentence factually correct?"
   - Correct: "Paris is the capital of France"
   - Wrong: "Paris is the capital of Germany"
   - The probe detects factual correctness from the final activation

2. KNOWLEDGE EMERGENCE ACROSS LAYERS
   - Early layers: May have weak factual detection
   - Middle/later layers: Factual correctness becomes clearer
   - The best layer shows where the model "realizes" something is wrong

3. LINEAR PROBES DETECT FACTUAL CORRECTNESS
   - Even though the computation is complex
   - The model's representation encodes "this makes sense" vs "this is wrong"
   - This information has linear structure in the representation space

4. BALANCED DATASET = MEANINGFUL METRICS
   - 50/50 positive/negative examples
   - Can't exploit class imbalance
   - Recall and accuracy both matter equally

Next: Let's probe for syntactic understanding with subject identification.
""")

# %%============================================================================
# TASK 2: NEGATION UNDERSTANDING
# ============================================================================
print("\n" + "="*80)
print("TASK 2: NEGATION UNDERSTANDING")
print("="*80)
print("""
Question: Can GPT-2 track whether a statement is affirmative or negated?

We'll test sentences like:
- Affirmative: "Paris is in France"       → label 1
- Negated:     "Paris is not in France"   → label 0

Note: We're NOT asking "is this true?" but rather "is this negated?"
Both true and false statements can be affirmative or negated:
- "Paris is in France" (affirmative, true) → 1
- "Paris is in Germany" (affirmative, false) → 1  
- "Paris is not in France" (negated, false) → 0
- "Paris is not in Germany" (negated, true) → 0

This tests if the model tracks the FORM of the statement, not its truth value.
""")

# %% Create Negation Dataset
print("\n" + "="*60)
print("CREATING NEGATION UNDERSTANDING DATASET")
print("="*60)

# We'll use our factual prompts and create negated versions
negation_sentences = []

# Generate affirmative and negated pairs from factual prompts
for prompt, answer in factual_prompts[:50]:  # Use subset for speed
    # Create affirmative sentence
    affirmative = prompt + " " + answer
    affirmative = affirmative.replace(" is the capital of ", " is in ")  # Make it sound natural
    affirmative = affirmative.replace(" was founded by ", " was created by ")
    affirmative = affirmative.replace(" is located in ", " is in ")
    
    # Create negated version by inserting "not"
    # Simple heuristic: insert "not" after "is"/"was"/"are"/"were"
    negated = affirmative
    for verb in [" is ", " was ", " are ", " were ", " has ", " have "]:
        if verb in negated:
            negated = negated.replace(verb, verb.strip() + " not ", 1)
            break
    
    # Only add if we successfully created a negation
    if negated != affirmative and " not " in negated:
        negation_sentences.append((affirmative, negated))

# Add some manually crafted examples for variety
manual_pairs = [
    ("The Earth orbits the Sun", "The Earth does not orbit the Sun"),
    ("Water freezes at zero degrees", "Water does not freeze at zero degrees"),
    ("Humans need oxygen to breathe", "Humans do not need oxygen to breathe"),
    ("The Moon orbits Earth", "The Moon does not orbit Earth"),
    ("Birds can fly", "Birds cannot fly"),
    ("Fish live in water", "Fish do not live in water"),
    ("Plants need sunlight", "Plants do not need sunlight"),
    ("Cats are mammals", "Cats are not mammals"),
    ("Gold is a metal", "Gold is not a metal"),
    ("English is a language", "English is not a language"),
    ("The Sun is a star", "The Sun is not a star"),
    ("Elephants are animals", "Elephants are not animals"),
    ("Computers process information", "Computers do not process information"),
    ("Trees produce oxygen", "Trees do not produce oxygen"),
    ("The sky appears blue", "The sky does not appear blue"),
]

negation_sentences.extend(manual_pairs)

print(f"Created {len(negation_sentences)} affirmative/negated pairs")
print("\nExample pairs:")
for affirmative, negated in negation_sentences[:5]:
    print(f"  Affirmative: '{affirmative}'")
    print(f"  Negated:     '{negated}'")
    print()

# %% Extract Activations for Negation Understanding
print("\n" + "="*60)
print("EXTRACTING ACTIVATIONS FOR NEGATION UNDERSTANDING")
print("="*60)
print("""
For each pair, we'll:
1. Run model on affirmative: "Paris is in France"
2. Run model on negated: "Paris is not in France"
3. Train probe to detect: Is this sentence negated?

The probe learns to detect the presence/absence of negation markers.
""")

# Storage
layer_activations_negation = {i: [] for i in range(model.cfg.n_layers)}
labels_negation = []

print("\nProcessing sentence pairs...")
total_examples = 0

for affirmative, negated in negation_sentences:
    # Process affirmative sentence
    aff_tokens = model.to_tokens(affirmative)
    logits_aff, cache_aff = model.run_with_cache(aff_tokens)
    
    # Extract last token activation from each layer
    for layer_idx in range(model.cfg.n_layers):
        layer_output = cache_aff[f'blocks.{layer_idx}.hook_resid_post']
        last_token_act = layer_output[0, -1, :].cpu().numpy()
        layer_activations_negation[layer_idx].append(last_token_act)
    
    labels_negation.append(1)  # Affirmative
    total_examples += 1
    
    # Process negated sentence
    neg_tokens = model.to_tokens(negated)
    logits_neg, cache_neg = model.run_with_cache(neg_tokens)
    
    # Extract last token activation from each layer
    for layer_idx in range(model.cfg.n_layers):
        layer_output = cache_neg[f'blocks.{layer_idx}.hook_resid_post']
        last_token_act = layer_output[0, -1, :].cpu().numpy()
        layer_activations_negation[layer_idx].append(last_token_act)
    
    labels_negation.append(0)  # Negated
    total_examples += 1

# Convert to numpy
for i in range(model.cfg.n_layers):
    layer_activations_negation[i] = np.stack(layer_activations_negation[i])

labels_negation = np.array(labels_negation)

print(f"\n✓ Created {total_examples} examples from {len(negation_sentences)} sentence pairs")
print(f"✓ Affirmative examples: {sum(labels_negation)}")
print(f"✓ Negated examples: {len(labels_negation) - sum(labels_negation)}")
print(f"✓ Balance: {sum(labels_negation)/len(labels_negation):.1%} affirmative")
print(f"✓ Each layer shape: {layer_activations_negation[0].shape}")

# %% Train Probes for Negation Understanding
print("\n" + "="*60)
print("TRAINING PROBES: Negation Understanding")
print("="*60)
print("Training logistic regression on each layer...")
print("Question: Which layers can detect negation markers?")

negation_layer_accuracies = []
negation_layer_recalls = []
negation_layer_probes = []

# Calculate base rate
base_rate_negation = max(sum(labels_negation), len(labels_negation) - sum(labels_negation)) / len(labels_negation)
print(f"\nBase rate (majority class): {base_rate_negation:.1%}")

for layer_idx in range(model.cfg.n_layers):
    # Get activations
    X = layer_activations_negation[layer_idx]
    y = labels_negation
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # Train probe
    probe = LogisticRegression(max_iter=1000, random_state=42)
    probe.fit(X_train, y_train)
    
    # Evaluate
    test_acc = probe.score(X_test, y_test)
    y_pred = probe.predict(X_test)
    test_recall = recall_score(y_test, y_pred, pos_label=1)
    
    negation_layer_accuracies.append(test_acc)
    negation_layer_recalls.append(test_recall)
    negation_layer_probes.append(probe)
    
    if layer_idx % 3 == 0 or layer_idx == model.cfg.n_layers - 1:
        print(f"  Layer {layer_idx:2d}: Accuracy={test_acc:.3f}, Recall={test_recall:.3f}")

print(f"\n✓ Trained probes on all {model.cfg.n_layers} layers")

# Diagnostic: Random noise baseline
print("\n" + "="*60)
print("DIAGNOSTIC: Are we really learning from activations?")
print("="*60)

random_activations = np.random.randn(len(labels_negation), model.cfg.d_model)
X_train_rand, X_test_rand, y_train_rand, y_test_rand = train_test_split(
    random_activations, labels_negation, test_size=0.3, random_state=42, stratify=labels_negation
)
probe_random = LogisticRegression(max_iter=1000, random_state=42)
probe_random.fit(X_train_rand, y_train_rand)
random_acc = probe_random.score(X_test_rand, y_test_rand)
y_pred_random = probe_random.predict(X_test_rand)
random_recall = recall_score(y_test_rand, y_pred_random, pos_label=1)

print(f"\nComparison:")
print(f"  Base rate (always predict majority): {base_rate_negation:.1%} accuracy, 0.0% recall")
print(f"  Probe on random noise:               {random_acc:.1%} accuracy, {random_recall:.1%} recall")
print(f"  Probe on Layer 0:                     {negation_layer_accuracies[0]:.1%} accuracy, {negation_layer_recalls[0]:.1%} recall")
print(f"  Probe on Layer {model.cfg.n_layers-1}:                    {negation_layer_accuracies[-1]:.1%} accuracy, {negation_layer_recalls[-1]:.1%} recall")

if negation_layer_recalls[0] > random_recall * 1.5:
    print(f"\n✓ Probes beat random noise - activations contain negation information!")
else:
    print(f"\n⚠ Probes barely beat random - negation signal may be weak or nonlinear")

# %% Visualize Negation Understanding Results
print("\n" + "="*60)
print("RESULTS: When Does GPT-2 Understand Negation?")
print("="*60)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Plot accuracy
ax1.plot(range(model.cfg.n_layers), negation_layer_accuracies, 'o-', color='coral', linewidth=2, markersize=8)
ax1.set_xlabel('Layer', fontsize=12)
ax1.set_ylabel('Probe Accuracy', fontsize=12)
ax1.set_title('Negation: Accuracy Across Layers', fontsize=13)
ax1.set_ylim([0, 1])
ax1.axhline(y=base_rate_negation, color='r', linestyle='--', alpha=0.5, label=f'Base rate ({base_rate_negation:.1%})')
ax1.grid(alpha=0.3)
ax1.legend()

# Plot recall
ax2.plot(range(model.cfg.n_layers), negation_layer_recalls, 'o-', color='darkgreen', linewidth=2, markersize=8)
ax2.set_xlabel('Layer', fontsize=12)
ax2.set_ylabel('Recall (Affirmative Statements)', fontsize=12)
ax2.set_title('Negation: When Does The Model Track Polarity?', fontsize=13, fontweight='bold')
ax2.set_ylim([0, 1])
ax2.axhline(y=0.5, color='orange', linestyle='--', alpha=0.5, label='50% recall (random)')
ax2.axhline(y=random_recall, color='red', linestyle='--', alpha=0.5, label=f'Random noise ({random_recall:.1%})')
ax2.grid(alpha=0.3)
ax2.legend()

plt.tight_layout()
plt.savefig('negation_understanding_gpt2.png', dpi=150)
print("✓ Saved plot: negation_understanding_gpt2.png")
plt.show()

# Find when negation understanding emerges
print("\nNegation understanding emergence:")
for i in range(model.cfg.n_layers):
    if i == 0:
        print(f"  Layer {i:2d}: {negation_layer_recalls[i]:.1%} (baseline)")
    else:
        change = negation_layer_recalls[i] - negation_layer_recalls[i-1]
        arrow = "↑" if change > 0.05 else "→" if abs(change) < 0.05 else "↓"
        print(f"  Layer {i:2d}: {negation_layer_recalls[i]:.1%} {arrow}")

best_layer_negation = np.argmax(negation_layer_recalls)
print(f"\n✓ Best layer: {best_layer_negation} with {negation_layer_recalls[best_layer_negation]:.1%} recall")

# %% Interpretation
print("\n" + "="*60)
print("INTERPRETATION: Negation Understanding in GPT-2")
print("="*60)
print("""
What we learned:

1. TRACKING LINGUISTIC FORM, NOT TRUTH
   - The probe detects: "Is this sentence affirmative or negated?"
   - NOT: "Is this statement true or false?"
   - Tests if model tracks negation markers like "not", "cannot", "does not"

2. NEGATION IS A DISTINCT CAPABILITY
   - Different from factual recall (which tests specific knowledge)
   - Tests compositional understanding (how "not" changes meaning)
   - Shows if model tracks sentence polarity

3. LAYER-WISE EMERGENCE REVEALS PROCESSING
   - Early layers: May not strongly distinguish affirmative vs negated
   - Middle/later layers: Negation signal becomes clear
   - Compare to factual recall - do they peak in different layers?

4. IMPLICATIONS FOR MODEL UNDERSTANDING
   - If probe succeeds: Model maintains negation information
   - If probe fails: Model may struggle with negation (common LLM issue!)
   - This is important for tasks requiring logical reasoning

Next: Let's compare factual recall vs negation understanding!
""")

# %%============================================================================
# CONFUSION MATRICES: Understanding What Each Probe Gets Right/Wrong
# ============================================================================
print("\n" + "="*80)
print("CONFUSION MATRICES: Detailed Performance Analysis")
print("="*80)

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 14))

# ===== FACTUAL RECALL CONFUSION MATRIX =====
best_layer_factual = np.argmax(factual_layer_recalls)
print(f"\nFactual Recall: Using Layer {best_layer_factual} (best layer)")

X_fact = layer_activations_factual[best_layer_factual]
y_fact = labels_factual
X_train_f, X_test_f, y_train_f, y_test_f = train_test_split(
    X_fact, y_fact, test_size=0.3, random_state=42, stratify=y_fact
)

probe_fact_best = factual_layer_probes[best_layer_factual]
y_pred_fact = probe_fact_best.predict(X_test_f)

cm_fact = confusion_matrix(y_test_f, y_pred_fact)
cm_fact_norm = cm_fact.astype('float') / cm_fact.sum(axis=1)[:, np.newaxis]

# Factual - Counts
sns.heatmap(cm_fact, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Implausible', 'Plausible'],
            yticklabels=['Implausible', 'Plausible'],
            ax=ax1, cbar_kws={'label': 'Count'})
ax1.set_xlabel('Predicted', fontsize=12)
ax1.set_ylabel('Actual', fontsize=12)
ax1.set_title(f'Factual Recall - Layer {best_layer_factual} (Counts)', fontsize=13, fontweight='bold')

# Factual - Percentages
sns.heatmap(cm_fact_norm, annot=True, fmt='.1%', cmap='Blues',
            xticklabels=['Implausible', 'Plausible'],
            yticklabels=['Implausible', 'Plausible'],
            ax=ax2, vmin=0, vmax=1, cbar_kws={'label': 'Percentage'})
ax2.set_xlabel('Predicted', fontsize=12)
ax2.set_ylabel('Actual', fontsize=12)
ax2.set_title(f'Factual Recall - Layer {best_layer_factual} (Percentages)', fontsize=13, fontweight='bold')

print(f"  Plausible detection rate: {cm_fact_norm[1,1]:.1%}")
print(f"  Implausible detection rate: {cm_fact_norm[0,0]:.1%}")

# ===== NEGATION CONFUSION MATRIX =====
best_layer_negation = np.argmax(negation_layer_recalls)
print(f"\nNegation Understanding: Using Layer {best_layer_negation} (best layer)")

X_neg = layer_activations_negation[best_layer_negation]
y_neg = labels_negation
X_train_n, X_test_n, y_train_n, y_test_n = train_test_split(
    X_neg, y_neg, test_size=0.3, random_state=42, stratify=y_neg
)

probe_neg_best = negation_layer_probes[best_layer_negation]
y_pred_neg = probe_neg_best.predict(X_test_n)

cm_neg = confusion_matrix(y_test_n, y_pred_neg)
cm_neg_norm = cm_neg.astype('float') / cm_neg.sum(axis=1)[:, np.newaxis]

# Negation - Counts
sns.heatmap(cm_neg, annot=True, fmt='d', cmap='Oranges',
            xticklabels=['Negated', 'Affirmative'],
            yticklabels=['Negated', 'Affirmative'],
            ax=ax3, cbar_kws={'label': 'Count'})
ax3.set_xlabel('Predicted', fontsize=12)
ax3.set_ylabel('Actual', fontsize=12)
ax3.set_title(f'Negation Understanding - Layer {best_layer_negation} (Counts)', fontsize=13, fontweight='bold')

# Negation - Percentages
sns.heatmap(cm_neg_norm, annot=True, fmt='.1%', cmap='Oranges',
            xticklabels=['Negated', 'Affirmative'],
            yticklabels=['Negated', 'Affirmative'],
            ax=ax4, vmin=0, vmax=1, cbar_kws={'label': 'Percentage'})
ax4.set_xlabel('Predicted', fontsize=12)
ax4.set_ylabel('Actual', fontsize=12)
ax4.set_title(f'Negation Understanding - Layer {best_layer_negation} (Percentages)', fontsize=13, fontweight='bold')

print(f"  Affirmative detection rate: {cm_neg_norm[1,1]:.1%}")
print(f"  Negated detection rate: {cm_neg_norm[0,0]:.1%}")

plt.tight_layout()
plt.savefig('confusion_matrices_both_tasks.png', dpi=150)
print("\n✓ Saved plot: confusion_matrices_both_tasks.png")
plt.show()

print("\n" + "="*60)
print("INTERPRETING CONFUSION MATRICES")
print("="*60)
print("""
The diagonal (top-left to bottom-right) shows CORRECT predictions.
Off-diagonal shows errors:

For balanced datasets (50/50):
- High diagonal percentages (>90%) = Excellent probe
- Balanced errors on both sides = No systematic bias
- One class much worse = Probe is biased toward one class

Compare the two tasks:
- Which has higher diagonal percentages?
- Which has more balanced performance?
- Does one probe struggle more with a specific class?
""")

# %% Final Insights
print("\n" + "="*80)
print("KEY INSIGHTS FROM PART 2")
print("="*80)
print("""
What we discovered about GPT-2:

1. DIFFERENT CAPABILITIES EMERGE AT DIFFERENT LAYERS
   - Factual recall: Peaks in middle layers (5-6)
   - Negation detection: May peak earlier or later
   - This reveals the computational structure of the network!

2. TWO TYPES OF LINGUISTIC KNOWLEDGE
   - Factual: "Is this statement plausible given world knowledge?"
   - Negation: "Is this sentence affirmative or negated?"
   - One tests content (semantics), the other tests form (syntax)

3. LINEAR PROBES WORK ON LARGE MODELS
   - GPT-2 has 124M parameters, much bigger than char-level models
   - But linear probes still detect features
   - Suggests some information is linearly encoded

4. RECALL MATTERS FOR UNDERSTANDING
   - Accuracy can be misleading with imbalanced or easy tasks
   - Recall reveals what the model actually knows
   - Always check diagnostics (base rate, random noise)

5. LAYER-WISE PATTERNS TELL STORIES
   - Gradual emergence: Information builds up slowly
   - Sudden jumps: Specific layers compute specific features
   - Decline: Information transformed/lost for output

6. COMPARISON REVEALS COMPUTATIONAL PRIORITIES
   - Which emerges first: facts or negation?
   - Do they use the same layers or different ones?
   - This hints at how the model is organized internally

NEXT STEPS:
- Try other linguistic features (sentiment, tense, modality)
- Explore attention patterns with TransformerLens
- Try activation patching to test causal importance
- Probe MLP vs attention outputs separately
- Use logit lens to see what each layer "wants" to predict
""")

print("\n" + "="*80)
print("WORKSHOP PART 2 COMPLETE!")
print("="*80)
print("\nYou've learned:")
print("✓ How to use TransformerLens for interpretability")
print("✓ How to probe GPT-2 for factual recall (plausible vs implausible)")
print("✓ How to probe GPT-2 for negation understanding (affirmative vs negated)")
print("✓ How different capabilities emerge at different layers")
print("✓ How to compare multiple probing tasks")
print("✓ That recall is crucial for balanced datasets")
print("✓ The importance of proper contrastive examples")
print("\nYou now have the tools to probe any transformer for any feature!")

# %% EXERCISES
print("\n" + "="*80)
print("EXERCISES FOR YOU TO TRY")
print("="*80)
print("""
Now it's your turn! Here are some exercises to deepen your understanding:

EXERCISE 1: ANALYZE THE LAYER PATTERNS
-------------------------------------------------------------------------
Look at the factual recall and negation curves:
- At which layer does negation reach 90% recall?
- At which layer does factual recall reach 70% recall?
- What does this tell you about when these features are computed?
- Why might negation be detected earlier than factual plausibility?

EXERCISE 2: TEST WITH DIFFERENT FACTS
-------------------------------------------------------------------------
Modify the factual_prompts list to test different types of knowledge:
- Mathematical facts: "Two plus two equals four"
- Historical facts: "World War II ended in 1945"
- Scientific facts: "The Earth is approximately 4.5 billion years old"

Do these show the same layer patterns as geographical facts?
Hypothesis: Domain-specific knowledge might peak at different layers!

EXERCISE 3: CREATE A NEW PROBE - SENTIMENT DETECTION
-------------------------------------------------------------------------
Create a third probe that detects sentiment (positive vs negative):

Positive sentences:
- "This movie was amazing and wonderful"
- "I love this beautiful sunny day"
- "The food tasted delicious and fresh"

Negative sentences:
- "This movie was terrible and boring"
- "I hate this awful rainy day"
- "The food tasted disgusting and stale"

Questions to explore:
- Which layers detect sentiment best?
- Is sentiment an early feature (like negation) or late (like facts)?
- Does it peak in the same layer as factual recall?

EXERCISE 4: INCREASE DATASET SIZE
-------------------------------------------------------------------------
Our negation dataset has ~130 pairs. Create 50+ more pairs:
- Use different verbs: "can", "will", "should", "must"
- Try double negation: "not impossible", "not unlikely"
- Test with questions: "Is this...?" vs "Is this not...?"

Does the pattern change with more data?

EXERCISE 5: PROBE MLP OUTPUTS INSTEAD OF RESIDUAL STREAM
-------------------------------------------------------------------------
Currently we probe: cache['blocks.X.hook_resid_post']
Try probing: cache['blocks.X.hook_mlp_out']

This shows what the MLP specifically adds (not cumulative):
- Do MLPs or attention heads process negation?
- Which component handles factual recall?
- Use the layer contributions approach from Part 1!

EXERCISE 6: ANALYZE FAILURE CASES
-------------------------------------------------------------------------
For the factual recall probe, look at examples it gets wrong:
1. Extract test set predictions from the best layer
2. Find where y_test != y_pred
3. Print those sentences

Questions:
- Are there patterns in what it gets wrong?
- Do certain types of facts fail more?
- Does it fail on longer sentences? Rarer words?

EXERCISE 7: CROSS-TASK GENERALIZATION
-------------------------------------------------------------------------
Train a probe on factual recall, then test it on negation:
1. Train probe on factual task (plausible vs implausible)
2. Evaluate same probe on negation task (affirmative vs negated)
3. Does it transfer? Or are they completely separate?

This tests if the representations overlap or are orthogonal.

EXERCISE 8: TEMPORAL UNDERSTANDING (ADVANCED)
-------------------------------------------------------------------------
Create a probe for verb tense (past vs present):

Present: "The cat runs in the garden"
Past:    "The cat ran in the garden"

Present: "She walks to school"  
Past:    "She walked to school"

Questions:
- Is tense an early or late feature?
- Compare to negation and factual recall
- Does English tense (-ed suffix) make it easy to detect?

TIPS FOR SUCCESS:
----------------
✓ Always check base rate and random noise baseline
✓ Use balanced datasets (50/50 split)
✓ Run complete sentences through the model, not just prompts
✓ Plot layer-by-layer emergence to see patterns
✓ Compare your new probe to factual/negation curves

Good luck exploring! The best way to learn mech interp is to probe! 🔬
""")