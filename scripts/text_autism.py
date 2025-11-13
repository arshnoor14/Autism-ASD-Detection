import pandas as pd
import random
from faker import Faker
import numpy as np

fake = Faker()

# Expanded autism indicators with more natural phrasing and variations
autism_phrases = [
    # Social difficulties
    ("I often struggle with eye contact", 0.9),
    ("Making friends feels challenging sometimes", 0.8),
    ("I sometimes miss social cues", 0.7),
    ("Group settings make me anxious", 0.85),
    ("I prefer quiet time alone", 0.75),
    
    # Sensory issues
    ("Loud noises bother me", 0.9),
    ("Some clothing textures feel uncomfortable", 0.8),
    ("Bright lights can be painful", 0.85),
    ("Strong smells sometimes make me nauseous", 0.7),
    ("Certain sounds really irritate me", 0.8),
    
    # Routine needs
    ("I prefer sticking to my routines", 0.9),
    ("Changes to my schedule upset me", 0.85),
    ("I like knowing what to expect", 0.8),
    ("I eat similar foods regularly", 0.7),
    ("I organize my things carefully", 0.75),
    
    # Communication
    ("People say I'm very literal", 0.8),
    ("I sometimes don't get jokes right away", 0.7),
    ("I really enjoy talking about my interests", 0.75),
    ("Conversations can be confusing", 0.7),
    ("I don't always read facial expressions well", 0.8)
]

neutral_phrases = [
    ("I enjoy socializing with people", 0.1),
    ("I'm okay with last-minute changes", 0.2),
    ("Concerts and parties are fun", 0.15),
    ("I like trying new restaurants", 0.2),
    ("I adapt well to new situations", 0.15),
    ("Meeting new people excites me", 0.1),
    ("Eye contact feels natural", 0.1),
    ("I usually understand sarcasm", 0.2),
    ("Different textures don't bother me", 0.15),
    ("I'm flexible with my schedule", 0.2)
]

def add_variations(text):
    """Add natural language variations"""
    variations = [
        "",
        " " + fake.sentence(),
        " I've noticed this for a while.",
        " It depends on my mood.",
        " My " + random.choice(["friend", "family member", "colleague"]) + " says this is normal.",
        " I'm working on this."
    ]
    return text + random.choice(variations)

def add_typos(text, prob=0.1):
    """Add occasional typos or informal language"""
    if random.random() < prob:
        text = text.replace('ing ', 'in\' ')
        text = text.replace('some', 'sum')
        text = text.replace('things', 'thingz')
    return text

def generate_example(is_autism):
    """Generate a single example with more realistic variations"""
    if is_autism:
        # Select phrases weighted by their autism probability
        phrases, weights = zip(*autism_phrases)
        selected = random.choices(phrases, weights=weights, k=random.randint(1, 3))
        text = ". ".join(selected) + "."
    else:
        # Select neutral phrases with inverse weights
        phrases, weights = zip(*neutral_phrases)
        selected = random.choices(phrases, weights=[1-w for w in weights], k=random.randint(1, 2))
        text = ". ".join(selected) + "."
    
    # Add variations and imperfections
    text = add_variations(text)
    text = add_typos(text)
    
    # Occasionally make the text less clear
    if random.random() < 0.2:
        text = text.lower()
        if random.random() < 0.3:
            text = text.replace(".", "...")
    
    return text.capitalize()

# Generate more realistic dataset with some overlap
data = []
n_samples = 5000  # Total samples

# Generate clear examples
for _ in range(int(n_samples * 0.8)):
    is_autism = random.random() > 0.5
    data.append([generate_example(is_autism), int(is_autism)])

# Generate ambiguous examples (10%)
for _ in range(int(n_samples * 0.1)):
    # Autism phrases with neutral label
    text = generate_example(is_autism=True)
    data.append([text, 0])
    
    # Neutral phrases with autism label
    text = generate_example(is_autism=False)
    data.append([text, 1])

# Shuffle the dataset
random.shuffle(data)

# Create DataFrame and save
df = pd.DataFrame(data, columns=["text", "Autism_Diagnosis"])
df.to_csv("data/txt_autism_dataset_enhanced.csv", index=False)
print(f"Dataset generated with {len(df)} samples ({(df['Autism_Diagnosis'].mean()*100):.1f}% autism examples)")