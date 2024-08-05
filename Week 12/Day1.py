import spacy

# Load the pre-trained model
nlp = spacy.load('en_core_web_sm')

# Sample text
text = "Apple is looking at buying U.K. startup for $1 billion"

# Process the text
doc = nlp(text)

# Print the entities in the text
for entity in doc.ents:
    print(f"{entity.text} ({entity.label_})")
