import json
import re
from vocabulary import vocabulary

flat_list = []
for words in vocabulary.values():
    flat_list.extend(words)

vocabulary = flat_list

with open('response.json', 'r') as file:
    story_list = json.load(file)

def preprocess_string(s):
    s = re.sub(r'\*\*.*?\*\*', '', s)
    s = s.lower()
    s = s.replace('\n', '')
    s = s.replace('\"', '')
    s = s.replace('  ', ' ')
    s = s.replace(' me ', ' i ')
    
    # Ensure whitespace after punctuation
    s = re.sub(r'([,.!?])([^\s])', r'\1 \2', s)
    return s.strip()

processed_string_list = [preprocess_string(s) for s in story_list]

def validate_string(s, allowed_words):
    s = re.sub(r'\*\*.*?\*\*', '', s)
    s = s.lower()
    s = s.replace('\n', '')
    s = s.replace('\\"', '')
    words = re.findall(r'\b\w+\b', s)
    invalid_words = [word for word in words if word not in allowed_words]
    return invalid_words

for i, s in enumerate(processed_string_list):
    invalid_words = validate_string(s, vocabulary)
    if invalid_words:
        print(f"String {i} contains invalid words: {invalid_words}")
    else:
        print(f"String {i} is valid.")

valid_strings = [s for s in processed_string_list if not validate_string(s, vocabulary)]

def ensure_punctuation_space(s):
    # Final pass to ensure whitespace after punctuation before writing
    return re.sub(r'([,.!?])([^\s])', r'\1 \2', s)

with open('valid_strings.txt', 'w') as file:
    for string in valid_strings:
        # Apply the punctuation space check one more time before writing
        story = ensure_punctuation_space(string)
        story = story.replace('  ', ' ')
        file.write(story)