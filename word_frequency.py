import json
from collections import Counter
import re
from vocabulary import vocabulary

def load_vocabulary():
    # Your vocabulary structure
        return vocabulary

def flatten_vocabulary(vocab_dict):
    """Convert the nested vocabulary dictionary into a flat set of words"""
    all_words = set()
    for category in vocab_dict.values():
        all_words.update(word.lower() for word in category)
    return all_words

def extract_words(text):
    # Use regex to find all words (including apostrophes for contractions)
    words = re.findall(r"\b[\w'-]+\b", text.lower())
    return words

def analyze_non_vocabulary_words(file_path):
    try:
        # Load and flatten vocabulary
        vocabulary = load_vocabulary()
        vocab_words = flatten_vocabulary(vocabulary)
        
        # Load JSON file
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Extract all text (modify based on your JSON structure)
        all_text = ""
        if isinstance(data, dict):
            for value in data.values():
                if isinstance(value, str):
                    all_text += " " + value
        elif isinstance(data, list):
            for item in data:
                if isinstance(item, str):
                    all_text += " " + item
                elif isinstance(item, dict):
                    for value in item.values():
                        if isinstance(value, str):
                            all_text += " " + value
        
        # Get all words not in vocabulary
        words = extract_words(all_text)
        non_vocab_words = [word for word in words if word not in vocab_words]
        
        # Create frequency table
        freq_table = Counter(non_vocab_words)
        
        # Display results
        print(f"Total words analyzed: {len(words)}")
        print(f"Unique vocabulary words: {len(vocab_words)}")
        print(f"Unique non-vocabulary words found: {len(freq_table)}")
        print("\nFrequency table of non-vocabulary words (sorted by frequency):")
        
        # Sort by frequency (descending) and then alphabetically
        for word, count in sorted(freq_table.items(), key=lambda x: (-x[1], x[0])):
            print(f"{word}: {count}")
            
        return freq_table
        
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
    except json.JSONDecodeError:
        print(f"Error: File '{file_path}' is not valid JSON.")
    except Exception as e:
        print(f"An unexpected error occurred: {str(e)}")

if __name__ == "__main__":
    # Analyze the response.json file
    file_path = 'response.json'
    frequency_table = analyze_non_vocabulary_words(file_path)
    
    # Optional: Save results to a file
    with open('non_vocabulary_words.txt', 'w', encoding='utf-8') as f:
        f.write("Non-vocabulary words frequency table:\n")
        f.write("====================================\n\n")
        for word, count in sorted(frequency_table.items(), key=lambda x: (-x[1], x[0])):
            f.write(f"{word}: {count}\n")