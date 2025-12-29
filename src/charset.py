# src/charset.py
# Defines full printable ASCII characters and mappings for OCR

import string

# Full printable ASCII characters (95 chars)
ASCII_CHARS = string.printable[:-6]  # remove last 6 non-printable whitespace chars

# Create mappings
char_to_idx = {c: i for i, c in enumerate(ASCII_CHARS)}
idx_to_char = {i: c for i, c in enumerate(ASCII_CHARS)}

# Number of characters (needed for model output)
NUM_CLASSES = len(ASCII_CHARS)

if __name__ == "__main__":
    print(f"Total characters: {NUM_CLASSES}")
    print("Character set:", ASCII_CHARS)
