# src/charset.py
import string

# Full printable ASCII characters
all_chars = string.printable[:-6]  # remove last 6 non-printable whitespace chars

# Illegal filename characters on Windows
ILLEGAL_CHARS = '<>:"/\\|?*_.!@#$%^&()+=`~,;\' '

# Filter out illegal characters
ASCII_CHARS = "".join(c for c in all_chars if c not in ILLEGAL_CHARS)

# Create mappings
char_to_idx = {c: i for i, c in enumerate(ASCII_CHARS)}
idx_to_char = {i: c for i, c in enumerate(ASCII_CHARS)}

# Number of characters (needed for model output)
NUM_CLASSES = len(ASCII_CHARS)

if __name__ == "__main__":
    print(f"Total characters: {NUM_CLASSES}")
    print("Character set:", ASCII_CHARS)
