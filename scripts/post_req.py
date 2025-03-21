import nltk
import os

# Set custom NLTK data directory (optional)
nltk_data_dir = os.path.join(os.getcwd(), "nltk_data")
os.makedirs(nltk_data_dir, exist_ok=True)
nltk.data.path.append(nltk_data_dir)

# List of required corpora
required_corpora = ["wordnet", "omw-1.4", "punkt"]

# Download each corpus
for corpus in required_corpora:
    try:
        print(f"Checking if '{corpus}' is available...")
        nltk.data.find(f"corpora/{corpus}")
        print(f"'{corpus}' is already available.")
    except LookupError:
        print(f"Downloading '{corpus}'...")
        nltk.download(corpus, download_dir=nltk_data_dir)
