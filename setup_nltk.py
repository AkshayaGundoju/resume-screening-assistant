import nltk

packages = [
    "stopwords",
    "wordnet",
    "omw-1.4",
    "punkt"
]

for package in packages:
    print(f"Downloading {package}...")
    nltk.download(package)

print("All NLTK packages downloaded successfully!")