import nltk

def download_nltk_resources():
    """Download required NLTK resources"""
    print("Downloading NLTK resources...")
    
    # Try both package names
    try:
        nltk.download('punkt_tab')
        print("Successfully downloaded punkt_tab")
    except:
        print("punkt_tab not found in NLTK repository, downloading punkt instead")
        nltk.download('punkt')
    
    # Additional useful resources
    resources = [
        'stopwords',
        'wordnet',
        'averaged_perceptron_tagger'
    ]
    
    for resource in resources:
        print(f"Downloading {resource}...")
        nltk.download(resource)
        
    print("NLTK resources downloaded successfully!")

if __name__ == "__main__":
    download_nltk_resources()
    