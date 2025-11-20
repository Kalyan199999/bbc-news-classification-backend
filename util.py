import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

# Example setup
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Pattern: match special characters (except letters/numbers)
pattern = r'[^a-zA-Z0-9\s]'

def preprocessing(text):
    
    # Replace special characters with a space
    text = re.sub(pattern=pattern, repl=' ', string=text)

    # Tokenize
    words = word_tokenize(text)

    # Remove stopwords and lemmatize
    lemma_words = [lemmatizer.lemmatize(word) for word in words if word.lower() not in stop_words]

    # Join back into string
    res = ' '.join(lemma_words)

    return res
