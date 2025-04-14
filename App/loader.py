import json
from nltk.tokenize import sent_tokenize, word_tokenize

def load_json(filename):  
    with open(filename, 'r') as file:
        data = json.load(file)
    return data

def word_chunk(text, chunk_size=100, overlap=20):
    words = word_tokenize(text)
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = ' '.join(words[i:i + chunk_size])
        if chunk.strip():
            chunks.append(chunk.strip())
    return chunks

def smart_chunk(text, sentence_threshold=50):
    # If text is a list, join it into a string
    if isinstance(text, list):
        text = ' '.join(text)

    # If short text, use sentence tokenization
    if len(text.split()) <= sentence_threshold:
        return sent_tokenize(text)
    else:
        # Long description – use word chunking
        return word_chunk(text, chunk_size=10, overlap=10)

def chunk_text(product):
    """this function chunks the text into smaller parts
       split the product fiel(Name, Description, Features) into smaller parts

       # Chunking Strategy:
       - use Hybrid approach: for short product text(shor description/features) use sentence based chunking(via Nltk)
       - For long or poorly punctuated text, apply overlapping word-based chunking.
       
       # Justifucation:
        - Sentence tokenization gives natural semantic units for embedding.
        - Word-chunking prevents long chunks from degrading embedding quality or getting truncated.
    """
    chunks = []
    product_id = product['product_id']
    for field in ['name', 'description', 'features','price']:
        if field in product:
            text = product[field]          
            for chunk in smart_chunk(text):
                if chunk.strip():
                    chunks.append((product_id, chunk.strip()))
    return chunks
