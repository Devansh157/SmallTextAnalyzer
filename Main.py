from transformers import pipeline
from keybert import KeyBERT

analyzer = pipeline("sentiment-analysis")
kw_model = KeyBERT()

def analyze_chat(text):
    
    #Get Sentiment
    sentiment = analyzer(text[:512])[0]
    
    #Get keywords
    keywords = kw_model.extract_keywords(text,top_n=5)
    keywords = [k[0] for k in keywords]
    
    print("AI chat analysis")
    print(f"Sentiment: {sentiment['label']}({sentiment['score']:2f})")
    print("Top interests:",", ".join(keywords))

analyze_chat("Hey i love playing cricket \n Harsh is chaddi and whats should i say")
