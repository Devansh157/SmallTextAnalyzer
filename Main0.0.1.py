import re
from collections import Counter
from transformers import pipeline
from keybert import KeyBERT
import nltk

#download NLTK stopwords
nltk.download('stopwords')
from nltk.corpus import stopwords

#Load AI models
sentiment_analyzer = pipeline("sentiment-analysis")
kw_model = KeyBERT()
stop_words = set(stopwords.words('english'))

#Remove date time and sender from Whatsapp chat
def clean_chat_line(line):
    #Remove dd/mm//yyyy, hh:mm am/pm - Name
    msg = re.sub(r'^\d{1,2}/\d{1,2}/\d{2,4},?\s*\d{1,2}:\d{2}\s*(am|pm)?\s*-\s*[^:]+:\s*','',line)
    return msg.strip()
    
#Analyze a whatsapp chat file
def ai_chat_analyzer(file_path):
    with open(file_path,"r",encoding = "UTF-8") as f:
        lines = f.readlines()
    
    messages = [clean_chat_line(line) for line in lines if ':' in line]
    
    num_photos = sum(1 for m in messages if "[photo]" in m.lower())
    num_videos = sum(1 for m in messages if "[video]" in m.lower())
    num_media = sum(1 for m in messages if "<media omitted>" in m.lower())
    
    messages = [m for m in messages if "[photo]" not in m.lower() and "[video]" not in m.lower() and "<media omitted>" not in m.lower()]
    
    chat_text = "".join(messages)
    
    #Sentiment
    sentiment = sentiment_analyzer(chat_text[:512])[0]
    
    #Keywords/Interests
    keywords = kw_model.extract_keywords(chat_text,top_n=8)
    keywords = [k[0] for k in keywords]
    
    #Word Frequency
    words = re.findall(r'\b[a-z]+\b',chat_text.lower())
    words = [w for w in words if w not in stop_words]
    top_words = Counter(words).most_common(10)
    
    
    #Average message length
    avg_length = sum(len(m.split()) for m in messages)/ len(messages)
    
    #Results
    print("\n AI chat Analysis Report")
    print(f"Messages analyzed: {len(messages)}")
    print(f"Overall sentiment: {sentiment['label']}({sentiment['score']:.2f})")
    print(f"Average message length : {avg_length:.1f} words\n")
    print("Top interests (AI-detected):")
    print(", ".join(keywords), "\n")
    
    print("Most common words: ")
    for w,c in top_words:
        print(f"{w}: {c}")
        
ai_chat_analyzer("D:\project\Chat AI\chat.txt")