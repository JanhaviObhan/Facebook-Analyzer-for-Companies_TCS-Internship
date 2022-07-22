from csv import writer
from facebook_scraper import get_posts
import pandas as pd
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from lda import lda_graph

def extracter(name, page_no):
    count = 0
    filename = 'scrapped_data.csv'
    with open(filename, 'w', encoding='utf8', newline='') as f:
        thewriter = writer(f)
        header = ['Post ID', 'Captions', 'Likes']
        thewriter.writerow(header)

        for post in get_posts(name, options={"posts_per_page": page_no}, cookies="facebook.com_cookies.txt"):

            if count == int(page_no):
                break
            else:
                postid = post['post_id']
                caption = post['text']
                like = post['likes']
            
                info = [postid, caption, like]
                thewriter.writerow(info)
            
            print(count)
            count += 1
    return

def maxlikes():
    df=pd.read_csv('scrapped_data.csv')
    p=df['Likes'].max()
    post=df.loc[df['Likes'] == p]
    print(post)

    return post

def wordcount():
    data = pd.read_csv(r"scrapped_data.csv", encoding ="latin-1")
    # Iterating through the .csv data file 
    comment_words=""

    stop_words = stopwords.words('english')
    l = ('https','bitly','http','com','in')
    for item in l:
        stop_words.append(item)
    

    for i in data['Captions']: 
        i = str(i) 
        separate = i.split() 
        for j in range(len(separate)): 
            separate[j] = separate[j].lower() 
        
        comment_words += " ".join(separate)+" "
    # Creating the Word Cloud
    final_wordcloud = WordCloud(width = 750, height = 500, 
                    background_color ='white', 
                    stopwords = stop_words, 
                    min_font_size = 7).generate(comment_words)
    # Displaying the WordCloud                    
    plt.figure(figsize = (8.25, 6), facecolor = None) 
    plt.imshow(final_wordcloud) 
    plt.axis("off") 
    plt.tight_layout(pad = 0) 
    
    plt.savefig('./static/images/wordcloud.png')
    plt.close()

    return
