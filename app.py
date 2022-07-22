from flask import Flask, render_template, request, url_for, redirect
import pandas as pd
from scrapper import extracter, maxlikes, wordcount
from lda import lda_graph

app = Flask(__name__)

@app.route("/")
def index():
    return render_template('index.html')

@app.route("/result", methods=['POST', 'GET'])
def result():
    if request.method == 'POST':
        profile_url = str(request.form['profile_url'])[25:]
        page_no = request.form['post_no']

        extracter(profile_url, page_no)
        post = maxlikes()

        post_no = list(post.index)[0]
        post_id = list(post['Post ID'])[0]
        post_caption = list(post['Captions'])[0]
        post_likes = list(post['Likes'])[0]

        wordcount()

        df = pd.read_csv('scrapped_data.csv')
        lda_graph(df)

        context = {
            'profile_url': profile_url,
            'page_no': page_no,
            'post_no': post_no,
            'post_id': post_id,
            'post_caption': post_caption,
            'post_likes': post_likes
        }

        return render_template("result.html", **context)
    return redirect(url_for('index'))

@app.route("/topics")
def topics():
    return render_template("topics.html")

if __name__ == "__main__":
    app.run(debug=True)