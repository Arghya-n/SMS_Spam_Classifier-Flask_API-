from flask import Flask, request, render_template
import string
import nltk
from nltk.corpus import stopwords

nltk.download("punkt")
nltk.download('stopwords')
from nltk.stem.porter import PorterStemmer
import pickle


app = Flask(__name__)
ps = PorterStemmer()

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

tfidf=pickle.load(open('vectorizer.pkl','rb'))
model=pickle.load(open('model.pkl','rb'))


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():

    '''
    
    For rendering results on HTML GUI
    '''
    # int_features =
    # print(int_features)
    transform_sms = transform_text(request.form['msg'])
    vector_input = tfidf.transform([transform_sms])
    result = model.predict(vector_input)[0]

    if result == 1:
        output="Spam"
    else:
        output="Not Spam"
    return render_template('index.html', prediction_text=output)


if __name__ == "__main__":
    app.run(debug=True)