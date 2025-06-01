from flask import Flask, render_template, request
from transformers import pipeline

app = Flask(__name__)

# Load the summarization pipeline
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

@app.route('/', methods=['GET', 'POST'])
def index():
    summary = ""
    if request.method == 'POST':
        input_text = request.form['message']
        if input_text.strip():
            # Limit to 1024 characters (to fit BART model input limits)
            truncated_text = input_text[:1024]
            result = summarizer(truncated_text, max_length=100, min_length=25, do_sample=False)
            summary = result[0]['summary_text']
    return render_template('index.html', summary=summary)

if __name__ == '__main__':
    app.run(debug=True)
