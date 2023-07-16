from flask import Flask, request, render_template
import joblib
app = Flask(__name__)
model = joblib.load(open("fraud_pickle.pkl", "rb"))
tf = joblib.load(open('tfidf_job_pickle.pkl', 'rb'))


@app.route("/")
def home():
    return render_template("Fake_Job_Home.html")

def transform_text(text):
    input_data = [text]
    vectorized_input_data = tf.transform(input_data)
    prediction = model.predict(vectorized_input_data)
    return prediction

@app.route("/predict", methods=["POST"])
def predict():
    if request.method == "POST":
        text = request.form['text']
        prediction= transform_text(text)
        return render_template('Fake_Job_Home.html', prediction=prediction)


if __name__ == "__main__":
    app.run(debug=True)
