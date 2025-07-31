from flask import Flask,render_template,request,redirect,url_for,session
import os

from src.pipeline.predect_pipeline import predictPipeline,CustomData
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY")

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["GET","POST"])
def prediction():
    if request.method == "GET":
        return render_template("home.html")
    
    else :
        data = CustomData(
            age=request.form.get("age"),
            cgpa=request.form.get("cgpa"),
            communication_skill = request.form.get("communication_skill"),
            attendance_percentage = request.form.get("attendance_percentage"),
            gender = request.form.get("gender"),
            department = request.form.get("department"),
            year = request.form.get("year"),
            internship_done = request.form.get("internship_done") 
        )

        pred_df = data.get_data_as_dataframe()
        print(pred_df)

        predict_pipeline = predictPipeline()
        results = predict_pipeline.predict(pred_df)

        print("Result:", results[0])

        if float(results) == 1.0:
            session["placed"] = True
            return redirect(url_for("successful"))
        
        else:
            session["failed"] = True
            return redirect(url_for("failure"))
        
@app.route("/success")
def successful():
    if session.get("placed"):
        session.pop("placed")
        return render_template("success.html")
        
    else:
        return redirect(url_for("prediction"))
    
@app.route("/fail")
def failure():
        if session.get("failed"):
            session.pop("failed")
            return render_template("failure.html")
        
        else:
            return redirect(url_for("prediction"))

if __name__ == "__main__":
    app.run(debug=True)