from flask import Flask, render_template, request, redirect, url_for, session, jsonify
from dotenv import load_dotenv
import google.generativeai as genai
import os

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)


app = Flask(__name__)

app.secret_key = "ai-road-guardian-secret-key"


UPLOAD_FOLDER = "static/uploads"

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)



# Demo complaint data

complaints = {

    "RG-1001": {

        "location": "Main Market Road",

        "description": "Large pothole on the road",

        "priority": "High",

        "status": "Pending"

    }

}



# Home

@app.route("/")
def home():

    return render_template(
        "index.html"
    )



# Report

@app.route("/report")
def report():

    return render_template(
        "report.html"
    )



# Submit report

@app.route("/submit-report", methods=["POST"])
def submit_report():

    photo = request.files.get("photo")


    if photo and photo.filename:

        photo_path = os.path.join(
            app.config["UPLOAD_FOLDER"],
            photo.filename
        )

        photo.save(photo_path)


    return render_template(
        "analysis.html",
        complaint_id="RG-1001"
    )



# Analysis

@app.route("/analysis")
def analysis():

    return render_template(
        "analysis.html",
        complaint_id="RG-1001"
    )



# Track complaint

@app.route("/track")
def track():

    complaint_id = request.args.get(
        "id",
        "RG-1001"
    )


    complaint = complaints.get(
        complaint_id
    )


    return render_template(
        "track.html",
        complaint_id=complaint_id,
        complaint=complaint
    )



# =========================
# GOVERNMENT LOGIN
# =========================

@app.route("/government-login")
def government_login():

    return render_template(
        "government_login.html"
    )



@app.route(
    "/government-auth",
    methods=["POST"]
)
def government_auth():

    username = request.form.get(
        "username"
    )

    password = request.form.get(
        "password"
    )


    if username == "officer@government.in" and password == "road123":


        session["government_logged_in"] = True


        return jsonify({

            "success": True,

            "message":
            "Login successful! Opening dashboard..."

        })


    return jsonify({

        "success": False,

        "message":
        "Incorrect email or password."

    })



# Dashboard

@app.route("/dashboard")
def dashboard():


    if not session.get(
        "government_logged_in"
    ):

        return redirect(
            "/government-login"
        )


    return render_template(
        "dashboard.html",
        complaints=complaints
    )



# Update status

@app.route(
    "/update-status/<complaint_id>",
    methods=["POST"]
)
def update_status(complaint_id):


    if complaint_id in complaints:


        complaints[complaint_id]["status"] = request.form.get(
            "status"
        )


    return redirect(
        url_for("dashboard")
    )



# Logout

@app.route("/logout")
def logout():

    session.pop(
        "government_logged_in",
        None
    )


    return redirect(
        "/government-login"
    )



# Run

if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=5000
    )