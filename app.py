from flask import Flask, render_template, request, redirect, url_for, session, jsonify
from dotenv import load_dotenv
from google import genai
import os
import json
import uuid

load_dotenv()


# =========================================================
# GEMINI SETUP
# =========================================================

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if GEMINI_API_KEY:
    print("Gemini API key loaded successfully.")
    client = genai.Client(api_key=GEMINI_API_KEY)
else:
    print("WARNING: GEMINI_API_KEY not found.")
    client = None


# =========================================================
# FLASK SETUP
# =========================================================

app = Flask(__name__)

app.secret_key = "ai-road-guardian-secret-key"

UPLOAD_FOLDER = "static/uploads"

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# =========================================================
# COMPLAINT DATA
# =========================================================

complaints = {
    "RG-1001": {
        "location": "Main Market Road",
        "description": "Large pothole on the road",
        "priority": "High",
        "status": "Pending"
    }
}


# =========================================================
# HOME
# =========================================================

@app.route("/")
def home():
    return render_template("index.html")


# =========================================================
# REPORT PAGE
# =========================================================

@app.route("/report")
def report():
    return render_template("report.html")


# =========================================================
# GEMINI ROAD IMAGE ANALYSIS
# =========================================================

def analyze_road_image(image_path):

    if client is None:

        return {
            "issue_type": "AI unavailable",
            "estimated_depth": "Not available",
            "priority": "Medium",
            "possible_cause": "Gemini API key is not configured.",
            "recommendation": "Please configure the Gemini API key."
        }

    print()
    print("========================================")
    print("AI ROAD GUARDIAN")
    print("Gemini 3.5 Flash is analyzing the image...")
    print("========================================")
    print()

    try:

        # Upload image to Gemini
        uploaded_file = client.files.upload(
            file=image_path
        )

        # AI prompt
        prompt = """
You are an expert AI road-condition assessment assistant
for a system called "AI Road Guardian".

Carefully inspect the uploaded road image.

Analyze ONLY what is visually supported by the image.

Your task is to help government road authorities
understand the road problem and decide what action
may be required.

Analyze these five things:

1. ISSUE TYPE

Identify the most likely visible road problem.

Possible examples:

- Pothole
- Road crack
- Surface deterioration
- Waterlogging
- Broken road edge
- Uneven pavement
- Damaged manhole area
- Road debris
- No obvious road damage

2. ESTIMATED DEPTH

Estimate the approximate depth ONLY when the image
provides reasonable visual evidence or a reference.

If depth cannot be reliably determined, write:

"Not reliably measurable from image."

NEVER invent a numerical measurement.

3. PRIORITY LEVEL

Choose exactly ONE:

Low
Medium
High

Base the priority on visible factors such as:

- apparent size of damage
- location on the road
- danger to vehicles
- danger to pedestrians
- obstruction of traffic
- possibility of damage worsening
- visible water accumulation
- visible structural deterioration

Do NOT provide a separate Severity field.

4. POSSIBLE CAUSE

Give the most plausible cause based on visible evidence.

Possible causes include:

- Water infiltration
- Poor drainage
- Pavement wear
- Heavy traffic
- Weather damage
- Poor maintenance
- Road-edge failure

If the cause cannot be determined from the image,
say that physical inspection is required.

Do not present speculation as confirmed fact.

5. RECOMMENDATION

Give a practical recommendation for road authorities.

Consider:

- immediate safety measures
- inspection
- temporary repair
- permanent repair
- drainage correction
- resurfacing

The recommendation should be realistic and concise.

IMPORTANT RULES:

- Do not invent facts.
- Do not invent measurements.
- Do not claim certainty when the image is unclear.
- Do not analyze anything unrelated to road conditions.
- Do not include a Severity field.
- Keep the response concise and useful for a government officer.
- Return ONLY valid JSON.
- Do not use Markdown.
- Do not put JSON inside code fences.

Return exactly this structure:

{
    "issue_type": "",
    "estimated_depth": "",
    "priority": "",
    "possible_cause": "",
    "recommendation": ""
}
"""

        # Gemini request
        response = client.models.generate_content(

            model="gemini-3.5-flash",

            contents=[
                uploaded_file,
                prompt
            ]
        )

        # Get AI response
        result = response.text.strip()

        print("Gemini response received.")
        print(result)
        print()

        # Remove markdown fences if Gemini adds them
        result = result.replace("```json", "")
        result = result.replace("```", "")
        result = result.strip()

        # Convert JSON to Python dictionary
        ai_result = json.loads(result)

        # Make sure all required fields exist
        ai_result = {

            "issue_type": ai_result.get(
                "issue_type",
                "Not identified"
            ),

            "estimated_depth": ai_result.get(
                "estimated_depth",
                "Not reliably measurable from image."
            ),

            "priority": ai_result.get(
                "priority",
                "Medium"
            ),

            "possible_cause": ai_result.get(
                "possible_cause",
                "Could not be determined reliably."
            ),

            "recommendation": ai_result.get(
                "recommendation",
                "Physical inspection recommended."
            )
        }

        # Normalize priority
        priority = str(
            ai_result["priority"]
        ).strip().capitalize()

        if priority not in ["Low", "Medium", "High"]:
            priority = "Medium"

        ai_result["priority"] = priority

        print("Structured AI result:")
        print(ai_result)
        print()

        return ai_result

    except json.JSONDecodeError:

        print("Gemini returned invalid JSON.")

        return {
            "issue_type": "Road damage detected",
            "estimated_depth": "Not reliably measurable from image.",
            "priority": "Medium",
            "possible_cause": "Physical inspection is required.",
            "recommendation": "A road authority engineer should inspect the reported location."
        }

    except Exception as error:

        print("Gemini analysis error:")
        print(error)

        return {
            "issue_type": "AI analysis unavailable",
            "estimated_depth": "Not available",
            "priority": "Medium",
            "possible_cause": "AI analysis could not be completed.",
            "recommendation": "Please retry the report or arrange a physical inspection."
        }


# =========================================================
# SUBMIT REPORT
# =========================================================

@app.route("/submit-report", methods=["POST"])
def submit_report():

    print()
    print("========================================")
    print("New road report received.")
    print("========================================")

    # Receive uploaded photo
    photo = request.files.get("photo")

    if not photo or not photo.filename:

        print("ERROR: No photo uploaded.")

        return "No photo uploaded", 400

    print("Photo received:", photo.filename)

    # Get extension
    extension = os.path.splitext(
        photo.filename
    )[1].lower()

    # Allow common image formats
    if extension not in [
        ".jpg",
        ".jpeg",
        ".png",
        ".webp"
    ]:

        return (
            "Unsupported image format. "
            "Please upload JPG, JPEG, PNG, or WEBP.",
            400
        )

    # Create unique filename
    filename = (
        str(uuid.uuid4())
        + extension
    )

    photo_path = os.path.join(
        app.config["UPLOAD_FOLDER"],
        filename
    )

    # Save image
    photo.save(photo_path)

    print("Photo saved:", photo_path)

    # Send image to Gemini
    ai_result = analyze_road_image(
        photo_path
    )

    # Create complaint ID
    complaint_id = (
        f"RG-{1001 + len(complaints)}"
    )

    # Store complaint
    complaints[complaint_id] = {

        "location": request.form.get(
            "location",
            "Location not provided"
        ),

        "description": request.form.get(
            "description",
            "Road problem reported"
        ),

        "priority": ai_result["priority"],

        "status": "Pending",

        "photo": filename,

        "ai_result": ai_result
    }

    print(
        "Complaint created:",
        complaint_id
    )

    print(
        "Priority:",
        ai_result["priority"]
    )

    print("========================================")
    print()

    # Show AI analysis page
    return render_template(

        "analysis.html",

        complaint_id=complaint_id,

        ai_result=ai_result
    )


# =========================================================
# ANALYSIS PAGE
# =========================================================

@app.route("/analysis")
def analysis():

    complaint_id = request.args.get(
        "id",
        "RG-1001"
    )

    complaint = complaints.get(
        complaint_id
    )

    if complaint and "ai_result" in complaint:

        return render_template(

            "analysis.html",

            complaint_id=complaint_id,

            ai_result=complaint["ai_result"]
        )

    # Fallback for demo complaint
    ai_result = {

        "issue_type": "Pothole",

        "estimated_depth":
            "Not reliably measurable from image.",

        "priority": "High",

        "possible_cause":
            "Water damage and surface wear",

        "recommendation":
            "Physical inspection and appropriate road repair are recommended."
    }

    return render_template(

        "analysis.html",

        complaint_id=complaint_id,

        ai_result=ai_result
    )


# =========================================================
# TRACK COMPLAINT
# =========================================================

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


# =========================================================
# GOVERNMENT LOGIN
# =========================================================

@app.route("/government-login")
def government_login():

    return render_template(
        "government_login.html"
    )


# =========================================================
# GOVERNMENT AUTHENTICATION
# =========================================================

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

    if (
        username == "officer@government.in"
        and password == "road123"
    ):

        session[
            "government_logged_in"
        ] = True

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


# =========================================================
# GOVERNMENT DASHBOARD
# =========================================================

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


# =========================================================
# UPDATE STATUS
# =========================================================

@app.route(
    "/update-status/<complaint_id>",
    methods=["POST"]
)
def update_status(complaint_id):

    if complaint_id in complaints:

        complaints[
            complaint_id
        ]["status"] = request.form.get(
            "status"
        )

    return redirect(
        url_for("dashboard")
    )


# =========================================================
# LOGOUT
# =========================================================

@app.route("/logout")
def logout():

    session.pop(
        "government_logged_in",
        None
    )

    return redirect(
        "/government-login"
    )


# =========================================================
# RUN
# =========================================================

if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=5000
    )