from flask import Flask, render_template, request, redirect, url_for, session, jsonify
from dotenv import load_dotenv
from google import genai
import os
import json
import uuid
import sqlite3
import requests

# =========================================================
# ENVIRONMENT
# =========================================================

load_dotenv()


# =========================================================
# GEMINI SETUP
# =========================================================

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if GEMINI_API_KEY:

    print("Gemini API key loaded successfully.")

    client = genai.Client(
        api_key=GEMINI_API_KEY
    )

else:

    print(
        "WARNING: GEMINI_API_KEY not found."
    )

    client = None


# =========================================================
# FLASK SETUP
# =========================================================

app = Flask(__name__)

app.secret_key = (
    "ai-road-guardian-secret-key"
)


UPLOAD_FOLDER = "static/uploads"

app.config[
    "UPLOAD_FOLDER"
] = UPLOAD_FOLDER


os.makedirs(
    UPLOAD_FOLDER,
    exist_ok=True
)


# =========================================================
# DATABASE
# =========================================================

DATABASE = "road_guardian.db"


def get_db():

    connection = sqlite3.connect(
        DATABASE
    )

    connection.row_factory = (
        sqlite3.Row
    )

    return connection


# =========================================================
# DATABASE INITIALIZATION + MIGRATION
# =========================================================

def init_database():

    connection = get_db()


    # -----------------------------------------------------
    # Create table if it does not exist
    # -----------------------------------------------------

    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS complaints (

            complaint_id TEXT PRIMARY KEY,

            location TEXT,

            description TEXT,

            problem TEXT,

            priority TEXT,

            status TEXT,

            photo TEXT,

            ai_result TEXT,

            created_at TIMESTAMP
                DEFAULT CURRENT_TIMESTAMP

        )
        """
    )


    # -----------------------------------------------------
    # Check existing columns
    # -----------------------------------------------------

    columns = connection.execute(
        "PRAGMA table_info(complaints)"
    ).fetchall()


    existing_columns = {
        column["name"]
        for column in columns
    }


    # -----------------------------------------------------
    # Add missing columns
    #
    # IMPORTANT:
    # This upgrades the existing database.
    # It does NOT delete existing complaints.
    # -----------------------------------------------------

    required_columns = {

        "location":
            "TEXT",

        "description":
            "TEXT",

        "problem":
            "TEXT",

        "priority":
            "TEXT",

        "status":
            "TEXT",

        "photo":
            "TEXT",

        "ai_result":
            "TEXT",

        "created_at":
            "TIMESTAMP"

    }


    for column_name, column_type in (
        required_columns.items()
    ):

        if column_name not in existing_columns:

            connection.execute(
                f"""
                ALTER TABLE complaints
                ADD COLUMN {column_name}
                {column_type}
                """
            )

            print(
                f"Database fix: added "
                f"'{column_name}' column."
            )


    # -----------------------------------------------------
    # Give old records safe defaults
    # -----------------------------------------------------

    connection.execute(
        """
        UPDATE complaints
        SET priority = 'Medium'
        WHERE priority IS NULL
           OR priority = ''
        """
    )


    connection.execute(
        """
        UPDATE complaints
        SET status = 'Pending'
        WHERE status IS NULL
           OR status = ''
        """
    )


    connection.execute(
        """
        UPDATE complaints
        SET problem = 'Road problem reported'
        WHERE problem IS NULL
           OR problem = ''
        """
    )


    connection.execute(
        """
        UPDATE complaints
        SET description = 'Road problem reported'
        WHERE description IS NULL
           OR description = ''
        """
    )


    connection.execute(
        """
        UPDATE complaints
        SET location = 'Location not provided'
        WHERE location IS NULL
           OR location = ''
        """
    )


    connection.commit()

    connection.close()


    print(
        "Database initialization completed."
    )


# Run database initialization
init_database()


# =========================================================
# CONVERT DATABASE ROW TO DICTIONARY
# =========================================================

def row_to_complaint(row):

    if row is None:

        return None


    # -----------------------------------------------------
    # Safely read AI result
    # -----------------------------------------------------

    ai_result = {}


    try:

        raw_ai_result = row["ai_result"]


        if raw_ai_result:

            ai_result = json.loads(
                raw_ai_result
            )


            if not isinstance(
                ai_result,
                dict
            ):

                ai_result = {}


    except Exception:

        ai_result = {}


    # -----------------------------------------------------
    # Return complaint dictionary
    # -----------------------------------------------------

    return {

        "location":
            row["location"] or "",

        "description":
            row["description"] or "",

        "problem":
            row["problem"] or "",

        "priority":
            row["priority"] or "Medium",

        "status":
            row["status"] or "Pending",

        "photo":
            row["photo"] or "",

        "ai_result":
            ai_result,

        "created_at":
            row["created_at"]

    }


# =========================================================
# GET ONE COMPLAINT
# =========================================================

def get_complaint(
    complaint_id
):

    connection = get_db()


    row = connection.execute(
        """
        SELECT *
        FROM complaints
        WHERE complaint_id = ?
        """,
        (
            complaint_id,
        )
    ).fetchone()


    connection.close()


    return row_to_complaint(
        row
    )


# =========================================================
# GET ALL COMPLAINTS
# =========================================================

def get_all_complaints():

    connection = get_db()


    rows = connection.execute(
        """
        SELECT *
        FROM complaints
        ORDER BY created_at ASC
        """
    ).fetchall()


    connection.close()


    complaints = {}


    for row in rows:

        complaints[
            row["complaint_id"]
        ] = row_to_complaint(
            row
        )


    return complaints


# =========================================================
# CREATE NEXT DYNAMIC RG NUMBER
# =========================================================

def create_complaint_id():

    connection = get_db()


    rows = connection.execute(
        """
        SELECT complaint_id
        FROM complaints
        """
    ).fetchall()


    connection.close()


    highest_number = 1000


    for row in rows:

        complaint_id = (
            row["complaint_id"]
            or ""
        )


        if complaint_id.startswith(
            "RG-"
        ):

            try:

                number = int(
                    complaint_id[
                        3:
                    ]
                )


                if number > highest_number:

                    highest_number = (
                        number
                    )


            except ValueError:

                pass


    return (
        f"RG-{highest_number + 1}"
    )


# =========================================================
# HOME
# =========================================================

@app.route("/")
def home():

    return render_template(
        "index.html"
    )


# =========================================================
# REPORT PAGE
# =========================================================

@app.route("/report")
def report():

    return render_template(
        "report.html"
    )


# =========================================================
# GEMINI ROAD IMAGE ANALYSIS
# =========================================================

def analyze_road_image(
    image_path
):

    # -----------------------------------------------------
    # Gemini unavailable
    # -----------------------------------------------------

    if client is None:

        return {

            "issue_type":
                "AI unavailable",

            "estimated_depth":
                "Not available",

            "priority":
                "Medium",

            "possible_cause":
                "Gemini API key is not configured.",

            "recommendation":
                "Please configure the Gemini API key."

        }


    print()
    print(
        "========================================"
    )
    print(
        "AI ROAD GUARDIAN"
    )
    print(
        "Gemini is analyzing the image..."
    )
    print(
        "========================================"
    )
    print()


    try:

        # -------------------------------------------------
        # Upload image to Gemini
        # -------------------------------------------------

        uploaded_file = client.files.upload(
            file=image_path
        )


        # -------------------------------------------------
        # AI prompt
        # -------------------------------------------------

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


        # -------------------------------------------------
        # Generate AI response
        # -------------------------------------------------

        response = client.models.generate_content(

            model="gemini-3.5-flash",

            contents=[
                uploaded_file,
                prompt
            ]

        )


        result = (
            response.text or ""
        ).strip()


        print(
            "Gemini response:"
        )

        print(result)


        # -------------------------------------------------
        # Remove accidental Markdown fences
        # -------------------------------------------------

        result = result.replace(
            "```json",
            ""
        )

        result = result.replace(
            "```",
            ""
        )

        result = result.strip()


        # -------------------------------------------------
        # Parse JSON
        # -------------------------------------------------

        ai_result = json.loads(
            result
        )


        if not isinstance(
            ai_result,
            dict
        ):

            raise ValueError(
                "Gemini response was not a JSON object."
            )


        # -------------------------------------------------
        # Normalize AI result
        # -------------------------------------------------

        ai_result = {

            "issue_type":
                ai_result.get(
                    "issue_type",
                    "Not identified"
                ),

            "estimated_depth":
                ai_result.get(
                    "estimated_depth",
                    "Not reliably measurable from image."
                ),

            "priority":
                ai_result.get(
                    "priority",
                    "Medium"
                ),

            "possible_cause":
                ai_result.get(
                    "possible_cause",
                    "Could not be determined reliably."
                ),

            "recommendation":
                ai_result.get(
                    "recommendation",
                    "Physical inspection recommended."
                )

        }


        # -------------------------------------------------
        # Normalize priority
        # -------------------------------------------------

        priority = str(
            ai_result[
                "priority"
            ]
        ).strip().capitalize()


        if priority not in [
            "Low",
            "Medium",
            "High"
        ]:

            priority = "Medium"


        ai_result[
            "priority"
        ] = priority


        return ai_result


    # =====================================================
    # INVALID JSON
    # =====================================================

    except json.JSONDecodeError:

        print(
            "Gemini returned invalid JSON."
        )


        return {

            "issue_type":
                "Road damage detected",

            "estimated_depth":
                "Not reliably measurable from image.",

            "priority":
                "Medium",

            "possible_cause":
                "Physical inspection is required.",

            "recommendation":
                "A road authority engineer should inspect the reported location."

        }


    # =====================================================
    # OTHER AI ERROR
    # =====================================================

    except Exception as error:

        print(
            "Gemini analysis error:"
        )

        print(error)


        return {

            "issue_type":
                "AI analysis unavailable",

            "estimated_depth":
                "Not available",

            "priority":
                "Medium",

            "possible_cause":
                "AI analysis could not be completed.",

            "recommendation":
                "Please retry the report or arrange a physical inspection."

        }


# =========================================================
# SUBMIT REPORT
# =========================================================

def get_location_name(latitude, longitude):

    try:

        latitude = float(latitude)
        longitude = float(longitude)

        response = requests.get(
            "https://nominatim.openstreetmap.org/reverse",
            params={
                "lat": latitude,
                "lon": longitude,
                "format": "json",
                "zoom": 18,
                "addressdetails": 1
            },
            headers={
                "User-Agent": "AI-Road-Guardian/1.0"
            },
            timeout=10
        )

        response.raise_for_status()

        data = response.json()

        address = data.get("address", {})

        road = (
            address.get("road")
            or address.get("pedestrian")
            or address.get("residential")
        )

        locality = (
            address.get("suburb")
            or address.get("neighbourhood")
            or address.get("village")
            or address.get("town")
            or address.get("city")
        )

        district = (
            address.get("city_district")
            or address.get("district")
        )

        state = address.get("state")

        country = address.get("country")

        parts = []

        for value in [
            road,
            locality,
            district,
            state,
            country
        ]:

            if value and value not in parts:
                parts.append(value)

        if parts:

            return ", ".join(parts)

        return f"{latitude}, {longitude}"

    except Exception as error:

        print(
            "Reverse geocoding error:",
            error
        )

        return f"{latitude}, {longitude}"

@app.route(
    "/submit-report",
    methods=["POST"]
)
def submit_report():

    print()
    print(
        "========================================"
    )
    print(
        "New road report received."
    )
    print(
        "========================================"
    )


    # -----------------------------------------------------
    # Get uploaded photo
    # -----------------------------------------------------

    photo = request.files.get(
        "photo"
    )


    if not photo or not photo.filename:

        return (
            "No photo uploaded",
            400
        )


    # -----------------------------------------------------
    # Validate image format
    # -----------------------------------------------------

    extension = os.path.splitext(
        photo.filename
    )[1].lower()


    allowed_extensions = [

        ".jpg",
        ".jpeg",
        ".png",
        ".webp"

    ]


    if extension not in (
        allowed_extensions
    ):

        return (

            "Unsupported image format. "
            "Please upload JPG, JPEG, PNG, or WEBP.",

            400

        )


    # -----------------------------------------------------
    # Generate unique filename
    # -----------------------------------------------------

    filename = (
        str(uuid.uuid4())
        + extension
    )


    photo_path = os.path.join(

        app.config[
            "UPLOAD_FOLDER"
        ],

        filename

    )


    # -----------------------------------------------------
    # Save photo
    # -----------------------------------------------------

    photo.save(
        photo_path
    )


    print(
        "Photo saved:",
        photo_path
    )


    # -----------------------------------------------------
    # AI ANALYSIS
    # -----------------------------------------------------

    ai_result = analyze_road_image(
        photo_path
    )


    # -----------------------------------------------------
    # Dynamic RG ID
    # -----------------------------------------------------

    complaint_id = (
        create_complaint_id()
    )


    # -----------------------------------------------------
    # User-submitted information
    # -----------------------------------------------------

    location = request.form.get(
        "location",
        "Location not provided"
    ).strip()


    description = request.form.get(
        "description",
        "Road problem reported"
    ).strip()


    problem = request.form.get(
        "problem",
        ""
    ).strip()


    # If user did not provide problem,
    # use the AI-detected issue type.

    if not problem:

        problem = ai_result[
            "issue_type"
        ]


    # -----------------------------------------------------
    # Save complaint
    # -----------------------------------------------------

    connection = get_db()


    try:

        connection.execute(
            """
            INSERT INTO complaints
            (
                complaint_id,
                location,
                description,
                problem,
                priority,
                status,
                photo,
                ai_result
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,

            (

                complaint_id,

                location,

                description,

                problem,

                # AI priority
                ai_result[
                    "priority"
                ],

                # Initial workflow status
                "Pending",

                filename,

                json.dumps(
                    ai_result
                )

            )
        )


        connection.commit()


    finally:

        connection.close()


    print(
        "Complaint created:",
        complaint_id
    )


    print(
        "AI Priority:",
        ai_result[
            "priority"
        ]
    )


    # -----------------------------------------------------
    # Show AI analysis
    # -----------------------------------------------------

    return render_template(

        "analysis.html",

        complaint_id=
            complaint_id,

        ai_result=
            ai_result

    )


# =========================================================
# ANALYSIS PAGE
# =========================================================

@app.route("/analysis")
def analysis():

    complaint_id = request.args.get(
        "id"
    )


    if not complaint_id:

        return redirect(
            url_for("report")
        )


    complaint = get_complaint(
        complaint_id
    )


    if not complaint:

        return (
            "Complaint not found",
            404
        )


    return render_template(

        "analysis.html",

        complaint_id=
            complaint_id,

        ai_result=
            complaint[
                "ai_result"
            ]

    )


# =========================================================
# TRACK COMPLAINT
# =========================================================

@app.route("/track")
def track():

    complaint_id = request.args.get(
        "id"
    )


    # -----------------------------------------------------
    # No RG entered
    # -----------------------------------------------------

    if not complaint_id:

        return render_template(

            "track.html",

            complaint_id="",

            complaint=None

        )


    # -----------------------------------------------------
    # Normalize RG input
    # -----------------------------------------------------

    complaint_id = (
        complaint_id
        .strip()
        .upper()
    )


    # -----------------------------------------------------
    # Get complaint
    # -----------------------------------------------------

    complaint = get_complaint(
        complaint_id
    )


    return render_template(

        "track.html",

        complaint_id=
            complaint_id,

        complaint=
            complaint

    )


# =========================================================
# GOVERNMENT LOGIN PAGE
# =========================================================

@app.route(
    "/government-login"
)
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
        "username",
        ""
    ).strip()


    password = request.form.get(
        "password",
        ""
    )


    # -----------------------------------------------------
    # Existing government login
    # -----------------------------------------------------

    if (

        username ==
        "officer@government.in"

        and

        password ==
        "road123"

    ):

        session[
            "government_logged_in"
        ] = True


        return jsonify({

            "success":
                True,

            "message":
                "Login successful! Opening dashboard..."

        })


    return jsonify({

        "success":
            False,

        "message":
            "Incorrect email or password."

    })


# =========================================================
# GOVERNMENT DASHBOARD
# =========================================================

@app.route("/dashboard")
def dashboard():

    # -----------------------------------------------------
    # Require government login
    # -----------------------------------------------------

    if not session.get(
        "government_logged_in"
    ):

        return redirect(
            "/government-login"
        )


    # -----------------------------------------------------
    # Get all RG complaints dynamically
    # -----------------------------------------------------

    complaints = (
        get_all_complaints()
    )


    return render_template(

        "dashboard.html",

        complaints=
            complaints

    )


# =========================================================
# UPDATE STATUS FOR ANY RG
# =========================================================

@app.route(
    "/update-status/<complaint_id>",
    methods=["POST"]
)
def update_status(
    complaint_id
):

    # -----------------------------------------------------
    # Require government login
    # -----------------------------------------------------

    if not session.get(
        "government_logged_in"
    ):

        return redirect(
            "/government-login"
        )


    # -----------------------------------------------------
    # Normalize RG
    # -----------------------------------------------------

    complaint_id = (
        complaint_id
        .strip()
        .upper()
    )


    # -----------------------------------------------------
    # Get requested status
    # -----------------------------------------------------

    new_status = request.form.get(
        "status",
        ""
    ).strip()


    allowed_statuses = [

        "Pending",

        "Engineer Assigned",

        "In Progress",

        "Completed"

    ]


    if new_status not in (
        allowed_statuses
    ):

        return (
            "Invalid status",
            400
        )


    # -----------------------------------------------------
    # Update database
    # -----------------------------------------------------

    connection = get_db()


    try:

        existing = connection.execute(
            """
            SELECT complaint_id
            FROM complaints
            WHERE complaint_id = ?
            """,
            (
                complaint_id,
            )
        ).fetchone()


        if not existing:

            return (
                "Complaint not found",
                404
            )


        connection.execute(
            """
            UPDATE complaints
            SET status = ?
            WHERE complaint_id = ?
            """,

            (

                new_status,

                complaint_id

            )
        )


        connection.commit()


    finally:

        connection.close()


    print(
        f"{complaint_id} "
        f"status updated to "
        f"{new_status}"
    )


    return redirect(
        url_for("dashboard")
    )


# =========================================================
# EDIT ANY RG
# =========================================================

@app.route(
    "/edit-report/<complaint_id>",
    methods=["POST"]
)
def edit_report(
    complaint_id
):

    # -----------------------------------------------------
    # Require government login
    # -----------------------------------------------------

    if not session.get(
        "government_logged_in"
    ):

        return redirect(
            "/government-login"
        )


    # -----------------------------------------------------
    # Normalize RG
    # -----------------------------------------------------

    complaint_id = (
        complaint_id
        .strip()
        .upper()
    )


    # -----------------------------------------------------
    # Check complaint
    # -----------------------------------------------------

    complaint = get_complaint(
        complaint_id
    )


    if not complaint:

        return (
            "Complaint not found",
            404
        )


    # -----------------------------------------------------
    # Read editable fields
    # -----------------------------------------------------

    location = request.form.get(
        "location"
    )


    description = request.form.get(
        "description"
    )


    problem = request.form.get(
        "problem"
    )


    # -----------------------------------------------------
    # Update fields
    # -----------------------------------------------------

    connection = get_db()


    try:

        if location is not None:

            connection.execute(
                """
                UPDATE complaints
                SET location = ?
                WHERE complaint_id = ?
                """,

                (

                    location.strip(),

                    complaint_id

                )
            )


        if description is not None:

            connection.execute(
                """
                UPDATE complaints
                SET description = ?
                WHERE complaint_id = ?
                """,

                (

                    description.strip(),

                    complaint_id

                )
            )


        if problem is not None:

            connection.execute(
                """
                UPDATE complaints
                SET problem = ?
                WHERE complaint_id = ?
                """,

                (

                    problem.strip(),

                    complaint_id

                )
            )


        connection.commit()


    finally:

        connection.close()


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
# RUN APPLICATION
# =========================================================

if __name__ == "__main__":

    app.run(

        host="0.0.0.0",

        port=5000

    )