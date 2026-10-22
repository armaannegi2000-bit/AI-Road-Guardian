from flask import Flask, render_template, request, redirect, url_for, session, jsonify
from dotenv import load_dotenv
from google import genai
import os
import json
import uuid
import sqlite3
import requests
from werkzeug.security import check_password_hash, generate_password_hash

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
            "TIMESTAMP",

        "assigned_engineer":
            "TEXT",

        "assigned_engineer_id":
            "INTEGER"

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
# COMPLAINT STATUS HISTORY
# =========================================================

def init_status_history():
    connection = get_db()

    try:
        connection.execute("""
            CREATE TABLE IF NOT EXISTS complaint_status_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                complaint_id TEXT NOT NULL,
                status TEXT NOT NULL,
                changed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        connection.commit()

    finally:
        connection.close()


init_status_history()


def init_repair_verifications():

    connection = get_db()

    try:

        connection.execute("""
            CREATE TABLE IF NOT EXISTS repair_verifications (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                complaint_id TEXT NOT NULL,
                engineer_name TEXT NOT NULL,
                note TEXT NOT NULL,
                photo TEXT NOT NULL,
                review_status TEXT NOT NULL DEFAULT 'Pending Review',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                reviewed_at TIMESTAMP
            )
        """)

        connection.commit()

    finally:

        connection.close()


init_repair_verifications()


def init_engineer_work_updates():

    connection = get_db()

    try:

        connection.execute("""
            CREATE TABLE IF NOT EXISTS engineer_work_updates (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                complaint_id TEXT NOT NULL,
                engineer_name TEXT NOT NULL,
                status TEXT NOT NULL,
                note TEXT NOT NULL,
                photo TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        connection.commit()

    finally:
        connection.close()


init_engineer_work_updates()


def init_notifications():

    connection = get_db()

    try:

        connection.execute("""
            CREATE TABLE IF NOT EXISTS notifications (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                audience TEXT NOT NULL,
                complaint_id TEXT NOT NULL,
                message TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                is_read INTEGER NOT NULL DEFAULT 0
            )
        """)

        connection.commit()

    finally:
        connection.close()


init_notifications()


def init_engineers():
    connection = get_db()
    try:
        connection.execute("""
            CREATE TABLE IF NOT EXISTS engineers (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                full_name TEXT NOT NULL,
                email TEXT NOT NULL UNIQUE,
                password_hash TEXT NOT NULL,
                active INTEGER NOT NULL DEFAULT 1,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        connection.commit()
    finally:
        connection.close()


init_engineers()


def get_engineers():
    connection = get_db()
    try:
        return connection.execute(
            "SELECT id, full_name, email FROM engineers WHERE active = 1 ORDER BY full_name"
        ).fetchall()
    finally:
        connection.close()


def add_notification(connection, audience, complaint_id, message):

    connection.execute(
        """
        INSERT INTO notifications (audience, complaint_id, message)
        VALUES (?, ?, ?)
        """,
        (audience, complaint_id, message)
    )


def get_notifications(audience, limit=10):

    connection = get_db()

    try:
        rows = connection.execute(
            """
            SELECT id, complaint_id, message, created_at, is_read
            FROM notifications
            WHERE audience = ?
            ORDER BY created_at DESC, id DESC
            LIMIT ?
            """,
            (audience, limit)
        ).fetchall()
        return [dict(row) for row in rows]

    finally:
        connection.close()


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
            row["created_at"],

        "assigned_engineer":
            row["assigned_engineer"] or "",

        "assigned_engineer_id":
            row["assigned_engineer_id"] if "assigned_engineer_id" in row.keys() else None

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

def get_complaint_status_history(complaint_id):

    connection = get_db()

    try:

        rows = connection.execute(
            """
            SELECT status, changed_at
            FROM complaint_status_history
            WHERE complaint_id = ?
            ORDER BY changed_at ASC, id ASC
            """,
            (complaint_id,)
        ).fetchall()

        return [dict(row) for row in rows]

    finally:

        connection.close()


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

        complaint = row_to_complaint(row)
        complaint["latest_verification"] = get_latest_repair_verification(
            row["complaint_id"]
        )
        work_updates = get_engineer_work_updates(row["complaint_id"])
        complaint["work_updates"] = work_updates
        complaint["latest_work_update"] = work_updates[0] if work_updates else None
        complaints[row["complaint_id"]] = complaint


    return complaints


def get_engineer_work_updates(complaint_id):

    connection = get_db()

    try:

        rows = connection.execute(
            """
            SELECT id, engineer_name, status, note, photo, created_at
            FROM engineer_work_updates
            WHERE complaint_id = ?
            ORDER BY created_at DESC, id DESC
            """,
            (complaint_id,)
        ).fetchall()

        return [dict(row) for row in rows]

    finally:
        connection.close()


def get_latest_repair_verification(complaint_id):

    connection = get_db()

    try:

        row = connection.execute(
            """
            SELECT *
            FROM repair_verifications
            WHERE complaint_id = ?
            ORDER BY created_at DESC, id DESC
            LIMIT 1
            """,
            (complaint_id,)
        ).fetchone()

        return dict(row) if row else None

    finally:

        connection.close()


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

    latitude = request.form.get("latitude", "").strip()
    longitude = request.form.get("longitude", "").strip()

    # The report form submits GPS coordinates in its location field.
    # Also accept separate coordinate fields if a form sends them.
    if not (latitude and longitude) and "," in location:

        possible_latitude, possible_longitude = (
            part.strip()
            for part in location.split(",", 1)
        )

        try:

            float(possible_latitude)
            float(possible_longitude)
            latitude = possible_latitude
            longitude = possible_longitude

        except ValueError:

            pass

    if latitude and longitude:

        gps_coordinates = (
            f"{float(latitude):.6f}, {float(longitude):.6f}"
        )
        address = get_location_name(
            latitude,
            longitude
        )

        if address == gps_coordinates:
            location = f"GPS coordinates: {gps_coordinates}"
        else:
            location = (
                f"{address} (GPS coordinates: {gps_coordinates})"
            )


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

        connection.execute(
            """
            INSERT INTO complaint_status_history
            (complaint_id, status)
            VALUES (?, ?)
            """,
            (complaint_id, "Pending")
        )

        add_notification(
            connection,
            "government",
            complaint_id,
            f"New road complaint {complaint_id} was submitted."
        )
        add_notification(
            connection,
            "engineers",
            complaint_id,
            f"New complaint {complaint_id} is available in the engineer inbox."
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

            complaint=None,

            work_updates=[],

            repair_verification=None

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

    status_history = (
        get_complaint_status_history(complaint_id)
        if complaint
        else []
    )

    work_updates = (
        get_engineer_work_updates(complaint_id)
        if complaint
        else []
    )

    return render_template(

        "track.html",

        complaint_id=
            complaint_id,

        complaint=
            complaint,

        status_history=
            status_history,

        work_updates=
            work_updates,

        repair_verification=(
            get_latest_repair_verification(complaint_id)
            if complaint
            else None
        )

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

        session.clear()

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
            complaints,

        notifications=
            get_notifications("government"),

        engineers=
            get_engineers(),

        unread_notification_count=
            sum(notice["is_read"] == 0 for notice in get_notifications("government"))

    )


# =========================================================
# ENGINEER DASHBOARD
# =========================================================

@app.route("/engineer-login")
def engineer_login():
    if session.get("engineer_id"):
        return redirect(url_for("engineer_dashboard"))
    return render_template("engineer_login.html")


@app.route("/engineer-auth", methods=["POST"])
def engineer_auth():
    email = request.form.get("email", "").strip().lower()
    password = request.form.get("password", "")
    connection = get_db()
    try:
        engineer = connection.execute(
            "SELECT id, full_name, email, password_hash FROM engineers WHERE email = ? AND active = 1",
            (email,)
        ).fetchone()
    finally:
        connection.close()

    if not engineer or not check_password_hash(engineer["password_hash"], password):
        return render_template("engineer_login.html", error="Email or password is incorrect.")

    session.clear()
    session["engineer_id"] = engineer["id"]
    session["engineer_name"] = engineer["full_name"]
    return redirect(url_for("engineer_dashboard"))


@app.route("/create-engineer", methods=["POST"])
def create_engineer():
    if not session.get("government_logged_in"):
        return redirect("/government-login")

    full_name = request.form.get("full_name", "").strip()
    email = request.form.get("email", "").strip().lower()
    password = request.form.get("password", "")
    if not full_name or not email or len(password) < 8:
        return ("Enter the engineer name and email, and use a password of at least 8 characters.", 400)

    connection = get_db()
    try:
        connection.execute(
            "INSERT INTO engineers (full_name, email, password_hash) VALUES (?, ?, ?)",
            (full_name, email, generate_password_hash(password))
        )
        connection.commit()
    except sqlite3.IntegrityError:
        return ("An engineer account already uses that email.", 409)
    finally:
        connection.close()

    return redirect(url_for("dashboard"))


@app.route("/engineer-dashboard")
def engineer_dashboard():

    engineer_id = session.get("engineer_id")
    if not engineer_id:
        return redirect("/engineer-login")

    engineer_name = session.get("engineer_name", "Engineer")
    all_complaints = get_all_complaints()

    complaints = {
        complaint_id: complaint
        for complaint_id, complaint in all_complaints.items()
        if not complaint.get("assigned_engineer")
        or complaint.get("assigned_engineer_id") == engineer_id
    }

    return render_template(
        "engineer_dashboard.html",
        engineer_name=engineer_name,
        complaints=complaints,
        notifications=get_notifications("engineers"),
        unread_notification_count=sum(
            notice["is_read"] == 0
            for notice in get_notifications("engineers")
        )
    )


@app.route("/claim-complaint/<complaint_id>", methods=["POST"])
def claim_complaint(complaint_id):

    engineer_id = session.get("engineer_id")
    if not engineer_id:
        return redirect("/engineer-login")

    complaint_id = complaint_id.strip().upper()
    engineer_name = session.get("engineer_name", "Engineer")

    connection = get_db()

    try:

        cursor = connection.execute(
            """
            UPDATE complaints
            SET assigned_engineer = ?, assigned_engineer_id = ?, status = 'Engineer Assigned'
            WHERE complaint_id = ?
              AND (assigned_engineer IS NULL OR assigned_engineer = '')
              AND status != 'Completed'
            """,
            (engineer_name, engineer_id, complaint_id)
        )

        if cursor.rowcount != 1:
            exists = connection.execute(
                "SELECT complaint_id FROM complaints WHERE complaint_id = ?",
                (complaint_id,)
            ).fetchone()
            if not exists:
                return ("Complaint not found", 404)
            return ("This complaint has already been claimed or completed", 409)

        connection.execute(
            """
            INSERT INTO complaint_status_history (complaint_id, status)
            VALUES (?, 'Engineer Assigned')
            """,
            (complaint_id,)
        )
        add_notification(
            connection,
            "government",
            complaint_id,
            f"An engineer claimed complaint {complaint_id}."
        )
        connection.commit()

    finally:
        connection.close()

    return redirect(url_for("engineer_dashboard"))


@app.route("/assign-engineer/<complaint_id>", methods=["POST"])
def assign_engineer(complaint_id):

    if not session.get("government_logged_in"):
        return redirect("/government-login")

    complaint_id = complaint_id.strip().upper()
    engineer_id = request.form.get("engineer_id", type=int)
    if not engineer_id:
        return ("Select an active engineer account", 400)

    connection = get_db()
    selected_engineer = connection.execute(
        "SELECT full_name FROM engineers WHERE id = ? AND active = 1",
        (engineer_id,)
    ).fetchone()
    if not selected_engineer:
        connection.close()
        return ("Engineer account not found", 404)
    engineer_name = selected_engineer["full_name"]

    try:

        existing = connection.execute(
            "SELECT complaint_id, status FROM complaints WHERE complaint_id = ?",
            (complaint_id,)
        ).fetchone()

        if not existing:
            return ("Complaint not found", 404)

        if existing["status"] == "Completed":
            return ("Completed complaints cannot be reassigned", 400)

        new_status = "Engineer Assigned"
        connection.execute(
            """
            UPDATE complaints
            SET assigned_engineer = ?, assigned_engineer_id = ?, status = ?
            WHERE complaint_id = ?
            """,
            (engineer_name, engineer_id, new_status, complaint_id)
        )

        if existing["status"] != new_status:
            connection.execute(
                """
                INSERT INTO complaint_status_history (complaint_id, status)
                VALUES (?, ?)
                """,
                (complaint_id, new_status)
            )

        add_notification(
            connection,
            "engineers",
            complaint_id,
            f"Complaint {complaint_id} was assigned to {engineer_name}."
        )
        connection.commit()

    finally:
        connection.close()

    return redirect(url_for("dashboard"))


@app.route("/post-engineer-update/<complaint_id>", methods=["POST"])
def post_engineer_update(complaint_id):

    engineer_id = session.get("engineer_id")
    if not engineer_id:
        return redirect("/engineer-login")

    complaint_id = complaint_id.strip().upper()
    engineer_name = session.get("engineer_name", "Engineer")
    note = request.form.get("note", "").strip()
    new_status = request.form.get("status", "In Progress").strip()
    photo = request.files.get("photo")

    if not note:
        return ("A work update is required", 400)

    if new_status not in ["Engineer Assigned", "In Progress"]:
        return ("Invalid work status", 400)

    filename = None
    saved_path = None

    if photo and photo.filename:
        extension = os.path.splitext(photo.filename)[1].lower()
        if extension not in [".jpg", ".jpeg", ".png", ".webp"]:
            return ("Unsupported image format", 400)
        filename = str(uuid.uuid4()) + extension
        saved_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        photo.save(saved_path)

    connection = get_db()

    try:

        complaint = connection.execute(
            """
            SELECT status, assigned_engineer, assigned_engineer_id
            FROM complaints
            WHERE complaint_id = ?
            """,
            (complaint_id,)
        ).fetchone()

        if not complaint:
            return ("Complaint not found", 404)

        if complaint["assigned_engineer_id"] != engineer_id:
            return ("This complaint is not assigned to that engineer", 403)

        if complaint["status"] == "Completed":
            return ("Completed complaints cannot receive new work updates", 400)

        connection.execute(
            """
            INSERT INTO engineer_work_updates
            (complaint_id, engineer_name, status, note, photo)
            VALUES (?, ?, ?, ?, ?)
            """,
            (complaint_id, engineer_name, new_status, note, filename)
        )

        if complaint["status"] != new_status:
            connection.execute(
                "UPDATE complaints SET status = ? WHERE complaint_id = ?",
                (new_status, complaint_id)
            )
            connection.execute(
                """
                INSERT INTO complaint_status_history (complaint_id, status)
                VALUES (?, ?)
                """,
                (complaint_id, new_status)
            )

        add_notification(
            connection,
            "government",
            complaint_id,
            f"Engineer {engineer_name} posted a progress update for {complaint_id}."
        )
        connection.commit()

    except Exception:
        if saved_path and os.path.exists(saved_path):
            os.remove(saved_path)
        raise

    finally:
        connection.close()

    return redirect(url_for("engineer_dashboard"))


@app.route("/submit-repair-verification/<complaint_id>", methods=["POST"])
def submit_repair_verification(complaint_id):

    engineer_id = session.get("engineer_id")
    if not engineer_id:
        return redirect("/engineer-login")

    complaint_id = complaint_id.strip().upper()
    engineer_name = session.get("engineer_name", "Engineer")
    note = request.form.get("note", "").strip()
    photo = request.files.get("photo")

    if not note:
        return ("A repair note is required", 400)

    if not photo or not photo.filename:
        return ("A repair verification photo is required", 400)

    extension = os.path.splitext(photo.filename)[1].lower()
    if extension not in [".jpg", ".jpeg", ".png", ".webp"]:
        return ("Unsupported image format", 400)

    connection = get_db()
    saved_path = None

    try:

        complaint = connection.execute(
            """
            SELECT complaint_id, status, assigned_engineer, assigned_engineer_id
            FROM complaints
            WHERE complaint_id = ?
            """,
            (complaint_id,)
        ).fetchone()

        if not complaint:
            return ("Complaint not found", 404)

        if complaint["assigned_engineer_id"] != engineer_id:
            return ("This complaint is not assigned to that engineer", 403)

        filename = str(uuid.uuid4()) + extension
        saved_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        photo.save(saved_path)

        connection.execute(
            """
            INSERT INTO repair_verifications
            (complaint_id, engineer_name, note, photo)
            VALUES (?, ?, ?, ?)
            """,
            (complaint_id, engineer_name, note, filename)
        )

        if complaint["status"] != "In Progress":
            connection.execute(
                "UPDATE complaints SET status = ? WHERE complaint_id = ?",
                ("In Progress", complaint_id)
            )
            connection.execute(
                """
                INSERT INTO complaint_status_history (complaint_id, status)
                VALUES (?, ?)
                """,
                (complaint_id, "In Progress")
            )

        connection.commit()

    except Exception:
        if saved_path and os.path.exists(saved_path):
            os.remove(saved_path)
        raise

    finally:
        connection.close()

    return redirect(url_for("engineer_dashboard"))


@app.route("/review-repair-verification/<int:verification_id>", methods=["POST"])
def review_repair_verification(verification_id):

    if not session.get("government_logged_in"):
        return redirect("/government-login")

    decision = request.form.get("decision", "").strip()
    if decision not in ["approve", "rework"]:
        return ("Invalid verification decision", 400)

    connection = get_db()

    try:

        verification = connection.execute(
            """
            SELECT complaint_id, review_status
            FROM repair_verifications
            WHERE id = ?
            """,
            (verification_id,)
        ).fetchone()

        if not verification:
            return ("Repair verification not found", 404)

        if verification["review_status"] != "Pending Review":
            return ("This verification has already been reviewed", 400)

        new_review_status = "Approved" if decision == "approve" else "Needs Rework"
        new_complaint_status = "Completed" if decision == "approve" else "In Progress"

        current = connection.execute(
            "SELECT status FROM complaints WHERE complaint_id = ?",
            (verification["complaint_id"],)
        ).fetchone()

        connection.execute(
            """
            UPDATE repair_verifications
            SET review_status = ?, reviewed_at = CURRENT_TIMESTAMP
            WHERE id = ?
            """,
            (new_review_status, verification_id)
        )

        if current and current["status"] != new_complaint_status:
            connection.execute(
                "UPDATE complaints SET status = ? WHERE complaint_id = ?",
                (new_complaint_status, verification["complaint_id"])
            )
            connection.execute(
                """
                INSERT INTO complaint_status_history (complaint_id, status)
                VALUES (?, ?)
                """,
                (verification["complaint_id"], new_complaint_status)
            )

        add_notification(
            connection,
            "engineers",
            verification["complaint_id"],
            f"Government {new_review_status.lower()} repair verification for {verification['complaint_id']}."
        )
        connection.commit()

    finally:
        connection.close()

    return redirect(url_for("dashboard"))


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
            SELECT complaint_id, status
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


        current_status = existing["status"]

        if new_status == "Completed":
            approved = connection.execute(
                """
                SELECT 1
                FROM repair_verifications
                WHERE complaint_id = ?
                  AND review_status = 'Approved'
                LIMIT 1
                """,
                (complaint_id,)
            ).fetchone()

            if not approved:
                return (
                    "A repair verification must be approved before completion",
                    400
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

        if current_status != new_status:

            connection.execute(
                """
                INSERT INTO complaint_status_history
                (complaint_id, status)
                VALUES (?, ?)
                """,
                (complaint_id, new_status)
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


@app.route("/notifications/<int:notification_id>/read", methods=["POST"])
def mark_notification_read(notification_id):

    audience = request.form.get("audience", "")
    if audience not in ["government", "engineers"]:
        return ("Invalid notification audience", 400)
    if audience == "government" and not session.get("government_logged_in"):
        return redirect("/government-login")
    if audience == "engineers" and not session.get("engineer_id"):
        return redirect("/engineer-login")

    connection = get_db()

    try:
        connection.execute(
            "UPDATE notifications SET is_read = 1 WHERE id = ? AND audience = ?",
            (notification_id, audience)
        )
        connection.commit()
    finally:
        connection.close()

    if audience == "engineers":
        return redirect(url_for("engineer_dashboard"))
    return redirect(url_for("dashboard"))


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


@app.route("/engineer-logout")
def engineer_logout():
    session.pop("engineer_id", None)
    session.pop("engineer_name", None)
    return redirect(url_for("engineer_login"))


# =========================================================
# RUN APPLICATION
# =========================================================

if __name__ == "__main__":

    app.run(

        host="0.0.0.0",

        port=5000

    )
