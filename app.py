from flask import Flask, render_template, request, redirect, session, flash, url_for
import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import mysql.connector

app = Flask(__name__)
app.secret_key = "secret"

UPLOAD_FOLDER = "static/uploads/"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

model = load_model("model/vgg16_malignant_vs_benign.h5")

db = mysql.connector.connect(
    host="localhost",
    user="root",
    password="",
    database="skin_cancer_db"
)

cursor = db.cursor(dictionary=True)


@app.route('/')
def index():
    return render_template('login.html')


# ---------------- LOGIN ----------------
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        user = request.form.get("username")
        pwd  = request.form.get("password")

        cursor.execute("SELECT * FROM users WHERE username=%s", (user,))
        existing_user = cursor.fetchone()

        if not existing_user:
            flash(f"L'identifiant « {user} » n'existe pas. Créez votre compte.", "warning")
            return redirect(url_for("signup"))

        cursor.execute(
            "SELECT * FROM users WHERE username=%s AND password=%s",
            (user, pwd)
        )
        result = cursor.fetchone()

        if result:
            session["user"] = result["username"]
            session["role"] = result.get("role", "Médecin")
            return redirect(url_for("dashboard"))
        else:
            flash("Mot de passe incorrect ✖", "danger")

    return render_template("login.html")


# ---------------- SIGNUP ----------------
@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        first_name  = request.form.get("first_name", "").strip()
        last_name   = request.form.get("last_name", "").strip()
        username    = request.form.get("username", "").strip()
        email       = request.form.get("email", "").strip().lower()
        role        = request.form.get("role", "autre").strip()

        if role == "autre":
            role_custom = request.form.get("role_custom", "").strip()
            if role_custom:
                role = role_custom

        pwd         = request.form.get("password", "")
        confirm_pwd = request.form.get("confirm_password", "")

        errors = []

        if not all([first_name, last_name, username, email, pwd]):
            errors.append("Tous les champs sont obligatoires.")
        if len(username) < 3:
            errors.append("L'identifiant doit contenir au moins 3 caractères.")
        if len(pwd) < 8:
            errors.append("Le mot de passe doit contenir au moins 8 caractères.")
        if pwd != confirm_pwd:
            errors.append("Les mots de passe ne correspondent pas.")

        cursor.execute("SELECT id FROM users WHERE username=%s", (username,))
        if cursor.fetchone():
            errors.append(f"L'identifiant « {username} » est déjà utilisé.")

        cursor.execute("SELECT id FROM users WHERE email=%s", (email,))
        if cursor.fetchone():
            errors.append("Cette adresse e-mail est déjà associée à un compte.")

        if errors:
            for err in errors:
                flash(err, "danger")
            return render_template("signup.html")

        cursor.execute("""
            INSERT INTO users (username, password, first_name, last_name, email, role)
            VALUES (%s, %s, %s, %s, %s, %s)
        """, (username, pwd, first_name, last_name, email, role))

        db.commit()

        session["user"] = username
        session["role"] = role

        flash(f"Bienvenue, {first_name} ! Votre compte a été créé. ✔", "success")
        return redirect(url_for("dashboard"))

    return render_template("signup.html")


# ---------------- FORGOT PASSWORD ----------------
@app.route("/forgot-password", methods=["GET", "POST"])
def forgot_password():
    if request.method == "POST":
        email = request.form.get("email", "").strip().lower()
        cursor.execute("SELECT * FROM users WHERE email=%s", (email,))
        # Always show success to avoid email enumeration
        flash("Si cet e-mail est associé à un compte, un lien de réinitialisation a été envoyé.", "success")
        return redirect(url_for("forgot_password"))
    return render_template("forget_password.html")


# ---------------- DASHBOARD ----------------
@app.route("/dashboard")
def dashboard():
    if "user" not in session:
        return redirect(url_for("login"))

    cursor.execute("SELECT COUNT(*) AS total FROM patients")
    patients_count  = cursor.fetchone()["total"]
    analyses_count  = patients_count
    pending_count   = 0

    # Priority patients (high-risk)
    cursor.execute("""
        SELECT name, age, result, probability, image_path
        FROM patients
        WHERE result = 'Malignant' OR probability >= 0.50
        ORDER BY probability DESC
        LIMIT 5
    """)
    priority_patients = cursor.fetchall()
    priority_count    = len(priority_patients)

    # Recent patients for table (last 10)
    cursor.execute("""
        SELECT name, age, result, probability, image_path
        FROM patients
        ORDER BY id DESC
        LIMIT 10
    """)
    recent_patients = cursor.fetchall()

    notifications = []
    if priority_count > 0:
        notifications.append({
            "title":   "Priorité IA",
            "message": f"{priority_count} patient(s) nécessitent une attention.",
            "type":    "warning",
            "is_read": False
        })
    if patients_count > 0:
        notifications.append({
            "title":   "Analyses disponibles",
            "message": f"{patients_count} patient(s) enregistré(s).",
            "type":    "success",
            "is_read": False
        })
    else:
        notifications.append({
            "title":   "Bienvenue",
            "message": "Commencez par ajouter une première analyse.",
            "type":    "info",
            "is_read": False
        })

    notif_unread = sum(1 for n in notifications if not n["is_read"])

    return render_template(
        "dashboard.html",
        analyses_count    = analyses_count,
        pending_count     = pending_count,
        patients_count    = patients_count,
        notifications     = notifications,
        notif_unread      = notif_unread,
        priority_patients = priority_patients,
        recent_patients   = recent_patients
    )


# ---------------- PREDICTION ----------------
@app.route("/predict", methods=["GET", "POST"])
def predict():
    if "user" not in session:
        return redirect(url_for("login"))

    if request.method == "POST":
        try:
            name = request.form["name"]
            age  = request.form["age"]
            file = request.files["image"]

            if file.filename == "":
                flash("Veuillez choisir une image.", "warning")
                return redirect(url_for("predict"))

            filename = file.filename
            path     = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(path)

            db_path = "/" + path.replace("\\", "/")

            img = image.load_img(path, target_size=(224, 224))
            img = image.img_to_array(img) / 255.0
            img = np.expand_dims(img, axis=0)

            pred        = model.predict(img)[0][0]
            result      = "Malignant" if pred > 0.5 else "Benign"
            probability = float(pred)

            cursor.execute("""
                INSERT INTO patients (name, age, result, probability, image_path)
                VALUES (%s, %s, %s, %s, %s)
            """, (name, age, result, probability, db_path))

            db.commit()

            flash("Analyse réussie ✔ Patient ajouté avec succès.", "success")
            return render_template(
                "result.html",
                result = result,
                prob   = round(probability * 100, 1),
                img    = db_path,
                name   = name,
                age    = age
            )

        except Exception as e:
            print("Erreur predict :", e)
            flash(f"Erreur système : {e}", "danger")
            return redirect(url_for("predict"))

    return render_template("predict.html")


# ---------------- PATIENTS ----------------
@app.route("/patients")
def patients():
    if "user" not in session:
        return redirect(url_for("login"))
    cursor.execute("SELECT * FROM patients ORDER BY id DESC")
    data = cursor.fetchall()
    return render_template("patients.html", patients=data)


# ---------------- HELP PAGE ----------------
@app.route("/help")
def help_page():
    if "user" not in session:
        return redirect(url_for("login"))
    return render_template("help.html")


# ---------------- SETTINGS ----------------
@app.route("/settings")
def settings():
    if "user" not in session:
        return redirect(url_for("login"))
    return render_template("settings.html")


# ---------------- NOTIFICATIONS ----------------
@app.route("/notifications")
def notifications_page():
    if "user" not in session:
        return redirect(url_for("login"))
    return render_template("notifications.html")


# ---------------- LOGOUT ----------------
@app.route("/logout")
def logout():
    session.clear()
    flash("Déconnecté avec succès.", "info")
    return redirect(url_for("login"))


# ---------------- MAIN ----------------
if __name__ == "__main__":
    app.run(debug=True)