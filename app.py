import os, uuid, pickle
from flask import Flask, request, render_template, url_for, send_from_directory
from deepface import DeepFace
from werkzeug.utils import secure_filename

UPLOAD_FOLDER = "static/uploads"
TEMP_FOLDER = "static/temp"
EMBEDDINGS_FILE = "instance/embeddings.pkl"

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(TEMP_FOLDER, exist_ok=True)
os.makedirs("instance", exist_ok=True)

# load embeddings if exist
if os.path.exists(EMBEDDINGS_FILE):
    with open(EMBEDDINGS_FILE, "rb") as f:
        embeddings_db = pickle.load(f)
else:
    embeddings_db = []

# --------------- Photographer Upload -----------------
@app.route("/", methods=["GET", "POST"])
def upload_event_photos():
    if request.method == "POST":
        files = request.files.getlist("photos")
        event_id = str(uuid.uuid4())[:8]
        event_path = os.path.join(app.config['UPLOAD_FOLDER'], event_id)
        os.makedirs(event_path, exist_ok=True)

        for file in files:
            filename = secure_filename(file.filename)
            file_path = os.path.join(event_path, filename)
            file.save(file_path)

            try:
                embedding = DeepFace.represent(img_path=file_path, model_name="ArcFace")[0]["embedding"]
                embeddings_db.append({"path": file_path, "embedding": embedding, "event": event_id})
            except Exception as e:
                print("Face not detected in:", file_path, e)

        # save embeddings
        with open(EMBEDDINGS_FILE, "wb") as f:
            pickle.dump(embeddings_db, f)

        # generate link for users
        link = url_for("user_access", event_id=event_id, _external=True)
        return f"✅ Event uploaded! Share this link with friends: <a href='{link}'>{link}</a>"

    return render_template("upload.html")

# --------------- User Access -----------------
@app.route("/event/<event_id>", methods=["GET", "POST"])
def user_access(event_id):
    if request.method == "POST":
        file = request.files["user_photo"]
        temp_path = os.path.join(TEMP_FOLDER, secure_filename(file.filename))
        file.save(temp_path)

        try:
            user_embedding = DeepFace.represent(img_path=temp_path, model_name="ArcFace")[0]["embedding"]
        except:
            return "❌ No face detected in your photo. Try again."

        matches = []
        for item in embeddings_db:
            if item["event"] == event_id:
                dist = DeepFace.verify(user_embedding, item["embedding"], model_name="ArcFace")
                if dist["verified"]:
                    matches.append(item["path"])

        return render_template("results.html", matches=matches)

    return render_template("user_link.html", event_id=event_id)

# --------------- Download -----------------
@app.route("/download/<path:filename>")
def download_file(filename):
    return send_from_directory(directory=".", path=filename, as_attachment=True)

if __name__ == "__main__":
    app.run(debug=True)
