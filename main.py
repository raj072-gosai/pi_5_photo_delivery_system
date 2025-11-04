import os
import uuid
import pickle
import numpy as np
import hdbscan
from flask import Flask, request, render_template, url_for, send_from_directory
from deepface import DeepFace
from werkzeug.utils import secure_filename
from sklearn.preprocessing import normalize

# --- Configuration ---
UPLOAD_FOLDER = "static/uploads"
TEMP_FOLDER = "static/temp"
CLUSTERED_EMBEDDINGS_FILE = "instance/embeddings_clustered.pkl"
MIN_SAMPLES_FOR_CLUSTERING = 2

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(TEMP_FOLDER, exist_ok=True)
os.makedirs("instance", exist_ok=True)

# Load data
if os.path.exists(CLUSTERED_EMBEDDINGS_FILE):
    with open(CLUSTERED_EMBEDDINGS_FILE, "rb") as f:
        embeddings_db = pickle.load(f)
else:
    embeddings_db = []

# ===============================================================
# SECTION 1: PHOTOGRAPHER UPLOAD & PROCESSING ROUTES
# ===============================================================

@app.route("/", methods=["GET", "POST"])
def upload_event_photos():
    if request.method == "POST":
        files = request.files.getlist("photos")
        event_id = str(uuid.uuid4())[:8]
        event_path = os.path.join(app.config['UPLOAD_FOLDER'], event_id)
        os.makedirs(event_path, exist_ok=True)
        event_embeddings, event_file_paths = [], []

        for file in files:
            filename = secure_filename(file.filename)
            file_path = os.path.join(event_path, filename)
            file.save(file_path)
            try:
                face_objects = DeepFace.represent(img_path=file_path, model_name="ArcFace", detector_backend="retinaface")
                for face in face_objects:
                    event_embeddings.append(face["embedding"])
                    event_file_paths.append(file_path)
            except Exception as e:
                print(f"Could not process {filename}. Error: {e}")

        if len(event_embeddings) >= MIN_SAMPLES_FOR_CLUSTERING:
            data = np.array(event_embeddings)
            data = normalize(data, norm='l2')
            clusterer = hdbscan.HDBSCAN(min_cluster_size=2, metric='euclidean', cluster_selection_epsilon=0.5, min_samples=2)
            cluster_labels = clusterer.fit_predict(data)
        else:
            cluster_labels = [-1] * len(event_embeddings)

        for i in range(len(event_file_paths)):
            person_id = f"{event_id}_{cluster_labels[i]}"
            embeddings_db.append({"path": event_file_paths[i], "embedding": event_embeddings[i], "event": event_id, "person_id": person_id})

        with open(CLUSTERED_EMBEDDINGS_FILE, "wb") as f:
            pickle.dump(embeddings_db, f)

        link = url_for("user_access", event_id=event_id, _external=True)
        return f"✅ Event uploaded! Found {len(event_embeddings)} total faces. Share this link: <a href='{link}'>{link}</a>"
    return render_template("upload.html")

# ===============================================================
# SECTION 2: USER SEARCH & DOWNLOAD ROUTES
# ===============================================================

@app.route("/event/<event_id>", methods=["GET", "POST"])
def user_access(event_id):
    if request.method == "POST":
        # ... (user search logic remains the same)
        file = request.files["user_photo"]
        temp_path = os.path.join(TEMP_FOLDER, secure_filename(file.filename))
        file.save(temp_path)
        try:
            user_embedding = DeepFace.represent(img_path=temp_path, model_name="ArcFace", detector_backend="retinaface")[0]["embedding"]
        except:
            return "❌ No face detected in your photo. Try again."

        matches, event_data = [], [item for item in embeddings_db if item.get("event") == event_id]
        representatives, noise_points = {}, []
        for item in event_data:
            person_id = item.get("person_id")
            if person_id:
                if person_id.endswith("_-1"): noise_points.append(item)
                elif person_id not in representatives: representatives[person_id] = item["embedding"]
        
        found_person_id = None
        for person_id, rep_embedding in representatives.items():
            if DeepFace.verify(user_embedding, rep_embedding, model_name="ArcFace")["verified"]:
                found_person_id = person_id
                break
        
        if found_person_id:
            matches = [item["path"] for item in event_data if item.get("person_id") == found_person_id]
        elif noise_points:
            for item in noise_points:
                if DeepFace.verify(user_embedding, item["embedding"], model_name="ArcFace")["verified"]:
                    matches.append(item["path"])
                    break
        return render_template("results.html", matches=matches)
    return render_template("user_link.html", event_id=event_id)

@app.route("/download/<path:filename>")
def download_file(filename):
    return send_from_directory(directory=".", path=filename, as_attachment=True)

# ===============================================================
# SECTION 3: GALLERY & ALBUM ROUTES
# ===============================================================

@app.route("/gallery")
@app.route("/gallery/<event_id>")
def event_gallery(event_id=None):
    """
    Shows a gallery of all people. If an event_id is provided,
    it filters the gallery for just that event.
    """
    people, title = {}, "All People"
    db_to_scan = embeddings_db

    if event_id:
        title = f"People at Event: {event_id}"
        db_to_scan = [item for item in embeddings_db if item.get("event") == event_id]

    for item in db_to_scan:
        person_id = item.get("person_id")
        if person_id and not person_id.endswith("_-1"):
            if person_id not in people:
                people[person_id] = {"person_id": person_id, "cover_photo": item["path"], "all_photos": set()}
            people[person_id]["all_photos"].add(item["path"])

    for person_id in people:
        people[person_id]["photo_count"] = len(people[person_id]["all_photos"])
    
    people_list = sorted(list(people.values()), key=lambda x: x['person_id'])
    return render_template("gallery.html", people=people_list, title=title)

@app.route("/person/<person_id>")
def person_album(person_id):
    person_photos = set()
    for item in embeddings_db:
        if item.get("person_id") == person_id:
            person_photos.add(item["path"])
    return render_template("person_detail.html", person_id=person_id, photos=sorted(list(person_photos)))

# ===============================================================
# SECTION 4: MAIN EXECUTION
# ===============================================================

if __name__ == "__main__":
    app.run(debug=True)