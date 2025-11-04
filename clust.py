import os
import uuid
import pickle
import numpy as np
import hdbscan
from flask import Flask, request, render_template, url_for, send_from_directory
from deepface import DeepFace
from werkzeug.utils import secure_filename
# --- NEW IMPORT ---
from sklearn.preprocessing import normalize

UPLOAD_FOLDER = "static/uploads"
TEMP_FOLDER = "static/temp"
CLUSTERED_EMBEDDINGS_FILE = "instance/embeddings_clustered.pkl"

app = Flask(__name__)
# ... (rest of the initial app setup is the same)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(TEMP_FOLDER, exist_ok=True)
os.makedirs("instance", exist_ok=True)

if os.path.exists(CLUSTERED_EMBEDDINGS_FILE):
    with open(CLUSTERED_EMBEDDINGS_FILE, "rb") as f:
        embeddings_db = pickle.load(f)
else:
    embeddings_db = []

# --------------- Photographer Upload with Multi-Face Processing -----------------
@app.route("/", methods=["GET", "POST"])
def upload_event_photos():
    if request.method == "POST":
        files = request.files.getlist("photos")
        event_id = str(uuid.uuid4())[:8]
        event_path = os.path.join(app.config['UPLOAD_FOLDER'], event_id)
        os.makedirs(event_path, exist_ok=True)

        event_embeddings = []
        event_file_paths = []

        print("Starting photo processing...")
        # Loop through each uploaded file
        for file in files:
            filename = secure_filename(file.filename)
            file_path = os.path.join(event_path, filename)
            file.save(file_path)

            try:
                # ===============================================================
                # THE CRITICAL CHANGE IS HERE
                # Find ALL faces in the image, not just the first one.
                # ===============================================================
                face_objects = DeepFace.represent(
                    img_path=file_path,
                    model_name="ArcFace",
                    detector_backend="retinaface"
                )

                # Now, loop through each face found in the single image
                for face in face_objects:
                    embedding = face["embedding"]
                    event_embeddings.append(embedding)
                    # We add the same file path for each face found in it
                    event_file_paths.append(file_path)
                
                print(f"Found {len(face_objects)} faces in {filename}")

            except Exception as e:
                print(f"Could not process {filename}. Error: {e}")
        
        print(f"\nTotal faces found across all photos: {len(event_embeddings)}")

        if event_embeddings:
            # Check if we have enough samples for clustering
            MIN_SAMPLES_FOR_CLUSTERING = 2
            if len(event_embeddings) >= MIN_SAMPLES_FOR_CLUSTERING:
                print("Running HDBSCAN clustering...")
                data = np.array(event_embeddings)
                data = normalize(data, norm='l2')
                clusterer = hdbscan.HDBSCAN(
                    min_cluster_size=2,
                    metric='euclidean',
                    cluster_selection_epsilon=0.5,
                    min_samples=2
                )
                cluster_labels = clusterer.fit_predict(data)
            else:
                print("Not enough photos to perform clustering. Treating as noise.")
                cluster_labels = [-1] * len(event_embeddings)
            
            # Save the data
            for i in range(len(event_file_paths)):
                person_id = f"{event_id}_{cluster_labels[i]}"
                embeddings_db.append({
                    "path": event_file_paths[i],
                    "embedding": event_embeddings[i],
                    "event": event_id,
                    "person_id": person_id
                })

        with open(CLUSTERED_EMBEDDINGS_FILE, "wb") as f:
            pickle.dump(embeddings_db, f)

        link = url_for("user_access", event_id=event_id, _external=True)
        return f"✅ Event uploaded and processed! Found {len(event_embeddings)} total faces. Share this link: <a href='{link}'>{link}</a>"

    return render_template("upload.html")
# --------------- User Access with Hybrid Cluster and Noise Search -----------------
@app.route("/event/<event_id>", methods=["GET", "POST"])
def user_access(event_id):
    if request.method == "POST":
        file = request.files["user_photo"]
        temp_path = os.path.join(TEMP_FOLDER, secure_filename(file.filename))
        file.save(temp_path)

        try:
            user_embedding = DeepFace.represent(img_path=temp_path, model_name="ArcFace", detector_backend="retinaface")[0]["embedding"]
        except:
            return "❌ No face detected in your photo. Try again."

        matches = []
        event_data = [item for item in embeddings_db if item.get("event") == event_id]
        
        representatives = {}
        noise_points = []
        
        for item in event_data:
            person_id = item.get("person_id")
            if person_id:
                if person_id.endswith("_-1"):
                    noise_points.append(item)
                elif person_id not in representatives:
                    representatives[person_id] = item["embedding"]
        
        found_person_id = None
        for person_id, rep_embedding in representatives.items():
            dist = DeepFace.verify(user_embedding, rep_embedding, model_name="ArcFace")
            if dist["verified"]:
                found_person_id = person_id
                break

        if found_person_id:
            for item in event_data:
                if item.get("person_id") == found_person_id:
                    matches.append(item["path"])
        
        if not matches and noise_points:
            print("No cluster match found, searching noise points...")
            for item in noise_points:
                dist = DeepFace.verify(user_embedding, item["embedding"], model_name="ArcFace")
                if dist["verified"]:
                    matches.append(item["path"])
                    break

        return render_template("results.html", matches=matches)

    return render_template("user_link.html", event_id=event_id)

# --------------- Download -----------------
@app.route("/download/<path:filename>")
def download_file(filename):
    return send_from_directory(directory=".", path=filename, as_attachment=True)

if __name__ == "__main__":
    app.run(debug=True)