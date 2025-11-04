import os
import pickle
from flask import Flask, render_template

CLUSTERED_EMBEDDINGS_FILE = "instance/embeddings_clustered.pkl"

app = Flask(__name__)

# --- Main Gallery Route ---
@app.route("/")
def gallery():
    """
    Loads the clustered data and prepares it for the main gallery view.
    It finds one representative photo for each person.
    """
    if not os.path.exists(CLUSTERED_EMBEDDINGS_FILE):
        return "Error: Clustered embeddings file not found. Please run the processing app first."

    with open(CLUSTERED_EMBEDDINGS_FILE, "rb") as f:
        embeddings_db = pickle.load(f)

    people = {}
    # Group photos by person_id
    for item in embeddings_db:
        person_id = item.get("person_id")
        # Skip noise points for the main gallery
        if person_id and not person_id.endswith("_-1"):
            if person_id not in people:
                # This is the first time we've seen this person.
                # Use their photo as the "cover photo" and start their count.
                people[person_id] = {
                    "person_id": person_id,
                    "cover_photo": item["path"],
                    "photo_count": 0,
                    "all_photos": set() # Use a set to store unique photo paths
                }
            # Add the photo to their collection
            people[person_id]["all_photos"].add(item["path"])

    # Update the photo count for each person based on unique photos
    for person_id in people:
        people[person_id]["photo_count"] = len(people[person_id]["all_photos"])
    
    # Convert the dictionary to a list for easier templating
    people_list = sorted(list(people.values()), key=lambda x: x['person_id'])

    return render_template("gallery.html", people=people_list)


# --- Person Detail / Album Route ---
@app.route("/person/<person_id>")
def person_album(person_id):
    """
    Shows all the photos for a single selected person.
    """
    if not os.path.exists(CLUSTERED_EMBEDDINGS_FILE):
        return "Error: Clustered embeddings file not found."

    with open(CLUSTERED_EMBEDDINGS_FILE, "rb") as f:
        embeddings_db = pickle.load(f)

    # Find all photos for the given person_id
    person_photos = set()
    for item in embeddings_db:
        if item.get("person_id") == person_id:
            person_photos.add(item["path"])

    return render_template("person_detail.html", person_id=person_id, photos=sorted(list(person_photos)))


if __name__ == "__main__":
    app.run(debug=True)