import pickle
import os
import sys

CLUSTERED_EMBEDDINGS_FILE = "instance/embeddings_clustered.pkl"

def check_file_contents(target_event_id=None):
    if not os.path.exists(CLUSTERED_EMBEDDINGS_FILE):
        print(f"Error: The file '{CLUSTERED_EMBEDDINGS_FILE}' does not exist.")
        return

    try:
        with open(CLUSTERED_EMBEDDINGS_FILE, "rb") as f:
            data = pickle.load(f)

        if not data:
            print("Result: The data file is empty.")
            return
            
        # Filter data for a specific event if one is provided
        if target_event_id:
            data = [item for item in data if item.get("event") == target_event_id]
            if not data:
                print(f"Result: No data found for event ID '{target_event_id}'.")
                return

        total_faces = len(data)
        noise_count = 0
        people_clusters = set()

        for item in data:
            person_id = item.get("person_id")
            if person_id:
                if person_id.endswith("_-1"):
                    noise_count += 1
                else:
                    people_clusters.add(person_id)
        
        print("\n--- Data File Analysis ---")
        if target_event_id:
            print(f"--- Event: {target_event_id} ---")
        print(f"Total Faces Processed: {total_faces}")
        print(f"Unique People (Clusters): {len(people_clusters)}")
        print(f"Noise / Single Photos: {noise_count}")
        print("--------------------------\n")

        if len(people_clusters) == 0 and total_faces > 0:
            print("Diagnosis: The data for this event contains only noise points.")
            print("This is why the gallery page is blank.")
        elif len(people_clusters) > 0:
            print("Diagnosis: The data file looks good and contains valid clusters for this event.")

    except Exception as e:
        print(f"An error occurred while reading the file: {e}")

if __name__ == "__main__":
    # Check if an event ID was passed from the command line
    if len(sys.argv) > 1:
        event_id_to_check = sys.argv[1]
        check_file_contents(event_id_to_check)
    else:
        check_file_contents()