import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import os

# --- Configuration ---
EMBEDDINGS_FILE = "instance/embeddings.pkl"

def visualize_embeddings():
    """
    Loads face embeddings from a pickle file, uses t-SNE to reduce
    them to 2 dimensions, and creates a scatter plot visualization.
    """
    # --- Step 1: Load the Embeddings Data ---
    if not os.path.exists(EMBEDDINGS_FILE):
        print(f"Error: Embeddings file not found at '{EMBEDDINGS_FILE}'")
        return

    print("Loading embeddings from file...")
    with open(EMBEDDINGS_FILE, "rb") as f:
        embeddings_db = pickle.load(f)

    if not embeddings_db:
        print("The embeddings file is empty. No data to visualize.")
        return

    # --- Step 2: Prepare the Data for t-SNE ---
    embeddings_list = [item["embedding"] for item in embeddings_db]
    event_labels = [item["event"] for item in embeddings_db]
    embeddings_array = np.array(embeddings_list)
    
    print(f"Found {len(embeddings_array)} face embeddings to visualize.")

    # --- Step 3: Perform Dimensionality Reduction with t-SNE ---
    print("Performing t-SNE dimensionality reduction (this may take a minute)...")
    tsne = TSNE(n_components=2, perplexity=30, random_state=42, n_iter=1000)
    reduced_embeddings = tsne.fit_transform(embeddings_array)
    print("t-SNE complete.")

    # --- Step 4: Create the Visualization ---
    plt.figure(figsize=(14, 10))
    
    unique_events = sorted(list(set(event_labels)))
    colors = plt.cm.jet(np.linspace(0, 1, len(unique_events)))
    event_color_map = {event: color for event, color in zip(unique_events, colors)}

    for i, event in enumerate(unique_events):
        indices = [idx for idx, label in enumerate(event_labels) if label == event]
        plt.scatter(
            reduced_embeddings[indices, 0], 
            reduced_embeddings[indices, 1], 
            color=event_color_map[event],
            label=f'Event: {event}'
        )

    plt.title('2D Visualization of Face Embeddings using t-SNE')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.legend()
    plt.grid(True)
    
    print("Displaying plot. Close the plot window to exit.")
    plt.show()

# ===============================================================
#  THIS IS THE CRUCIAL PART THAT WAS LIKELY MISSING
#  It tells Python to run the visualize_embeddings function
#  when the script is executed directly.
# ===============================================================
if __name__ == "__main__":
    visualize_embeddings()