import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import os

# --- Configuration ---
# Make sure this points to your clustered embeddings file
EMBEDDINGS_FILE = "instance/embeddings_clustered.pkl"

def visualize_clustered_embeddings():
    """
    Loads clustered face embeddings, uses t-SNE to reduce them to 2D,
    and creates a scatter plot coloring points by their HDBSCAN person_id.
    """
    # --- Step 1: Load the Embeddings Data ---
    if not os.path.exists(EMBEDDINGS_FILE):
        print(f"Error: Embeddings file not found at '{EMBEDDINGS_FILE}'")
        return

    print("Loading clustered embeddings from file...")
    with open(EMBEDDINGS_FILE, "rb") as f:
        embeddings_db = pickle.load(f)

    if not embeddings_db:
        print("The embeddings file is empty. No data to visualize.")
        return

    # --- Step 2: Prepare Data, now using person_id for labels ---
    embeddings_list = [item["embedding"] for item in embeddings_db]
    # Get the person_id for each embedding to use as our label
    person_id_labels = [item.get("person_id", "N/A") for item in embeddings_db]
    
    embeddings_array = np.array(embeddings_list)
    print(f"Found {len(embeddings_array)} face embeddings to visualize.")

    # --- Step 3: Perform Dimensionality Reduction with t-SNE ---
    print("Performing t-SNE dimensionality reduction (this may take a minute)...")
    tsne = TSNE(n_components=2, perplexity=30, random_state=42, n_iter=1000)
    reduced_embeddings = tsne.fit_transform(embeddings_array)
    print("t-SNE complete.")

    # --- Step 4: Create Visualization colored by Person ID ---
    plt.figure(figsize=(16, 12))
    
    unique_person_ids = sorted(list(set(person_id_labels)))
    
    # Separate the actual clusters from the noise points
    cluster_ids = [pid for pid in unique_person_ids if not pid.endswith("_-1")]
    noise_id = next((pid for pid in unique_person_ids if pid.endswith("_-1")), None)

    # Generate a color for each unique person/cluster
    colors = plt.cm.tab20(np.linspace(0, 1, len(cluster_ids)))
    person_color_map = {pid: color for pid, color in zip(cluster_ids, colors)}

    # Plot each cluster with its assigned color
    for person_id, color in person_color_map.items():
        indices = [idx for idx, label in enumerate(person_id_labels) if label == person_id]
        plt.scatter(
            reduced_embeddings[indices, 0], 
            reduced_embeddings[indices, 1], 
            color=color,
            label=person_id # Adding label for clarity if needed, can be removed for less clutter
        )
    
    # Plot the noise points in a distinct way (black 'x')
    if noise_id:
        noise_indices = [idx for idx, label in enumerate(person_id_labels) if label == noise_id]
        plt.scatter(
            reduced_embeddings[noise_indices, 0], 
            reduced_embeddings[noise_indices, 1], 
            color='black',
            marker='x',
            s=50, # size
            label='Noise / Single Photos'
        )

    plt.title('2D Visualization of HDBSCAN Clusters using t-SNE')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    # Optional: To avoid a cluttered legend if there are many people, you can comment out the next line
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.grid(True)
    plt.tight_layout(rect=[0, 0, 0.85, 1]) # Adjust layout to make room for legend
    
    print("Displaying plot. Close the plot window to exit.")
    plt.show()

if __name__ == "__main__":
    visualize_clustered_embeddings()