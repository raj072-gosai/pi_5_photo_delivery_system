import pickle
import numpy as np
import pandas as pd
import os
import base64

from sklearn.manifold import TSNE
import plotly.express as px
import dash
from dash import dcc, html
from dash.dependencies import Input, Output

# --- Configuration ---
EMBEDDINGS_FILE = "instance/embeddings_clustered.pkl"

# --- Step 1: Load and Prepare Data ---
print("Loading and preparing data...")
with open(EMBEDDINGS_FILE, "rb") as f:
    embeddings_db = pickle.load(f)

# Convert data to a Pandas DataFrame for easier use with Plotly
df = pd.DataFrame(embeddings_db)

# Perform t-SNE reduction
embeddings_array = np.array(df["embedding"].tolist())
tsne = TSNE(n_components=2, perplexity=30, random_state=42, n_iter=1000)
reduced_embeddings = tsne.fit_transform(embeddings_array)

# Add t-SNE results back to the DataFrame
df['tsne_x'] = reduced_embeddings[:, 0]
df['tsne_y'] = reduced_embeddings[:, 1]

# Make the noise points more readable in the legend
df['display_id'] = df['person_id'].apply(lambda x: 'Noise / Single Photos' if x.endswith('_-1') else x)

print("Data preparation complete. Starting Dash server...")

# --- Step 2: Create the Interactive Plot with Plotly ---
fig = px.scatter(
    df,
    x='tsne_x',
    y='tsne_y',
    color='display_id',
    custom_data=['path'],  # Pass the image path to be used in the callback
    title="Interactive Visualization of HDBSCAN Clusters"
)
fig.update_traces(
    hoverinfo='none',      # Disable default hover info
    hovertemplate=None     # Disable the hover box
)
fig.update_layout(legend_title_text='Person ID')


# --- Step 3: Build the Dash Web Application ---
app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("Face Cluster Visualization"),
    html.Div([
        # The main graph
        dcc.Graph(
            id='cluster-graph', 
            figure=fig,
            style={'width': '60%', 'display': 'inline-block'}
        ),
        # The container for the image preview
        html.Div(
            html.Img(id='hover-image', src='', style={'width': '100%'}),
            style={'width': '35%', 'display': 'inline-block', 'vertical-align': 'top', 'padding-left': '20px'}
        )
    ])
])


# --- Step 4: Define the Interactivity (The Callback) ---
@app.callback(
    Output('hover-image', 'src'),
    Input('cluster-graph', 'hoverData')
)
def display_hover_image(hoverData):
    if hoverData is None:
        return "" # Return empty string if not hovering

    # Get the image path from the custom_data of the hovered point
    image_path = hoverData['points'][0]['customdata'][0]

    # Read the image file and encode it in Base64
    with open(image_path, 'rb') as f:
        encoded_image = base64.b64encode(f.read()).decode()

    # Return the image as a Data URI
    return f'data:image/png;base64,{encoded_image}'


# --- Step 5: Run the Application ---
if __name__ == '__main__':
    app.run(debug=True)