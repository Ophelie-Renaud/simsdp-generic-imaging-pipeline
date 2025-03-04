import plotly.graph_objects as go
import numpy as np
from plotly.subplots import make_subplots
import plotly.io as pio 
import pandas as pd
from sklearn.metrics import mean_squared_error 

# Liste des fichiers CSV (simulation et mesure)
simulated_sota_csv_files = {
    "g2g_clean": "moldable/simu_sota/g2g_clean.csv",
    "g2g": "moldable/simu_sota/g2g.csv",
    "dft": "moldable/simu_sota/dft.csv",
    "fft": "moldable/simu_sota/fft.csv",
}
simulated_csv_files = {
    "g2g_clean": "moldable/simu/g2g_clean.csv",
    "g2g": "moldable/simu/g2g.csv",
    "dft": "moldable/simu/dft.csv",
    "fft": "moldable/simu/fft.csv",
}

measured_csv_files = {
    "g2g_clean": "moldable/measure/g2g_clean.csv",
    "g2g": "moldable/measure/g2g.csv",
    "dft": "moldable/measure/dft.csv",
    "fft": "moldable/measure/fft.csv",
}

instrumented = {
    "g2g_clean": 90154,
    "g2g": 90154,
    "dft": 97517,#valid
    "fft": 60122,#valid
}

# Données
num_minor_cycles = [50, 100, 150]
grid_size = np.array([512, 1024, 1536])
num_vis = np.array([1308160, 1962240, 2616320])

# Charger le fichier CSV
def load_simu_csv_to_numpy(file_path):
    df = pd.read_csv(file_path, sep=';')
    
    # Extraction des dimensions uniques
    grid_sizes = sorted(df['GRID_SIZE'].unique())
    num_cycles = sorted(df['NUM_MINOR_CYCLES'].unique())
    num_visibilities = sorted(df['NUM_VISIBILITIES'].unique())
    
    # Création d'un tableau numpy pour stocker les latences
    latency_simu = np.zeros((len(num_visibilities), len(num_cycles), len(grid_sizes)))
    
    # Remplissage du tableau
    for i, vis in enumerate(num_visibilities):
        for j, cycles in enumerate(num_cycles):
            for k, grid in enumerate(grid_sizes):
                value = df[(df['GRID_SIZE'] == grid) & (df['NUM_MINOR_CYCLES'] == cycles) & (df['NUM_VISIBILITIES'] == vis)]['DurationII'].values[0]
                latency_simu[i, j, k] = value
    
    return latency_simu

def load_mes_csv_to_numpy(file_path):
    df = pd.read_csv(file_path, sep=';')
    
    # Trier les données pour garantir un ordre cohérent
    df = df.sort_values(by=["NUM_VISIBILITIES", "NUM_MINOR_CYCLES", "GRID_SIZE"])
    
    # Obtenir les dimensions uniques
    visibilities = df["NUM_VISIBILITIES"].unique()
    minor_cycles = df["NUM_MINOR_CYCLES"].unique()
    grid_sizes = df["GRID_SIZE"].unique()

    # Construire un tableau numpy
    latency_measured = np.zeros((len(visibilities), len(minor_cycles), len(grid_sizes)))

    # Remplir le tableau
    for i, vis in enumerate(visibilities):
        for j, cycle in enumerate(minor_cycles):
            for k, grid in enumerate(grid_sizes):
                latency_measured[i, j, k] = df[(df["NUM_VISIBILITIES"] == vis) & 
                                                (df["NUM_MINOR_CYCLES"] == cycle) & 
                                                (df["GRID_SIZE"] == grid)]["Latency"].values[0]
    
    return latency_measured


def compute_rmse(simulated, measured):
    """ Calcule la RMSE entre les valeurs simulées et mesurées """
    simulated_flat = simulated.flatten() 
    measured_flat = measured.flatten()
    if len(simulated_flat)!= len(measured_flat):
        return 0
    return np.sqrt(mean_squared_error(simulated_flat, measured_flat))


def plot_3d_comparison(file_key,simu_sota_path, simu_path, measure_path, df_instrumented):

  latency_instru = np.full((len(grid_size), len(num_vis), len(num_minor_cycles)), df_instrumented)  
  print("latency_instru: " + str(latency_instru))
  latency_simu_sota = load_simu_csv_to_numpy(simu_sota_path)
  print("latency_simu_sota: " + str(latency_simu_sota))
  latency_simu = load_simu_csv_to_numpy(simu_path)
  print("latency_simu: " + str(latency_simu))
  latency_measured = load_mes_csv_to_numpy(measure_path)
  print("latency_measured: " + str(latency_measured))
  
  # Déterminer les min/max pour homogénéiser l'échelle des axes Z
  zmin = min(np.min(latency_instru), np.min(latency_simu_sota), np.min(latency_simu), np.min(latency_measured))
  zmax = max(np.max(latency_instru), np.max(latency_simu_sota), np.max(latency_simu), np.max(latency_measured))
  
  rmse_values = [compute_rmse(latency_instru,latency_measured), compute_rmse(latency_simu_sota,latency_measured), compute_rmse(latency_simu,latency_measured), compute_rmse(latency_measured,latency_measured)]
  error_values = [(rmse_values[0]/ np.mean(latency_measured))*100, (rmse_values[1]/ np.mean(latency_measured))*100, (rmse_values[2]/ np.mean(latency_measured))*100, (rmse_values[3]/ np.mean(latency_measured))*100]


  # Création des titres dynamiques
  subplot_titles = [
      f"Latency Instrumented<br>RMSE = {rmse_values[0]:.2f}<br>Error = {error_values[0]:.2f}%",
      f"Latency Simulated SOTA<br>RMSE = {rmse_values[1]:.2f}<br>Error = {error_values[1]:.2f}%",
      f"Latency Simulated<br>RMSE = {rmse_values[2]:.2f}<br>Error = {error_values[2]:.2f}%",
      f"Latency Measured<br>RMSE = {rmse_values[3]:.2f}<br>Error = {error_values[3]:.2f}%"
  ]

  # Création de la figure avec deux sous-graphiques côte à côte
  fig = make_subplots(
      rows=1, cols=4, 
      specs=[[{"type": "surface"}, {"type": "surface"}, {"type": "surface"}, {"type": "surface"}]], # Corrected specs
      subplot_titles=subplot_titles,
      horizontal_spacing=0.05
  )

  # Ajout des surfaces initiales
  fig.add_trace(go.Surface(
      x=grid_size, y=num_vis, z=latency_instru[0], 
      colorscale="Inferno", cmin=zmin, cmax=zmax
  ), row=1, col=1)

  fig.add_trace(go.Surface(
      x=grid_size, y=num_vis, z=latency_simu_sota[0], 
      colorscale="Inferno", cmin=zmin, cmax=zmax
  ), row=1, col=2)

  fig.add_trace(go.Surface(
      x=grid_size, y=num_vis, z=latency_simu[0], 
      colorscale="Inferno", cmin=zmin, cmax=zmax
  ), row=1, col=3)

  fig.add_trace(go.Surface(
      x=grid_size, y=num_vis, z=latency_measured[0], 
      colorscale="Inferno", cmin=zmin, cmax=zmax
  ), row=1, col=4
  )

  # Création des frames pour l'animation
  color = "Inferno"
  frames = []
  for i, cycle in enumerate(num_minor_cycles):
      frames.append(go.Frame(
          name=f"Cycle {cycle}",
          data=[
              go.Surface(z=latency_instru[i], x=grid_size, y=num_vis, colorscale=color, cmin=zmin, cmax=zmax),
              go.Surface(z=latency_simu_sota[i], x=grid_size, y=num_vis, colorscale=color, cmin=zmin, cmax=zmax),
              go.Surface(z=latency_simu[i], x=grid_size, y=num_vis, colorscale=color, cmin=zmin, cmax=zmax),
              go.Surface(z=latency_measured[i], x=grid_size, y=num_vis, colorscale=color, cmin=zmin, cmax=zmax)
          ]
      ))

  # Assign frames to the figure object directly
  fig.frames = frames

  # Ajout du slider
  sliders = [{
      "active": 0,
      "yanchor": "top",
      "xanchor": "left",
      "currentvalue": {"prefix": "Cycle: ", "font": {"size": 20}},
      "pad": {"b": 10, "t": 50},
      "len": 0.9,
      "x": 0.1,
      "y": 0,
      "steps": [
          {"args": [[f"Cycle {t}"], {"frame": {"duration": 500, "redraw": True}, "mode": "immediate"}],
          "label": str(t), "method": "animate"} for t in num_minor_cycles
      ]
  }]

  # Mise à jour du layout (remove frames from here)
  fig.update_layout(
      annotations=[dict(font=dict(size=10))],
      title=file_key,
      width=1600,  # Ajuste la largeur pour éviter le chevauchement
      height=400,  # Ajuste la hauteur
      margin=dict(l=0, r=0, t=100, b=40),  # Réduit les marges 
      scene=dict(xaxis_title="Grid Size", yaxis_title="Num Vis", zaxis_title="Latency", zaxis=dict(range=[zmin, zmax])),
      scene2=dict(xaxis_title="Grid Size", yaxis_title="Num Vis", zaxis_title="Latency", zaxis=dict(range=[zmin, zmax])),
      scene3=dict(xaxis_title="Grid Size", yaxis_title="Num Vis", zaxis_title="Latency", zaxis=dict(range=[zmin, zmax])),
      scene4=dict(xaxis_title="Grid Size", yaxis_title="Num Vis", zaxis_title="Latency", zaxis=dict(range=[zmin, zmax])),
      sliders=sliders
  )

  # Sauvegarde du premier frame en PDF
  pio.write_image(fig, f"3D_comparison_{file_key}.pdf", format="pdf", engine="kaleido") # Use write_image instead
  print(f"Graphique enregistré : 3D_comparison_{file_key}.pdf")
  
  # Sauvegarde du premier frame en png
  pio.write_image(fig, f"3D_comparison_{file_key}.png", format="png", engine="kaleido") # Use write_image instead
  print(f"Graphique enregistré : 3D_comparison_{file_key}.png")

  # Sauvegarde de l'animation complète en HTML
  fig.write_html(f"3D_comparison_{file_key}.html")
  print(f"Animation enregistrée : 3D_comparison_{file_key}.html")

  # Affichage
  #fig.show()

# Boucle sur chaque fichier pour comparer simulations et mesures
for key in simulated_csv_files.keys():
    plot_3d_comparison(key, simulated_sota_csv_files[key], simulated_csv_files[key], measured_csv_files[key],instrumented[key])
