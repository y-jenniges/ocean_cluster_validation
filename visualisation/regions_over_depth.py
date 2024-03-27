# adapted from: https://plotly.com/python/sliders/
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import netCDF4

# load data
drop_variant = "A1"  # A2 A3
assign_variant = "R1"  # R2 R3
df = pd.read_csv(f"../output_final/dbscan/post_processing/re-assigned_{drop_variant}_{assign_variant}.csv")

# idea: have a smooth plot, not scatter (we can use e.g. this nc file for it along with Basemap)
# nc = netCDF4.Dataset('../output_final/dbscan/post_processing/regions_A1.nc')

# plot settings
scatter_size = 2
margin = 5

# Create figure
fig = go.Figure(layout=go.Layout(title="Regions over depth",
                                 showlegend=False,
                                 # geo=dict(projection_type="natural earth")
                                 )
                )

# Add traces, one for each slider step
for step, depth in enumerate(np.sort(df.LEV_M.unique())):
    temp = df[df.LEV_M == depth]  # filter data for depth level
    fig.add_trace(
        go.Scattergeo(
            lat=temp.LATITUDE, lon=temp.LONGITUDE,
            visible=False,
            mode='markers',
            marker=dict(color=temp.color, opacity=1),
            name=str(step),
            hovertemplate='Longitude: %{lon} °<br>' +
                          'Latitude: %{lat} °<br>' +
                          'Temperature: %{text[0]:.2f} °C<br>' +
                          'Salinity: %{text[1]:.2f} psu<br>' +
                          'Oxygen: %{text[2]:.2f} µmol/kg<br>' +
                          'Nitrate: %{text[3]:.2f} µmol/kg<br>' +
                          'Silicate: %{text[4]:.2f} µmol/kg<br>' +
                          'Phosphate: %{text[5]:.2f} µmol/kg<br>' +
                          'Label: %{text[6]}<extra></extra>',
            text=temp[["P_TEMPERATURE", "P_SALINITY", "P_OXYGEN", "P_NITRATE", "P_SILICATE", "P_PHOSPHATE", "label"]]
        ))

# make 0th trace visible
fig.data[0].visible = True

# create and add slider
steps = []
for i in range(len(fig.data)):
    step = dict(
        method="update",
        args=[{"visible": [False] * len(fig.data)},
              {"title": "Regions over depth"}],  # layout attribute
        label=str(np.sort(df.LEV_M.unique())[i])
    )
    step["args"][0]["visible"][i] = True  # Toggle i'th trace to "visible"
    steps.append(step)

sliders = [dict(
    active=0,
    currentvalue={"prefix": "Depth: ", "suffix": " m"},
    pad={"t": 50},
    steps=steps
)]

fig.update_layout(
    sliders=sliders
)

# limit the displayed geographic area
fig.update_geos(
    lonaxis_range=[round(df.LONGITUDE.min()) - margin, round(df.LONGITUDE.max()) + margin],
    lataxis_range=[round(df.LATITUDE.min()) - margin, round(df.LATITUDE.max()) + margin])

fig.write_html(f"../output_final/dbscan/post_processing/regions_over_depth_{drop_variant}_{assign_variant}.html")
fig.show()
