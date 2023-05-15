
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.graph_objs as go
import plotly.express as px
from plotly.subplots import make_subplots

# Data preprocessing
url = "https://raw.githubusercontent.com/vodzillaOG/VG-sales/main/vgsales.csv"
df = pd.read_csv(url).drop_duplicates()

# Start the app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server

# Build the components
header_component = html.H1("VIDEOGAME SALES ANALYSIS", style={'color': 'darkcyan'})

REGION_OPTIONS = [
    {"label": "Global Sales", "value": "Global_Sales"},
    {"label": "North America Sales", "value": "NA_Sales"},
    {"label": "Europe Sales", "value": "EU_Sales"},
    {"label": "Japan Sales", "value": "JP_Sales"},
    {"label": "Other Regions Sales", "value": "Other_Sales"},
]


# Visual components
# 1 - TREEMAP
total_sales = df[['JP_Sales', 'EU_Sales', 'NA_Sales', 'Other_Sales']].sum()
total_sales_billion = total_sales / 1000
labels = ['JAPAN', 'EUROPE', 'NORTH AMERICA', 'OTHER REGIONS']
colors = ['red', 'green', 'blue', 'grey']

treemap = go.Figure(go.Treemap(
    labels=labels,
    parents=['', '', '', ''],
    values=total_sales_billion,
    texttemplate="%{label}<br>Total Sales: %{value:.2f}B$<br>Percentage: %{percentParent:.2%}",
    textinfo='label+value+percent parent',
    textposition='middle center',
    marker=dict(line=dict(color='black', width=1), colors=colors)
)).update_layout(title='TOTAL SALES BY REGION')



# Component 2
df['Year'] = pd.to_datetime(df['Year'], format='%Y')
sales_by_year_region = df.groupby(df['Year'].dt.year)[['NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales']].sum().reset_index()
sales_by_year_region = pd.melt(sales_by_year_region, id_vars='Year', value_vars=['NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales'],
                               var_name='Sales Region', value_name='Total Sales (Millions)')

sales_by_year_region['Sales Region'] = sales_by_year_region['Sales Region'].map({
    'NA_Sales': 'NORTH AMERICA',
    'EU_Sales': 'EUROPE',
    'JP_Sales': 'JAPAN',
    'Other_Sales': 'OTHER REGIONS'
})

fig = px.line(sales_by_year_region, x='Year', y='Total Sales (Millions)', color='Sales Region',
              title='SALES EVOLUTION OVER TIME',
              color_discrete_map={'NORTH AMERICA': 'blue', 'EUROPE': 'green', 'JAPAN': 'red', 'OTHER REGIONS': 'grey'})

fig.update_layout(xaxis_title='Year', yaxis_title='Total Sales (Millions)',
                  plot_bgcolor='white', showlegend=True,
                  legend=dict(bgcolor='white', title=dict(text='Sales Region')),
                  xaxis=dict(showgrid=True, gridcolor='white', linecolor='black', linewidth=3),
                  yaxis=dict(showgrid=True, gridcolor='white', linecolor='black', linewidth=3))

max_values = sales_by_year_region.groupby('Sales Region')['Total Sales (Millions)'].max()
for region, value in max_values.items():
    df_region = sales_by_year_region[sales_by_year_region['Sales Region'] == region]
    max_year = df_region[df_region['Total Sales (Millions)'] == value]['Year'].values[0]
    fig.add_trace(go.Scatter(
        x=[max_year],
        y=[value],
        mode='markers+text',
        marker=dict(color='black', size=8),
        text=[f'{value:.2f}M'],
        textposition='top right',
        textfont=dict(size=10),
        showlegend=False,
        legendgroup=region,
        visible=True,
        name=region
    ))

unique_years = sales_by_year_region['Year'].unique()
for year in unique_years:
    df_year = sales_by_year_region[sales_by_year_region['Year'] == year]
    for index, row in df_year.iterrows():
        region = row['Sales Region']
        value = row['Total Sales (Millions)']
        fig.add_trace(go.Scatter(
            x=[year],
            y=[value],
            mode='markers',
            marker=dict(size=6, color='black'),
            showlegend=False,
            legendgroup=region,
            visible=True,
            name=region
        ))
        
# Update the legend click behavior
fig.for_each_trace(
    lambda trace: trace.on_click(
        lambda trace, points, selector: [
            t.update(visible=not t.visible) for t in fig.data if t.legendgroup == trace.name
        ]
    )
)

# component 3
genre_sales = df.groupby('Genre')['Global_Sales'].sum().sort_values(ascending=False)

fig2 = go.Figure(data=[go.Bar(x=genre_sales.index, y=genre_sales.values)])

fig2.update_layout(
    title='TOTAL SALES BY GENRE',
    xaxis=dict(title='Genre'),
    yaxis=dict(title='Total Sales (in millions)'),
    barmode='relative',
    bargap=0.1,
    bargroupgap=0.1,
    plot_bgcolor='white',  # Set the plot background color to white
    paper_bgcolor='white'  # Set the paper background color to white
)

# component 4

# Select the top 3 'Platform' for JP, NA, and EU regions
top_platform_jp = df[df['JP_Sales'] > 0].groupby('Platform')['JP_Sales'].sum().nlargest(5)
top_platform_na = df[df['NA_Sales'] > 0].groupby('Platform')['NA_Sales'].sum().nlargest(5)
top_platform_eu = df[df['EU_Sales'] > 0].groupby('Platform')['EU_Sales'].sum().nlargest(5)

# Create the subplots figure for the pie charts
fig3 = make_subplots(rows=1, cols=3, subplot_titles=['JAPAN', 'NORTH AMERICA', 'EUROPE'],
                     specs=[[{'type': 'pie'}, {'type': 'pie'}, {'type': 'pie'}]])

fig3.add_trace(go.Pie(labels=top_platform_jp.index, values=top_platform_jp.values,
                      name='JAPAN', textinfo='percent', hoverinfo='label+value+percent',
                      textfont=dict(size=10),
                      marker=dict(line=dict(color='black', width=1))),
               row=1, col=1)

# Create the pie chart for NA region
fig3.add_trace(go.Pie(labels=top_platform_na.index, values=top_platform_na.values,
                      name='NORTH AMERICA', textinfo='percent', hoverinfo='label+value+percent',
                      textfont=dict(size=10),
                      marker=dict(line=dict(color='black', width=1))),
               row=1, col=2)

# Create the pie chart for EU region
fig3.add_trace(go.Pie(labels=top_platform_eu.index, values=top_platform_eu.values,
                      name='EUROPE', textinfo='percent', hoverinfo='label+value+percent',
                      textfont=dict(size=10),
                      marker=dict(line=dict(color='black', width=1))),
               row=1, col=3)

# Update the layout of the figure
fig3.update_layout(title='TOP PLATFORMS BY REGION')



# component 5
desired_platforms = ['PS4', 'XOne', 'PC', 'WiiU', '3DS']
filtered_df = df[df['Platform'].isin(desired_platforms)]

def get_top_10_games(region):
    top_10_games = filtered_df.groupby('Name')[region].sum().nlargest(10).reset_index()
    top_10_games = top_10_games.sort_values(by=region, ascending=True)  # Sort by the specified region
    return top_10_games

desired_regions = ['Global_Sales', 'JP_Sales', 'NA_Sales', 'EU_Sales', 'Other_Sales']
region_labels = ['GLOBAL', 'JAPAN', 'NORTH AMERICA', 'EUROPE', 'OTHER REGIONS']  # Updated region labels
initial_region = desired_regions[0]
top_10_games = get_top_10_games(initial_region)

fig5 = px.bar(top_10_games, x=initial_region, y='Name', orientation='h',
              title=f'TOP 10 GAMES SOLD',
              labels={initial_region: 'Total Sales (in millions)'},
              text=initial_region)  # Set the text parameter to the sales value
              

dropdown_buttons = [
    dict(label=label, method="update",  # Use the updated label
         args=[{"x": [get_top_10_games(region)[region]],
                "y": [get_top_10_games(region)['Name']]},
               {"title": f"TOP 10 GAMES SOLD"}])  # Use the updated label
    for region, label in zip(desired_regions, region_labels)
]

fig5.update_layout(
    plot_bgcolor='white',  # Set the plot background color to white
    paper_bgcolor='white',  # Set the paper background color to white
    updatemenus=[
        dict(
            buttons=dropdown_buttons,
            direction="down",
            pad={"r": 10, "t": 10},
            showactive=True,
            x=1.5,
            xanchor="right",
            y=1.04,
            yanchor="top"
        ),
    ]
)

fig5.update_traces(hovertemplate='%{text:.0f}M$', texttemplate='%{text:.0f}M$')







# Design the layout
app.layout = html.Div([
    dbc.Row([header_component]),
    dbc.Row([
        dbc.Col([
            dcc.Graph(id='platform-pie-charts', figure=fig3)
        ], width=6),  # Specify the width of the column to take up half of the row
        dbc.Col([
            dcc.Graph(
        id='bar-chart',
        figure=fig5)
        ], width=6),  # Specify the width of the column to take up half of the row
    ]),
    dbc.Row([dbc.Col([dcc.Graph(figure=fig)])]),
    dbc.Row([
        dbc.Col([dcc.Graph(figure=treemap)]),
        dbc.Col([
            dcc.Dropdown(id="region-dropdown", options=REGION_OPTIONS, value="Global_Sales"),
            dcc.Graph(id="genre-sales-bar-chart")
        ])
    ])
])






# callbacks
@app.callback(
    Output("genre-sales-bar-chart", "figure"),
    Input("region-dropdown", "value")
)
def update_genre_sales_bar_chart(selected_region):
    genre_sales = df.groupby('Genre')[selected_region].sum().sort_values(ascending=False)

    fig2 = go.Figure(data=[go.Bar(x=genre_sales.index, y=genre_sales.values)])

    fig2.update_layout(
        title=f'TOTAL SALES BY GENRE',
        xaxis=dict(title='Genre'),
        yaxis=dict(title='Total Sales (in millions)'),
        barmode='relative',
        bargap=0.1,
        bargroupgap=0.1,
        plot_bgcolor='white',  # Set the plot background color to white
        paper_bgcolor='white'  # Set the paper background color to white
    )

    return fig2









# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
