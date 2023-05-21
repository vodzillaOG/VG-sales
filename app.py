
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots


# Data preprocessing
url = "https://raw.githubusercontent.com/vodzillaOG/VG-sales/main/vgsales.csv"
df = pd.read_csv(url).drop_duplicates()

# Start the app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server

# Building some components
header_component = html.H1(
    "PlayMetrics - videogame sales analysis",
    style={
        'color': '#D9ED92',
        'text-align': 'center'
    }
)

signature_component = html.H2(
    ["Data Visualization Project", html.Br(), "2022-2023",html.Br(), "Volodymyr Mykhayliv (20221336)"],
    style={
        'color': '#D9ED92',
        'text-align': 'left',
        'font-size': '15px'
    }
)


REGION_OPTIONS = [
    {"label": "Global Sales", "value": "Global_Sales"},
    {"label": "North America Sales", "value": "NA_Sales"},
    {"label": "Europe Sales", "value": "EU_Sales"},
    {"label": "Japan Sales", "value": "JP_Sales"},
    {"label": "Other Regions Sales", "value": "Other_Sales"},
]

# Building visual components
# COMPONENT 1 - TREEMAP
total_sales = df[['JP_Sales', 'EU_Sales', 'NA_Sales', 'Other_Sales']].sum()
total_sales_billion = total_sales / 1000
labels = ['JAPAN', 'EUROPE', 'NORTH AMERICA', 'OTHER REGIONS']
colors = ['#34A0A4', '#1A759F', '#184E77', '#76C893']

treemap = go.Figure(go.Treemap(
    labels=labels,
    parents=['', '', '', ''],
    values=total_sales_billion,
    texttemplate="%{label}<br>Total Sales: %{value:.2f}B$<br> %{percentParent:.2%}",
    textinfo='label+value+percent parent',
    textposition='middle center',
    marker=dict(line=dict(color='#D9ED92', width=1), colors=colors),  # Set the line color to #D9ED92
    textfont=dict(color='black', size=14)
)).update_layout(
    title={
        'text': 'TOTAL SALES DISTRIBUTION',
        'x': 0.5,
        'font': {'color': '#D9ED92', 'size': 30},
        'xanchor': 'center',
        'yanchor': 'top'
    },
    plot_bgcolor='black',
    paper_bgcolor='black'
)

# COMPONENT 2 - STACKED AREA CHART
sales_by_region = df.groupby('Year')[['JP_Sales', 'NA_Sales', 'EU_Sales', 'Other_Sales', 'Global_Sales']].sum().reset_index()

filtered_data = sales_by_region[(sales_by_region['Year'] >= 2000) & (sales_by_region['Year'] <= 2015)]

total_sales = filtered_data[['JP_Sales', 'NA_Sales', 'EU_Sales', 'Other_Sales', 'Global_Sales']].sum().sort_values(ascending=False).index
colors = ['#184E77', '#1A759F', '#34A0A4', '#76C893', '#B5E48C']

legend_labels = {
    'JP_Sales': 'Japan',
    'NA_Sales': 'North America',
    'EU_Sales': 'Europe',
    'Other_Sales': 'Other Regions',
    'Global_Sales': 'Global Sales'
}
fig = go.Figure()

for region in ['JP_Sales', 'NA_Sales', 'EU_Sales', 'Other_Sales', 'Global_Sales']:
    fig.add_trace(go.Scatter(
        x=filtered_data['Year'],
        y=filtered_data[region],
        fill='tozeroy',
        name=legend_labels[region],
        line=dict(color=colors[total_sales.get_loc(region)], width=0.5),
        hovertemplate='%{y:.2f} mm',
        marker=dict(color='#D9ED92')
    ))

fig.update_layout(
    plot_bgcolor='rgba(0,0,0)',
    paper_bgcolor='black',
    title={
        'text': 'SALES EVOLUTION OVER TIME (2000-2015)',
        'x': 0.5,
        'font': {'size': 30, 'color': '#D9ED92'},
        'xanchor': 'center'
    },
    xaxis=dict(gridcolor='black', title='Year', color='#D9ED92', tickmode='linear', dtick=1, rangeslider=dict(visible=True)),
    yaxis=dict(showline=True, linecolor='#D9ED92', gridcolor='black', color='#D9ED92', title='Total Sales (mm)'),
    legend=dict(
        traceorder='reversed',
        font=dict(
            size=16,
            color='#D9ED92'
        )
    )
)

# COMPONENT 3 - BAR CHART - GENRE PREFERENCE
genre_sales = df.groupby('Genre')['Global_Sales'].sum().sort_values(ascending=False)

fig2 = go.Figure(data=[go.Bar(x=genre_sales.index, y=genre_sales.values)])

# COMPONENT 4 - STACKED BAR CHART
def get_top_platforms(region_sales_column):
    top_platforms = df.groupby('Platform')[region_sales_column].sum().nlargest(5)
    return top_platforms

na_top_platforms = get_top_platforms('NA_Sales')
eu_top_platforms = get_top_platforms('EU_Sales')
jp_top_platforms = get_top_platforms('JP_Sales')

platforms = list(na_top_platforms.index)

fig3 = go.Figure()

fig3.add_trace(go.Bar(
    x=platforms,
    y=na_top_platforms.values,
    name='North America',
    marker_color='#184E77',
    text=na_top_platforms.values,
    texttemplate='%{text:.2f}mm$',
    textposition='inside',
    insidetextanchor='middle',
    textfont=dict(color='black'),
    marker=dict(line=dict(color='#D9ED92', width=1))
))

fig3.add_trace(go.Bar(
    x=platforms,
    y=eu_top_platforms.values,
    name='Europe',
    marker_color='#168AAD',
    text=eu_top_platforms.values,
    texttemplate='%{text:.2f}mm$',
    textposition='inside',
    insidetextanchor='middle',
    textfont=dict(color='black'),
    marker=dict(line=dict(color='#D9ED92', width=1))
))

fig3.add_trace(go.Bar(
    x=platforms,
    y=jp_top_platforms.values,
    name='Japan',
    marker_color='#76C893',
    text=jp_top_platforms.values,
    texttemplate='%{text:.2f}mm$',
    textposition='inside',
    insidetextanchor='middle',
    textfont=dict(color='black'),
    marker=dict(line=dict(color='#D9ED92', width=1))
))

fig3.update_layout(
    barmode='relative',
    title_text='MOST POPULAR PLATFORMS IN EVERY REGION',
    xaxis_title='Platform',
    yaxis_title='Sales (mm$)',
    plot_bgcolor='black',
    paper_bgcolor='black',
    xaxis=dict(
        tickfont=dict(color='#D9ED92'),
        title=dict(text='Platform', font=dict(color='#D9ED92')),
        gridcolor='black',
        linecolor='black'
    ),
    yaxis=dict(
        tickfont=dict(color='#D9ED92'),
        title=dict(text='Sales (mm$)', font=dict(color='#D9ED92')),
        gridcolor='black'
    ),
    font=dict(color='#D9ED92'),
    title=dict(text='MOST POPULAR PLATFORMS IN EVERY REGION', font=dict(color='#D9ED92', size=30)),
    legend=dict(
        font=dict(
            size=16
        )
    )
)

# COMPONENT 5 - SALES HORIZONTAL BAR CHART
def get_top_10_games(df, platforms, region):
    filtered_df = df[df['Platform'].isin(platforms)]
    top_10_games = filtered_df.groupby('Name')[region].sum().nlargest(10).reset_index()
    top_10_games = top_10_games.sort_values(by=region, ascending=False)
    return top_10_games

desired_platforms = ['PS4', 'XOne', 'PC', 'WiiU', '3DS']
desired_regions = ['Global_Sales', 'JP_Sales', 'NA_Sales', 'EU_Sales', 'Other_Sales']
region_labels = ['GLOBAL', 'JAPAN', 'NORTH AMERICA', 'EUROPE', 'OTHER REGIONS']
initial_region = desired_regions[0]

top_10_games = get_top_10_games(df, desired_platforms, initial_region)

fig5 = go.Figure()

bar_colors = ['#184E77', '#1E6091', '#1A759F', '#168AAD', '#34A0A4', '#52B69A', '#76C893', '#99D98C', '#B5E48C', '#D9ED92']

fig5.add_trace(go.Bar(
    x=top_10_games[initial_region][::1],
    y=top_10_games['Name'][::1],
    orientation='h',
    text=top_10_games[initial_region][::1],
    texttemplate='%{text:.2f}mm$',
    textposition='inside',
    marker=dict(
        color=bar_colors,
    ),
    textfont=dict(color='black', size=14),
))

fig5.update_traces(marker_line_color='#D9ED92')

fig5.update_layout(
    title=dict(
        text='TOP 10 BESTSELLING GAMES (mm$)',
        x=0.5,
        font=dict(color='#D9ED92', size=30)
    ),
    xaxis=dict(title='', tickfont=dict(color='#D9ED92'), gridcolor='black', title_font=dict(color='#D9ED92'), showticklabels=False),
    yaxis=dict(autorange="reversed", tickfont=dict(color='#D9ED92'), gridcolor='black', linecolor='black'),
    plot_bgcolor='black',
    paper_bgcolor='black',
)

# Design the layout
app.layout = html.Div(
    style={'backgroundColor': '#D9ED92'},
    children=[
        dbc.Row([header_component], style={
            'background-color': 'black',
            'border-radius': '20px',
            'padding': '20px',
            'margin': '10px'
        }),
        dbc.Row([
            dbc.Col([
                dcc.Graph(id='bar-chart', figure=fig5)
            ], width=6),
            dbc.Col([
                dcc.Graph(id='platform-pie-charts', figure=fig3)
            ], width=6, )
        ], style={
            'background-color': 'black',
            'border-radius': '20px',
            'padding': '20px',
            'margin': '10px'
        }),
        dbc.Row([dbc.Col([dcc.Graph(figure=fig)])], style={
            'background-color': 'black',
            'border-radius': '15px',
            'padding': '0px',
            'margin': '10px'
        }),
        dbc.Row([
            dbc.Col([dcc.Graph(figure=treemap)]),
            dbc.Col([
    dcc.Graph(id="genre-sales-bar-chart"),
    dcc.RadioItems(
    id="region-radio",
    options=REGION_OPTIONS,
    value="Global_Sales",
    labelStyle={'display': 'inline-block', 'margin-right': '20px'}, 
    style={'color': '#D9ED92', 'text-align': 'center'}
),

])
        ], style={
            'background-color': 'black',
            'border-radius': '15px',
            'padding': '15px',
            'margin': '10px'
        }),
        dbc.Row([signature_component], style={
            'background-color': 'black',
            'border-radius': '15px',
            'padding': '20px',
            'margin': '10px'
        }),
    ]
)

# CALLBACKS - BAR CHART - GENRE PREFERENCE
@app.callback(
    Output("genre-sales-bar-chart", "figure"),
    Input("region-radio", "value")
)
def update_genre_sales_bar_chart(selected_region):
    genre_sales = df.groupby('Genre')[selected_region].sum().sort_values(ascending=False)

    colors = ['#D9ED92', '#B5E48C', '#99D98C', '#76C893', '#52B69A', '#34A0A4', '#168AAD', '#1A759F', '#1E6091', '#184E77']
    num_colors = len(colors)

    normalized_sales = (genre_sales - genre_sales.min()) / (genre_sales.max() - genre_sales.min())
    bar_colors = [colors[int((num_colors - 1) * value)] for value in normalized_sales]

    fig2 = go.Figure(data=[go.Bar(x=genre_sales.index, y=genre_sales.values, marker=dict(color=bar_colors))])

    fig2.update_layout(
    title={
        'text': 'GAME TYPE PREFERENCE',
        'x': 0.5,
        'font': {'color': '#D9ED92', 'size': 30},
        'xanchor': 'center',
        'yanchor': 'top'
    },
    xaxis=dict(
        title='Type',
        title_font={'color': '#D9ED92'},
        tickfont={'color': '#D9ED92'},
        gridcolor='black'
    ),
    yaxis=dict(
        title='Total Sales (mm$)',
        title_font={'color': '#D9ED92'},
        tickfont={'color': '#D9ED92'},
        gridcolor='black'
    ),
    barmode='relative',
    bargap=0.1,
    bargroupgap=0.1,
    plot_bgcolor='black',
    paper_bgcolor='black'
)

    fig2.update_traces(
    texttemplate='%{y:.0f}mm$',
    textposition='auto',
    textangle=0,
    textfont=dict(color='black', size=12)
)
    return fig2


# Running the APP
if __name__ == '__main__':
    app.run_server(debug=True)