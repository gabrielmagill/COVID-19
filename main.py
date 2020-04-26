# Import required libraries
import sys
import pathlib
#root=str(pathlib.Path(__file__).parent.absolute())+'/../'
root = './'
sys.path.append(root+'include/')

import pickle
import copy
import dash
import math
import datetime as dt
import pandas as pd
from dash.dependencies import Input, Output, State, ClientsideFunction
import dash_core_components as dcc
import dash_html_components as html

import plotly.graph_objects as go
import chart_studio.plotly as py  
import plotly.tools as tls   
from plotly.graph_objs import *




app = dash.Dash(
    __name__, meta_tags=[{"name": "viewport", "content": "width=device-width"}], assets_folder='assets'
)
server = app.server


observables=[
{'label':'Active Cases','value':'active'},
{'label':'Total Cases','value':'confirmed'},
{'label':'Total  Recoveries','value':'recovered'},
{'label':'Total Deaths','value':'death'},
{'label':'New Cases','value':'new cases'},
{'label':'Doubling Rate','value':'doubling_rate'}
]
observables_dict={}
for o in observables:
        observables_dict[o['value']]=o['label']


df_main=pd.read_csv(root+'output/temp/df_main.csv')
df_ref=pd.read_csv(root+'output/temp/df_ref.csv')

df_world=df_main.groupby(['dates']).agg({'confirmed':sum,'recovered':sum,'active':sum,'death':sum,'days':max,'new cases':sum}).reset_index()
df_main = df_main.append(df_ref[df_ref['region'].isin(['Doubles 5 days'])],sort=False)

max_days = max(df_world['days'])
min_days = min(df_world['days'])
max_datetime=min(df_world[df_world['days']==max_days]['dates'])
min_datetime=min(df_world[df_world['days']==min_days]['dates'])

clist = df_main['region'].unique()
clist.sort()
country_list = [{'label':c,'value':c} for c in clist ]



# Create app layout
app.layout = html.Div(
    [
        dcc.Store(id="aggregate_data"),
        # empty Div to trigger javascript file for graph resizing
        html.Div(id="output-clientside"),
        html.Div([
        	html.Div([
                	html.H3(
                        	"Covid-19",
                       		style={"margin-bottom": "0px"},
                        ),
                        html.H5(
                        	"Tracking", style={"margin-top": "0px"}
                       	),
                	],)
         	],
        	className="container",
        	id="title",
        ),
        html.Div(
            [
                html.Div(
                    [
                        html.P("World Data Date", className="control_label"),
			dcc.Slider(
                            id="world_date_slider",
                            min=min_days,
                            max=max_days,
                            loading_state={
				'is_loading':True,
				},
                            value=max_days,
			    marks={
				min_days:min_datetime,
				max_days:max_datetime
			    },
                            className="dcc_control",
                        ),
                        html.P("", className="control_label"),
                        html.P("", className="control_label"),
                        html.P("Select Metric", className="control_label"),
                        dcc.Dropdown(
                            id="obs",
                            options=observables,
                            multi=False,
                            value='active',
                            className="dcc_control",
                        ),
                        html.P("Select Countries", className="control_label"),
                        dcc.Dropdown(
                            id="countries",
                            options=country_list,
                            multi=True,
                            value=['USA','Canada','Argentina','China','Doubles 5 days'],
                            className="dcc_control",
                        ),
                    ],
                    className="pretty_container three columns",
                    id="cross-filter-options",
                ),
                html.Div(
                    [
                        html.Div(
                            [
                                html.Div(
                                    [html.H6(id="world_date"), html.P("World Statistics")],
                                    id="date",
                                    className="mini_container",
                                ),
                                html.Div(
                                    [html.H6(id="world_active"), html.P("Active Cases")],
                                    id="active",
                                    className="mini_container",
                                ),
                                html.Div(
                                    [html.H6(id="world_confirmed"), html.P("Total Cases")],
                                    id="confirmed",
                                    className="mini_container",
                                ),
                                html.Div(
                                    [html.H6(id="world_recoveries"), html.P("Total Recoveries")],
                                    id="recoveries",
                                    className="mini_container",
                                ),
                                html.Div(
                                    [html.H6(id="world_deaths"), html.P("Total Deaths")],
                                    id="deaths",
                                    className="mini_container",
                                ),
                                html.Div(
                                    [html.H6(id="world_new_cases"), html.P("New Cases")],
                                    id="new_cases",
                                    className="mini_container",
                                ),
                            ],
                            id="info-container",
                            className="row container-display",
                        ),
                	html.Div(
                	    [
                             dcc.Graph( id="main_graph",
                                        config={'displayModeBar': False}
                                      )
                            ],
                	    className="pretty_container eleven columns",
                	),
                    ],
                    id="right-column",
                    className="container",
                ),

            ],
            className="row flex-display",
        ),
	html.Div(
	[
             html.Iframe(id='map', srcDoc = open(root+'output/covid19.folium.map.html','r').read(), width='100%', height='500')
        ],
	className='pretty container offset-by-one column twelve columns',
	style={'text-align':'center'}
	)
    ],
    id="mainContainer",
    style={"display": "flex", "flex-direction": "column"},
)



# Selectors -> main graph
@app.callback(
    Output("main_graph", "figure"),
    [
        Input("obs", "value"),
        Input("countries", "value"),
    ],
)
def make_main_figure(obs,countries):

	default_observable=obs

	fig = go.Figure()

	traces=[]   
 
    	# Add curves
	for index,groups in df_main.groupby(['region']):


		# Set defaults
		text=['' for j in range(len(groups.values))]
		mode='lines'
		width=0.8
		opacity=0.4
		color='grey'

		# Highlight certain countries
		if(index in countries):
			text=['' for j in range(len(groups.values)-1)]+[index]
			mode='lines+text+markers'
			width=2
			opacity=1
			color='Black'

		# Populate figure 
		trace = go.Scatter(
				x=groups['days_since_200'].values,
				y=groups[obs].values,
				mode=mode,
				name=index,
				opacity=opacity,
				marker=dict(size=3),
				hoverlabel = dict(namelength = -1),
				line=dict(color=color, width=width),
				text=text,
				textposition='middle right',
				visible=True
		
			)		
		traces = traces+[trace]	
                
	for t in traces:
		fig.add_trace(t)    
    
	# Format axes
	axes = dict(
            showline=True,
            showgrid=False,
            showticklabels=True,
            linecolor='rgb(0, 0, 0)',
            linewidth=2,
            fixedrange=True,
            ticks='outside',
            tickfont=dict(
                family='Arial',
                size=12,
                color='rgb(82, 82, 82)'
            ) ,
        )

	# Format figure
	fig.update_layout(
        autosize=True,
      #  width=800,
        height=500,
	margin=dict(
        l=10,
        r=10,
        b=10,
        t=25,
        pad=4
    	),
        title_text='',
        showlegend=False,
        yaxis_type='log',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='white',
        xaxis=axes,
        yaxis=axes,
        annotations = [dict(xref='paper',
                            yref='paper',
                            x=0, y=1.05,
                            showarrow=False,
                            text='Data sources: <a href="https://systems.jhu.edu/research/public-health/ncov/">John Hopkins University</a> & <a href="https://github.com/CSSEGISandData/COVID-19">GitHub</a>'
                           )])    

	xmax=max(df_main['days_since_200'].values)
	fig.update_xaxes(range=[0, xmax+8])
	fig.update_xaxes(title_text='# days since 200th case')
	fig.update_yaxes(title_text=observables_dict[obs])


	return fig



# Helper functions
def human_format(num):
    if num == 0:
        return "0"

    magnitude = int(math.log(num, 1000))
    mantissa = str(int(num / (1000 ** magnitude)))
    return mantissa + ["", "K", "M", "G", "T", "P"][magnitude]

def filter_world(obs,days):
	return df_world[df_world['days']==int(days)][obs].values



@app.callback(
	Output("world_recoveries","children"),
	[
		Input('world_date_slider','value')
	]
)
def update_world_recoveries(world_date_slider):
	val = filter_world('recovered',world_date_slider)
	return human_format(val)


@app.callback(
	Output("world_date","children"),
	[
		Input('world_date_slider','value')
	]
)
def update_world_confirmed(world_date_slider):
	val = filter_world('dates',world_date_slider)
	return val



@app.callback(
	Output("world_confirmed","children"),
	[
		Input('world_date_slider','value')
	]
)
def update_world_confirmed(world_date_slider):
	val = filter_world('confirmed',world_date_slider)
	return human_format(val)



@app.callback(
	Output("world_active","children"),
	[
		Input('world_date_slider','value')
	]
)
def update_world_active(world_date_slider):
	val = filter_world('active',world_date_slider)
	return human_format(val)



@app.callback(
	Output("world_new_cases","children"),
	[
		Input('world_date_slider','value')
	]
)
def update_world_new_cases(world_date_slider):
	val = filter_world('new cases',world_date_slider)
	return human_format(val)




@app.callback(
	Output("world_deaths","children"),
	[
		Input('world_date_slider','value')
	]
)
def update_world_deaths(world_date_slider):
	val = filter_world('death',world_date_slider)
	return human_format(val)


# Main
if __name__ == "__main__":
	app.run_server(host='0.0.0.0', port=8080, debug=True, use_reloader=False)
