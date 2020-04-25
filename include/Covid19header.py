#python
import pandas as pd
import numpy as np
import os,sys
import math
import re
from scipy import optimize
from scipy.optimize import curve_fit
#plots
import seaborn as sns
import matplotlib.pyplot as plt
#maps
from folium import IFrame
import folium
import branca
from folium.plugins import TimestampedGeoJson
import subprocess
#plotly
import plotly.graph_objects as go
import chart_studio.plotly as py  
import plotly.tools as tls   
from plotly.graph_objs import *
#custom
import pathlib
root=str(pathlib.Path(__file__).parent.absolute())+'/../'
sys.path.append(root+'include/')





#### HELPER FUNCTIONS ####
def describe(df):
    return pd.concat([df.describe().T,df.sum().rename('sum')], sort=True,axis=1).T
def indloc(df,field,values):
    return(df[df.index.get_level_values(field).isin(values)])



#### LOADING AND PRE-PROCESSING ####
def load_data():
    explore_data=False
    ts_path=root+'/csse_covid_19_data/csse_covid_19_time_series/'
    df_deaths=pd.read_csv(ts_path+'time_series_covid19_deaths_global.csv')
    df_recovered=pd.read_csv(ts_path+'time_series_covid19_recovered_global.csv')
    df_confirmed=pd.read_csv(ts_path+'time_series_covid19_confirmed_global.csv')
    # Explore data
    if(explore_data==True):
        print('Describe Confirmed')
        display(describe(df_confirmed))
        print('Describe Confirmed')
        display(df_confirmed.head())
    return df_deaths,df_recovered,df_confirmed

def uncolonize(df):
    df_helper = df.groupby('region').filter(lambda x: len(x)>1 and any(x['subregion'].isna()))
    colonizers = df_helper['region'].unique()
    colonized = df_helper['subregion'].unique()

    # and we give it back to you... the people
    for i,row in df.iterrows():
        if(row['subregion'] in colonized):
            df.at[i,'region'] = df.at[i,'subregion']        
            df.at[i,'subregion'] = np.nan        

    return(df)


def format_input(df,name): 
    df.rename(columns={'Province/State':'subregion','Country/Region':'region','Lat':'lat','Long':'long'},inplace=True)
    df = uncolonize(df)
    df['subregion'].fillna(df['region']+'_country',inplace=True)

    df = df.set_index(['subregion','region','lat','long']).stack()
    df.index.names = ['subregion','region','lat','long','dates']
    df.name = name
    df = df.reset_index()
    df['dates'] = pd.to_datetime(df['dates'],format='%m/%d/%y')

    return df


def combine_data(df_deaths,df_confirmed,df_recovered):
    join_cols = ['subregion','region','lat','long','dates']
    df_all = format_input(df_deaths,'death')
    df_all = pd.merge(df_all,format_input(df_confirmed,'confirmed'),on=join_cols,how='outer')
    df_all = pd.merge(df_all,format_input(df_recovered,'recovered'),on=join_cols,how='outer')
    df_all = df_all.groupby(join_cols).sum().reset_index()

    return df_all


def augment_dataset(df, level):

    # Rename USA
    df['region'] = df['region'].apply(lambda x: re.sub('^US$','USA',x))
    df['confirmed'] = df['confirmed'].replace(-1,0)

    # Add # active cases
    df['active'] = df['confirmed'] - df['death'] - df['recovered']
    
    # Add day count column
    pivotalDates = df.groupby(level)['dates'].min()
    df['days'] = df.merge(pivotalDates,how='left',on=[level],suffixes=('', '_r')).apply(lambda x: (x['dates']-x['dates_r']).days,axis=1).fillna(0)
    df['days'] = df['days'].astype(int)

    # Add days since 200th case
    pivotalDates = df[df['confirmed']>200].groupby(level)['dates'].min()
    df['days_since_200'] = df.merge(pivotalDates,how='left',on=[level],suffixes=('', '_r')).apply(lambda x: (x['dates']-x['dates_r']).days,axis=1).fillna(0)
    df['days_since_200'] = df['days_since_200'].astype(int)
    
    # Add deltas
    columns = ['confirmed','death','recovered']
    df_deltas = df.groupby([level,'dates'])[columns].sum()
    df_deltas = df_deltas.groupby(level)[columns].diff()
    df_deltas.rename(columns={'confirmed':'new cases',
                              'death':'new deaths',
                              'recovered':'new recoveries'}
                     ,inplace=True)
    df_deltas.fillna(0,inplace=True)
    df = df.merge(df_deltas, how='left',on=[level,'dates'])


    
    # Add % Death
    df['death_pc'] = round(df['death'] /df['confirmed']*100,2)

    # Add doubling rate
    df = df.merge(doubling_rate(df,level),how='left',on=[level,'days'])
    
    return df

def material_countries(df,additional):
    material_countries = list(df.groupby('region').max().nlargest(4,'confirmed').index)
    [material_countries.append(i) for i in additional]

    df = df[df['region'].isin(material_countries)]

    return df

def latest_data(df):
    return(df[df['dates']==df['dates'].max()])


def reference_curve(df,daysdouble):
	xmax = np.max(df['days_since_200'])
	xmin = np.min(df['days_since_200'])
	ymax = np.max(df['confirmed'])*0.9
	a = 200
	b = pow(2,1/daysdouble)
	name = 'Doubles '+str(daysdouble)+' days'
	xy = [[name,i,a*pow(b,i),a*pow(b,i),a*pow(b,i)-a*pow(b,i-1),0] for i in range(xmin,xmax) if a*pow(b,i)<=ymax]
	df_out = pd.DataFrame(xy,columns=['region','days_since_200','confirmed','active','new cases','death'])
	return df_out

#### EXPONENTIAL FITTING ####
def line(x,a,b): 
    return a*x+b

def get_doubling_rate(x):
    zero=0.000000001
    if((x[0]<=zero) and (x[0]>=-zero)):
        return np.nan
    else:
        # Solve for c by looking at coefficient of x
        # log[d 2^(x/c-f/c)] = a x + b
        return np.log(2)/x[0]

def doubling_rate(df,agg_level):

    lookback = 1000
    sample_size = 15
    double_rate=[]

    for index,group in  df.groupby(agg_level)[['days','confirmed']]:
        maxdays = group['days'].max()
        for day in range(maxdays,maxdays-lookback,-1):
    
            if(day<sample_size):
                break
            
            x_data = group['days'].values
            y_data = group['confirmed'].values

            filters = np.argwhere((x_data<=day) & (x_data>=day-sample_size)).T[0] 
            
            if(len(filters)<sample_size):
                break
            
            x_data = x_data[filters]
            y_data = y_data[filters]

            if(np.any(y_data==0)):
                break
            else:
                y_data = np.log(y_data)
            
            popt, pcov = curve_fit(line, x_data, y_data,p0=[1.5,0])
            rate = round(get_doubling_rate(popt),2)
            double_rate.append({agg_level:index,'days':day,'doubling_rate':rate,'best_fit':popt})

    return(pd.DataFrame(double_rate))



#### INTERACTIVE MAPS ####

def color_palette():
#    return ['#67001f','#b2182b','#d6604d','#f4a582','#fddbc7','#f7f7f7','#d1e5f0','#92c5de','#4393c3','#2166ac','#053061']
    return ['#67001f','#b2182b','#d6604d','#f4a582','#fddbc7','#d1e5f0','#92c5de','#4393c3','#2166ac','#053061'][::-1]

def statistics(sr):
    return({'min':sr.min(),'max':sr.max()})

def color_producer(rate,stats):
    color_scale = np.array(color_palette())
    n_colors = len(color_scale)
    col=0
    increments = (stats['max']-stats['min'])/n_colors
    for i in range(n_colors):
        if(
            (rate >= stats['min']+i*increments) &
            (rate <= stats['min']+(i+1)*increments+1)
        ):
            col = i
            break
        
    return(color_scale[col])


def create_geojson_features(df,level):
    features = []
        
    dict_stats = statistics(df['death'])
            
    for _, row in df.iterrows():
        
        
        feature = {
            'type': 'Feature',
            'geometry': {
                'type':'Point', 
                'coordinates':[row['long'],row['lat']]
            },
            'properties': {
                'time': row['dates'].__str__(),
                'style': {'color' : color_producer(row['death'],dict_stats)},
                'icon': 'circle',
                'popup':'{} <br> total cases: {:,} <br> active cases: {:,} <br> new cases: {:,} <br> deaths: {:,} ({}% of total)  <br> recoveries: {:,} <br> total 2 double: {:,} days'.format(
                                                                                                                row[level],
                                                                                                                int(row['confirmed']),
                                                                                                                int(row['active']),
                                                                                                                int(row['new cases']),
                                                                                                                int(row['death']),
                                                                                                                row['death_pc'],
                                                                                                                int(row['recovered']),
                                                                                                                row['doubling_rate']),
                'iconstyle':{
                    'fillColor': color_producer(row['death'],dict_stats),
                    'fillOpacity': 0.5,
                    'stroke': 'false',
                    'radius': float(np.log(row['confirmed']+2))/1.5
                }
            }
        }
        features.append(feature)
    return features

def make_map(features,legend_range):
    m = folium.Map(location=[20,0],height='100%',width='100%', control_scale=True,  zoom_start=1.8)

    TimestampedGeoJson(
        {'type': 'FeatureCollection',
        'features': features}
        , period='P1D'
        , add_last_point=True
        , auto_play=False
        , loop=False
        , duration='P1D'
        , max_speed=10
        , loop_button=True
        , date_options='YYYY/MM/DD'
        , time_slider_drag_update=True
    ).add_to(m)
    
    color_range = np.round(np.linspace(legend_range[0],legend_range[1],10)/1000).astype(int)
    colormap = branca.colormap.LinearColormap(color_palette(),
                                          index=color_range,
                                          vmin=min(color_range),
                                          vmax=max(color_range),
                                         caption='Deaths (thousands)')
    colormap.add_to(m)
    
    
    caption="""<table>
  <tr>
    <td>Data sources: <a href="https://systems.jhu.edu/research/public-health/ncov/">John Hopkins University</a> & <a href="https://github.com/CSSEGISandData/COVID-19">GitHub</a></td> 
  </tr>
</table>"""

    m.get_root().html.add_child(folium.Element( caption ))
    
    return m


#### PLOTS ####
def format_for_plot(df,level):
    df = df[df['days_since_200']>=0]
    df = df.set_index(['days_since_200',level]).unstack(level)
    return(df)


def doubling_rate_calibration(df_root):
    # Quick sanity check on best fits
    fig,ax = plt.subplots(figsize=(15,10));

    df = df_root.set_index(['days','region']).unstack('region')
    colors = ['b','g','r','c','m','y','k','orange']
    np.log(df[df['confirmed']>0]['confirmed']).plot(ax=ax,style=colors)

    dat=[]
    for index,row in df.iterrows():
        if(index%20 == 0):
            for name,v in row['best_fit'].iteritems():
                try:
                    math.isnan(v)
                except:
                    dat.append([[name,i,v[0]*i+v[1]] for i in range(index-15,index+1)])

    dat = [i for sublist in dat for i in sublist]
    df_calibrate = pd.DataFrame(dat,columns=['region','days','value'])
    df_calibrate = df_calibrate.set_index(['days','region']).unstack('region')['value']
    df_calibrate.plot(ax=ax,linestyle='dashed',style=colors);

    ax.set_yscale('log')
    ax.set_xlim(30,64)
    ax.set_ylim(2,15)


    plt.grid(which='both')
    plt.close()
    return(fig,ax)


def make_fig(df_bkg_1,df_bkg_2,df_frg_1,df_frg_2,default_observable):
    n_bkg = 0
    n_frg = 0

    list_observable = ['confirmed','new cases','active','death','doubling_rate']
    id_observable = list_observable.index(default_observable)
    dict_observable = {}
    for o in list_observable:
        dict_observable[o]=[]

    fig = go.Figure()

    color=['lightgrey','Black','Blue','Blue']
    visible=[True,True,False,False]
    mode =['lines','lines+text','lines+markers','lines+markers']
    width=[0.8,1,4,4]
    df = [df_bkg_1,df_bkg_2,df_frg_1,df_frg_2]
    which=['bkg','bkg','frg','frg']
    
    
    # Add curves
    for i in range(len(df)):
        for index,groups in df[i].groupby(['region']):

            y=''
            for o in list_observable:
                if(o in groups.columns):
                    dict_observable[o] += [groups[o]]
                    if(default_observable == o):
                        y = groups[o].values
                else:
                    dict_observable[o] += [[0]]
                    if(default_observable == o):
                        y = [0]
            
            
            if(i==1):
                text=['' for j in range(len(y)-1)]
                text.append(index)
            else:
                text=''
                
            fig.add_trace(
                go.Scatter(
                        x=groups['days_since_200'].values,
                        y=y,
                        mode=mode[i],
                        name=index,
                        marker=dict(size=7),
                        hoverlabel = dict(namelength = -1),
                        line=dict(color=color[i], width=width[i]),
                        text=text,
                        textposition='middle right',
                        visible=visible[i] #default value
                    )
                )
                
            if(which[i]=='bkg'):
                n_bkg+=1
            if(which[i]=='frg'):
                n_frg+=1    
            
    
    counter=0
    default_view=[True if i <n_bkg else False for i in range(n_bkg+n_frg)]
    dropdown_region = [dict(label="Default",
                     method="restyle",
                     args=[{"visible": default_view}])]
    for index,group in df_frg_1.groupby(['region']):
        frg_view = [True for i in range(n_bkg)]+[False if i!=counter else True for i in range(n_frg)]
        dropdown_region.append(dict(label=index,
                              method="restyle",
                              args=[{"visible": frg_view}]))
        counter+=1
    for index,group in df_frg_2.groupby(['region']):
        frg_view = [True for i in range(n_bkg)]+[False if i!=counter else True for i in range(n_frg)]
        dropdown_region.append(dict(label=index,
                              method="restyle",
                              args=[{"visible": frg_view}]))
        counter+=1

  
    dropdown_observable = [
        dict(label="Confirmed Cases",
             method="update",
             args=[{"y": dict_observable['confirmed']}]),
        dict(label="New Cases",
             method="update",
             args=[{"y": dict_observable['new cases']}]),
        dict(label="Active Cases",
             method="update",
             args=[{"y": dict_observable['active']}]),
        dict(label="Deaths",
             method="update",
             args=[{"y": dict_observable['death']}]),
        dict(label="Doubling Rate",
             method="update",
             args=[{"y": dict_observable['doubling_rate']}])
    ]

    fig.update_layout(
        updatemenus=[
            dict(
                active=0,
                buttons=list(dropdown_region),
                direction = "down",
                x = 1,
                y = 1.1
            ),
            dict(
                active=id_observable,
                buttons=list(dropdown_observable),
                direction = "down",
                x = 0.6,
                y = 1.1

            )
        ]
    )

    axes = dict(
            showline=True,
            showgrid=False,
            showticklabels=True,
            linecolor='rgb(0, 0, 0)',
            linewidth=2,
            ticks='outside',
            tickfont=dict(
                family='Arial',
                size=12,
                color='rgb(82, 82, 82)'
            ) ,
        )

    fig.update_layout(
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

    fig.update_xaxes(title_text='# days since 200th case')
#    fig.update_xaxes(range=[0, 50])

    return fig


