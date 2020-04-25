#python
import pandas as pd
import numpy as np
import os,sys
import math
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
import Covid19header as h
pd.set_option('display.max_rows', 1000)



def main():
	# Load Data
	df_deaths,df_recovered,df_confirmed = h.load_data()
	df_all = h.combine_data(df_deaths,df_confirmed,df_recovered)
	df_all = h.augment_dataset(df_all,'subregion')

	# Region level view
	df_region = df_all.groupby(['region','dates']).agg({'death':sum,'recovered':sum,'confirmed':sum,'lat':np.mean,'long':np.mean}).reset_index()
	df_region = h.augment_dataset(df_region,'region')

	# Material countries only
	df_all_material = h.material_countries(df_all,['Canada','Argentina'])
	df_region_material = h.material_countries(df_region,['Canada','Argentina'])

	# Today's data only
	df_all_today = h.latest_data(df_all)
	df_region_today = h.latest_data(df_region)


	# Check doubling rate calibration (visualize fits to see if they are working)
	fig,ax = h.doubling_rate_calibration(df_region_material)
	#fig

	# Make main plotly figure 
	countries = df_region_material['region'].unique()
	df_frg_1 = df_region[(df_region['confirmed']>200) & df_region['region'].isin(countries)]
	df_frg_2 = df_region[(df_region['confirmed']>200)]
	df_bkg_1 = df_region[(df_region['confirmed']>200)]
	df_bkg_2 = h.reference_curve(df_bkg_1,5).append(df_frg_1,sort=True)

	df_bkg_1.to_csv(root+'output/temp/df_main.csv')
	df_bkg_2.to_csv(root+'output/temp/df_ref.csv')
	fig1 = h.make_fig(df_bkg_1,df_bkg_2,df_frg_1,df_frg_2,'confirmed')
	fig2 = h.make_fig(df_bkg_1,df_bkg_2,df_frg_1,df_frg_2,'active')



	# Make city movie
	df = df_all[~df_all['subregion'].str.contains('_country')]
	death_range = [min(df['death']),max(df['death'])]
	m_subregion = h.make_map(h.create_geojson_features(df,'subregion'),death_range)


	# Make country movie
	df = df_region
	df_region_cities = df_all[df_all['region']=='Canada']
	col = df_region_cities.apply(lambda x: x['subregion'],axis=1)
	df_region_cities = df_region_cities.assign(region=col.values)
	df = df.append(df_region_cities[df.columns])
	death_range = [min(df['death']),max(df['death'])]
	m_region = h.make_map(h.create_geojson_features(df,'region'),death_range)


	fig1_html = fig1.to_html(full_html = False,include_plotlyjs = 'cdn',default_width='100%',default_height='90%')
	fig2_html = fig2.to_html(full_html = False,include_plotlyjs = 'cdn',default_width='100%',default_height='90%')

	m_region.save(root+"output/covid19.folium.map.html")
	#m_subregion.save(output+'covid19_cities.html')

	f = open(root+"output/covid19.plotly.plot.html", "w")
	f.write(fig1_html)
	f.close()

	return None


if __name__ == "__main__":
	main()

