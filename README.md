# Do the following 

git clone https://github.com/gabrielmagill/COVID-19.git
cd COVID-19

python3 -m venv .venvs/dash
source .venvs/dash/bin/activate
pip3 install -r requirements.txt

python3 src/prep.main.py
python3 main.py

#on mac, see standalone plots
open output/covid19.folium.map.html   
open output/covid19.plotly.plot.html

# on chrome, point browser to this website:
http://0.0.0.0:8080/

# This is saved on GCP
https://simple-dash-app-engine-app-dot-covid19-91.ue.r.appspot.com/
