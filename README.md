# Do the following 

python3 -m venv ~/myenv
source ~/myenv/bin/activate

git clone https://github.com/gabrielmagill/COVID-19.git
cd COVID-19
pip3 install -r requirements.txt

python3 src/Covid19.py

#on mac
open output/covid19.plotly.dashboard.html 
#on linux
chrome output/covid19.plotly.dashboard.html
