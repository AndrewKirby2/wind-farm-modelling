git clone https://github.com/AndrewKirby2/wind-farm-modelling.git

cd wind-farm-modelling

python3 -m venv venv

source venv/bin/activate

pip install -r requirements.txt

python predict_turbine_power_coefficient.py

python plot_layout_effects.py
