Clone remote GitHub repository
```
git clone https://github.com/AndrewKirby2/wind-farm-modelling.git
```
Change working directory
```
cd wind-farm-modelling
```
Create python virtual environment
```
python3 -m venv venv
```
Activate virtual environment
```
source venv/bin/activate
```
Install dependencies
```
pip install -r requirements.txt
```
Make a plot of turbine power coefficients
```
python predict_turbine_power_coefficient.py
```
Make of plots of loss factors
```
python plot_layout_effects.py
```
Test repo copy
