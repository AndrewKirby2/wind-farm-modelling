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
The LES code for the wind farm actuator disk simulations can be found at https://code.metoffice.gov.uk/svn/monc/main/branches/dev/andrewkirby/r8169_turbines_v3 with the turbine implementation at /components/turbines/src/turbines.F90 (note a Met Office MOSRS account is required to assess this).
This code will produce incorrect results if:
- the grid resolution is so high that >10,000 grid points lie within a sphere with a diameter equal to turbine diameters which is centred on the centre of a turbine disk
- any turbine centre is within 1 turbine diameter of a boundary (e.g. top, bottom or side)
- any turbine centre is less than 2 turbine diameters away from another turbine centre.

