# online-iid
An online ion identification (iid) script for isochronous Schottky spectrum via jupyter lab

## Prerequisites
- `Python 3`, `jupyterLab`
- `numpy`, `scipy`, `pandas`, `matplotlib`, `sqlite3` (for iid database), `WolframClient` (for sweep_iid)
- `itables`, `ipympl` (for interactive jupyter lab)

## Usage
1. Open `powershell`
2. Start `jupyterlab`
```shell
> jupyter lab
```
3. Click `iid_script-188.ipynb` as an example. Carefully read it and make your own iid_scripy.ipnb.

## Description
Class `IID` in `iid_isochronous.py` provides simulated Schottky spectrum.
- `__init__` requires the ion database (./ionic_data.db) generated from [gen_nubase.py](https://github.com/NanaVan/lpp-viewer/tree/main/database-maker)
- `calc_isochronous_peak` is the main processing function to generate the simulated Schottky spectrum.

Class `SWEEP_IID` in `sweep_iid.py` provides method to loop the possible Brho-C pairs, and give a recommanded Brho-C for the experimental spectrum to show a proper iid result.
