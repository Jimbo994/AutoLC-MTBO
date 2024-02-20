[![CC BY-NC-SA 4.0][cc-by-nc-sa-shield]][cc-by-nc-sa] [![DOI:10.26434/chemrxiv-2024-5mmvj](https://img.shields.io/badge/DOI-10.26434%2Fchemrxiv.2024.5mmvj-B31B1B)](https://doi.org/10.26434/chemrxiv-2024-5mmvj)

Code related to publication titled [Enhancing LC×LC separations through Multi-Task Bayesian Optimization](https://doi.org/10.26434/chemrxiv-2024-5mmvj)

<div style="display: flex; justify-content: space-between;">
  <img src="/figures/cross_val_single_task.png" width="300"/>
  <img src="/figures/cross_val_multi_task.png" width="300"/>
</div>


This repository contains information regarding 

Order of notebooks:

- setting_up_a_2DLC_system.ipynb
- cross_validation.ipynb
- ax_mtbo.ipynb

Installation:
```bash
conda env create --file install_environment.yml
conda activate AutoLC-MTO
```

The actual code used for testing the algorithm on the real experiments is very similar but is less straightforward to run as it requires an advanced laboratory setup. Please do not hesitate to contact me if you need further clarification on this.

When using this code, please cite:
[![DOI:10.26434/chemrxiv-2024-5mmvj](https://img.shields.io/badge/DOI-10.26434%2Fchemrxiv.2024.5mmvj-B31B1B)](https://doi.org/10.26434/chemrxiv-2024-5mmvj)


This work is licensed under a
[Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License][cc-by-nc-sa].

[cc-by-nc-sa]: http://creativecommons.org/licenses/by-nc-sa/4.0/
[cc-by-nc-sa-image]: https://licensebuttons.net/l/by-nc-sa/4.0/88x31.png
[cc-by-nc-sa-shield]: https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg
