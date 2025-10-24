# Official Implementation of BFMDDT, a Novel GRN Inference from Myltiple types of dataset Based on Decision Tree Model

## To run
Install the needed packages, as

~~~
pip install -r requirement.txt
~~~
 
Run 'SS_TS.py'
~~~
python SS_TS.py
~~~

## Your dataset your way

### Multitype Datasets
In the example, a GRN is reconstructed bases on 'Ecoli-80_knockouts[0].tsv' and 'Ecoli-80_timeseries[0].tsv'. You can shape your own dataset as them to reconstruct GRNs based on your own datasets. 

### Single Type Datasets
BFMDDT can also reconstruct GRN from single type of dataset, as 'Steady.py' and 'Timeseries.py' demonstrates. 
~~~
python Steady.py
python Timeseries.py
~~~


## Citation

~~~
@ARTICLE{BFMDDT,
  author={Wang, Mingcan and Wang, Zhiqiong and Qu, Luxuan and Long, Kaifu and Xin, Junchang},
  journal={IEEE Transactions on Computational Biology and Bioinformatics}, 
  title={BFMDDT: A Decision-Tree-Based Gene Regulatory Network Inference From Multi-Type Datasets}, 
  year={2025},
  volume={22},
  number={4},
  pages={1778-1788},
  keywords={Decision trees;Training;Accuracy;Gene expression;Frequency modulation;Boosting;Bagging;Random forests;Prediction algorithms;Bioinformatics;Gene regulatory network (GRN);data integration;multi-type dataset;decision tree (DT)},
  doi={10.1109/TCBBIO.2025.3570817}}
~~~
