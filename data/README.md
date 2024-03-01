## Getting the data

run

```{sh}
sh getData.sh
```

The data files are:
```
.
├── HepG2-100-50-50
│   ├── rbp_HepG2.rds
│   └── regression_HepG2_binding_psi_psip_KD.rds
├── K562-100-50-50
│   ├── rbp_K562.rds
│   └── regression_K562_binding_psi_psip_KD.rds
└── K562_HepG2-100-50-50
    ├── rbp_K562_HepG2.rds
    └── regression_K562_HepG2_binding_psi_psip_KD.rds
```

- The three folders contain three different data sets
- The regression*.tar.gz contain the datasets for regression
- code to load the data can be found in the notebooks folder