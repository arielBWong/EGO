# EGO
surrogate model optimization with Gaussian regressor
## 


## Cross-validation on hyper-parameters in gaussian regression fitting
1. Training data is split into five-fold cross-validation fashion. 
2. If the number of samples is less than 5, then leave-one-out is used for the cross-validation
3. Multi-process calculating of cross-validation is enabled. The switch is in the method **cross_val_gpr** in file **cross_val_hyperp**
