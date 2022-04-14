# varreg-on-crack


Varreg on Crack(only GPU boost, no DL-based hyperparameter optimization -> Parameter as defined in PMB paper of RW):

| Case |  voxel TRE  | world TRE[mm] | no snap world TRE[mm] | sigma jacobian |
|:----:|:-----------:|:-------------:|:---------------------:|:--------------:|
|  01  | 0.60 (0.61) |  0.79 (0.94)  |      0.99 (0.50)      |      0.10      |
|  02  | 0.55 (0.60) |  0.79 (0.96)  |      0.97 (0.48)      |      0.09      |
|  03  | 0.65 (0.68) |  0.96 (1.09)  |      1.14 (0.66)      |      0.11      |
|  04  | 0.99 (0.92) |  1.45 (1.46)  |      1.49 (1.18)      |      0.17      |
|  05  | 1.00 (1.11) |  1.36 (1.54)  |      1.45 (1.26)      |      0.12      |
|  06  | 1.20 (1.18) |  1.59 (1.66)  |      1.67 (1.50)      |      0.19      |
|  07  | 1.17 (0.98) |  1.46 (1.30)  |      1.52 (1.03)      |      0.16      |
|  08  | 1.09 (1.08) |  1.40 (1.67)  |      1.51 (1.51)      |      0.17      |
|  09  | 1.04 (0.77) |  1.34 (1.09)  |      1.32 (0.71)      |      0.15      |
|  10  | 1.12 (1.36) |  1.41 (1.69)  |      1.50 (1.54)      |      0.15      |
| mean |    0.94     |     1.26      |         1.36          |      0.14      |

ITK Variational Registration (Parameter as defined in PMB paper of RW):

Call: 

```
VariationalRegistration -F fixed.mha -M moving.mha -S mask.mha -i 800 -l 4 -t 2 -u 0 -r 0 -v 4 -d 2 -g 0.00005 -h 1 
```

| Case |  voxel TRE  | world TRE[mm] | no snap world TRE[mm] | sigma jacobian |
|:----:|:-----------:|:-------------:|:---------------------:|:--------------:|
|  01  | 0.65 (0.62) |  0.86 (0.96)  |      1.04 (0.51)      |      0.14      |
|  02  | 0.58 (0.61) |  0.82 (0.96)  |      1.00 (0.49)      |      0.12      |
|  03  | 0.68 (0.66) |  1.02 (1.09)  |      1.16 (0.62)      |      0.15      |
|  04  | 1.28 (1.30) |  1.80 (1.88)  |      1.84 (1.71)      |      0.21      |
|  05  | 1.02 (1.07) |  1.41 (1.52)  |      1.48 (1.24)      |      0.15      |
|  06  | 1.43 (1.38) |  1.83 (1.65)  |      1.86 (1.42)      |      0.26      |
|  07  | 1.75 (1.97) |  2.08 (2.08)  |      2.10 (1.92)      |      0.23      |
|  08  | 1.12 (1.05) |  1.44 (1.50)  |      1.46 (1.23)      |      0.19      |
|  09  | 1.24 (0.93) |  1.55 (1.17)  |      1.53 (0.87)      |      0.23      |
|  10  | 1.12 (1.16) |  1.48 (1.60)  |      1.52 (1.39)      |      0.19      |
| mean |    1.09     |     1.43      |         1.50          |      0.19      |
