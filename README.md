# SR-Forest

SR-Forest is a general framework that supports ensemble learning for SR methods. It applies the following techniques to improve
the predictive performance of an ensemble of SR models:

* Automatic ensemble selection
* Residual learning based on decision trees

## Introduction
Genetic programming (GP) is a population-based ensemble learning framework. Thus, it is intuitive to use all good individuals in the final population to produce an accurate prediction. Even some packages, like "Evolutionary Forest", have been developed for this purpose. These packages are still limited to using a specific kind of symbolic regression algorithm.

To solve this issue, this open-source project intends to propose a general framework for GP-based ensemble-learning. In general, this package focuses on a general technique for improving the performance of GP-based ensemble learning, rather than improving a specific implementation of symbolic regression technique.

## Supported Base Learners

More base learners are still in development.

* Operon
* GP-GOMEA

## Citation

```bibtex
    @article{EvolutionaryForest,
        title = {An Evolutionary Forest for Regression},
        author = {Zhang, Hengzhe and Zhou, Aimin and Zhang, Hu},
        year = 2022,
        journal = {IEEE Transactions on Evolutionary Computation},
        volume = 26,
        number = 4,
        pages = {735--749},
        doi = {10.1109/TEVC.2021.3136667}
    }
```
