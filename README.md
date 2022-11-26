# Forecasting-of-nonlinear-dynamics-based-on-symbolic-invariance
Abstract

Forecasting unknown dynamics is of great interest across many physics-related disciplines. However, data-driven machine learning methods are bothered by the poor generalization issue. To this end, a forecasting model based on symbolic invariance (i.e., symbolic expressions/equations that represent intrinsic system mechanisms) is proposed. By training and pruning a symbolic neural network wrapped in a numerical integrator, we develop an invariant symbolic structure that represents the evolution function and thus can generalize well to unseen data. To counter noise effect, an algorithmic framework for probabilistic forecasting has also been developed by leveraging a non-parametric Bayesian inference method. Additionally, to account for univariate forecasting that is partially observed from a system with multiple state variables, we further leverage the delay coordinate embedding to find symbolic invariance of the partially observed system in a more self-contained embedding. The performance of the proposed framework has been demonstrated on both synthetic and real-world nonlinear dynamics and shown better generalization over popular deep learning models in short/medium forecasting horizons. Moreover, comparison with dictionary-based symbolic regression methods suggests better-behaved and more efficient optimization of the proposed framework when the function search space is enormous.

## Citation
<pre>
@article{chen2022forecasting,
  title={Forecasting of nonlinear dynamics based on symbolic invariance},
  author={Chen, Zhao and Liu, Yang and Sun, Hao},
  journal={Computer Physics Communications},
  volume={277},
  pages={108382},
  year={2022},
  publisher={Elsevier}
}
</pre>
