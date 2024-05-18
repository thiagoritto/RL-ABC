# RL-ABC

This repository is about the strategy called Reinforcement Learning and Approximate Bayesian Computation (RL-ABC), proposed by TG Ritto, S Beregi and DAW Barton in 2022, than extended in 2023. The references are:

(1) Reinforcement learning and approximate Bayesian computation for model selection and parameter calibration applied to a nonlinear dynamical system, Ritto, T.G., Beregi, S.,Barton, D.A.W., Mechanical Systems and Signal Processing, 2022, 181, 109485.

@ARTICLE{Ritto2022,
	author = {Ritto, T.G. and Beregi, S. and Barton, D.A.W.},
	title = {Reinforcement learning and approximate Bayesian computation for model selection and parameter calibration applied to a nonlinear dynamical system},
	year = {2022},
	journal = {Mechanical Systems and Signal Processing},
	volume = {181},
	doi = {10.1016/j.ymssp.2022.109485},
}

(2) Reinforcement learning and approximate Bayesian computation (RL-ABC) for model selection and parameter calibration of time-varying systems, Ritto, T.G., Beregi, S., Barton, D.A.W., Mechanical Systems and Signal Processing, 2023, 200, 110458.

@ARTICLE{Ritto2023,
	author = {Ritto, T.G. and Beregi, S. and Barton, D.A.W.},
	title = {Reinforcement learning and approximate Bayesian computation (RL-ABC) for model selection and parameter calibration of time-varying systems},
	year = {2023},
	journal = {Mechanical Systems and Signal Processing},
	volume = {200},
	doi = {10.1016/j.ymssp.2023.110458},
}

In the context of digital twins and integration of physics-based models with machine learning tools, RL-ABC (Ritto et al., 2022) is a new methodology for model selection and parameter identification. It combines (i) reinforcement learning (RL) for model selection through a Thompson-like sampling with (ii) approximate Bayesian computation (ABC) for parameter identification and uncertainty quantification. The initial Beta distribution that represents the likelihood of the model is updated depending on how successful the model is at reproducing the reference data (reinforcement learning strategy). At the same time, the prior distribution of the model parameters is updated using a likelihood-free strategy (ABC). In the end, the rewards and the posterior distribution of the parameters of each model are obtained.

RL-ABC was extended to time-varying systems (Ritto et al., 2023). To tackle slowly-varying systems and detect abrupt changes, new features are proposed. (1) The probability of sampling the worst model has now a lower bound; because it cannot disappear, once it might be useful in the future as the system evolves. (2) A memory term (sliding window) is introduced such that past data can be forgotten whilst updating the reward; which might be useful depending on how fast the system changes. (3) The algorithm detects a change in the system by monitoring the models’ acceptance; a significant drop in acceptance indicates a change. If the system changes the algorithm is reset: new parameter ranges are computed and the rewards are restarted.

RLABC.m is a matlab code where one can run RL-ABC strategy (Ritto et al., 2022) to a simple spring problem.

RLABC_time_varying.m is a matlab code that shows the structure of RL-ABC strategy for time varying systems (Ritto et al., 2023); it cannot be run!
