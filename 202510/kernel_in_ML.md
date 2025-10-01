# Understanding Kernel in Machine Learning

Reference [1]: [AlphaEdit: Null-Space Constrained Knowledge Editing for Language Models](https://arxiv.org/pdf/2410.02355)

Reference [2]: [Training Networks in Null Space of Feature Covariance for Continual Learning](https://arxiv.org/pdf/2103.07113)

My algebra teacher told me that **when encountering a morphism, you should consider its kernel**. But when I study machine learning, I often forget about this advice. So I was surprised to read the above two papers that explicitly use the concept of kernel in machine learning, especially in continual learning.

Consider a model parametered by a vector $ W \in \mathbb{R}^d $, which is trained by minimizing a loss function $L_0(W, D_0)$, where $D_0$ is the initial training dataset. Denote the minimum point as $ W_0 = \arg\min_W L_0(W, D_0) $ and the minimum loss as $ l_0 = L_0(W_0, D_0) $.

Now, suppose we want to further train the model on a new dataset $D_1$ by minimizing a new loss function $L_1(W, D_1)$, while ensuring that the performance on the original dataset $D_0$ does not degrade. This can be formulated as a constrained optimization problem:

$$
\begin{aligned}
& \min_W L_1(W, D_1) \\
& \text{s.t. } L_0(W, D_0) \leq l_0 + \epsilon
\end{aligned}
$$

where $\epsilon$ is a small tolerance level. Due to the linear nature of neural networks, we can reformulate the constraint as a linear constraint on the parameter update $\Delta W = W - W_0$:

$$
\nabla_W L_0(W_0, D_0)^\top \Delta W = 0 \\
\Delta W \in \mathrm{Ker} \left[ \nabla_W L_0(W_0, D_0)^\top \right]
$$

Suppose we use gradient descent to update the parameters, and the gradient of the new loss function at $W_0$ is given by $g_1 = \nabla_W L_1(W_0, D_1)$. An unconstrained gradient descent step would be:

$$
W' = W_0 - \eta g_1
$$

To satisfy the constraint, we need to project the gradient $g_1$ onto the kernel of the Jacobian of the original loss function. Let $J_0 = \nabla_W L_0(W_0, D_0)$ be the Jacobian matrix. The projection of $g_1$ onto the kernel of $J_0^\top$ can be computed as:

$$
g_1^{\perp} = g_1 - J_0 (J_0^\top J_0)^{-1} J_0^\top g_1
$$

Thus, the constrained update step becomes:

$$
W' = W_0 - \eta g_1^{\perp}
$$

This approach ensures that the update to the model parameters does not affect the performance on the original dataset $D_0$, effectively allowing the model to learn from the new dataset $D_1$ while preserving its knowledge from $D_0$. This technique is particularly useful in continual learning scenarios, where models need to adapt to new data without forgetting previously learned information.

## Q1.

<span style="text-decoration: underline">How to efficiently compute the projection, particularly when fine-tuning large foundation models?</span>

See [AlphaEdit: Null-Space Constrained Knowledge Editing for Language Models](https://arxiv.org/pdf/2410.02355) for details. The extra computation cost is mainly from maintaining a **null space model**. This technique is useful for editing large language models, as it allows for manually instilling knowledge or erasing harmful information.

## Q2.

<span style="text-decoration: underline">Can we apply this technique to fine-tuning methods based on reinforcement learning, such as in post training of embodied agents?</span>

A key observation is that the null space projection technique is something similar to TRPO (Trust Region Policy Optimization) in reinforcement learning, which also constrains the update step to stay within a trust region to prevent performance degradation on previous tasks. With a replay buffer (similar to $D_0$), the constrain of TRPO can be parameterized as KL-divergence between the old policy and the new policy:

$$
\mathcal{D}_{KL}(\pi_{\text{old}} || \pi_{\text{new}}) \leq \delta
$$

So if we can linearize the KL-divergence constraint (for example, by maintaining a null space model of $\mathcal{D}_{KL}(\pi_{\text{old}} || \pi_{\text{new}})$), we can apply the null space projection technique to reinforcement learning.

In my opinion, the $D_0$ in null-space RL can be a selected dataset, which contains expert demonstrations or high-advantage rollouts.



