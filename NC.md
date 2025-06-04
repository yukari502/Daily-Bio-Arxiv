<div id=toc></div>

# Table of Contents

- [q-bio.NC](#q-bio.NC) [Total: 10]
- [cs.LG](#cs.LG) [Total: 2]
- [cs.CV](#cs.CV) [Total: 4]


<div id='q-bio.NC'></div>

# q-bio.NC [[Back]](#toc)

### [1] [Non-invasive two-step strategy BCI: brain-muscle-hand interface](https://arxiv.org/abs/2506.02013)
*Sun Ye,Zuo Cuiming,Zhang Rui,Shi Bin,Pang Yajing,Gao Lingyun,Zhao Bowei,Wang Jing,Yao Dezhong,Liu Gang*

Main category: q-bio.NC

TL;DR: 本文提出了一种基于生物进化选择机制的两步策略脑-肌肉-手接口（BMHI），通过整合脑-肌肉接口和肌肉-手接口，验证了其有效性和优势。实验表明BMHI可行，预测准确率为0.79，训练时间减少18倍，解码精度提高，并能成功完成在线任务。


<details>
  <summary>Details</summary>
Motivation: 传统脑机接口（如SSVEP-BCI和MI-BCI）依赖非自然范式，限制了广泛应用。本文旨在通过模仿自然神经运动通路的两步策略，提高训练效率和预测精度。

Method: 提出BMHI（BM + MH），通过离线、对比和在线实验验证其有效性。对比传统单步脑-手接口，BMHI引入神经肌肉传输机制作为中间约束层。

Result: BMHI预测准确率为0.79，训练时间减少18倍，解码精度提高。在线实验中，虚拟手和机械手成功完成日常任务。

Conclusion: BMHI的两步解码策略模仿自然神经运动通路，提高了训练效率和预测精度，推动了脑机接口技术向更自然交互模式发展。

Abstract: Brain-computer interface enables direct interaction between brain and device.
However, common brain-computer interfaces often employ one-step strategy that
rely on non-natural paradigms, such as SSVEP-BCI and MI-BCI, are limited to
specific scenarios, restricting their broader application. This paper first
proposes a two-step strategic brain-muscular-hand interface (BMHI) based on
biological evolutionary selection mechanism, by integrating the brain-muscle
(BM) interface with the muscle-hand (MH) interface through crosstalk ("BMHI =
BM + MH"). To verify the effectiveness of BMHI and the advantages of a two-step
strategy inspired by natural evolution, we conducted offline, comparison
(comparing BMHI (two-step) and brain-hand interface (one-step)), and online
experiments (using BMHI to control a virtual/machine hand for daily tasks). The
results show that: (1) BMHI is feasible and the prediction accuracy is 0.79;
(2) Unlike traditional multi-layer neural networks that attempt to establish a
direct brain-signal-to-action mapping through a single end-to-end process
(brain-hand interface), BMHI incorporates the neuro-muscular transmission
mechanisms evolved in biological systems as an intermediate constraint layer.
This phased decoding strategy can reduce training time by approximately 18-fold
and improve decoding accuracy; (3) In the online control experiment, both the
virtual hand and the manipulator were able to successfully complete tasks, like
moving objects such as boxes or plates and holding water glasses. The results
show that BMHI adopts a two-step decoding strategy that mimics natural human
neural motor pathways, improves training efficiency and prediction accuracy,
and promotes the development of BCI technology to a more natural interaction
mode.

</details>


### [2] [A Brain Graph Foundation Model: Pre-Training and Prompt-Tuning for Any Atlas and Disorder](https://arxiv.org/abs/2506.02044)
*Xinxu Wei,Kanhao Zhao,Yong Jiao,Lifang He,Yu Zhang*

Main category: q-bio.NC

TL;DR: 论文提出了一种基于图预训练的新型脑图基础模型（BrainGFM），结合图对比学习和图掩码自编码器，用于大规模fMRI预训练，支持多图谱和任务设置。


<details>
  <summary>Details</summary>
Motivation: 随着大语言模型（LLM）的发展，构建大规模脑基础模型以推动神经科学研究的需求日益增长。现有模型多基于时间序列信号或ROI特征，而BrainGFM提出了一种图预训练范式。

Method: BrainGFM采用图对比学习和图掩码自编码器进行预训练，整合了图提示和语言提示以支持下游任务，并通过元学习优化图提示，实现少样本和零样本学习。

Result: 模型在27个神经影像数据集上预训练，涵盖25种神经和精神疾病、8种分区图谱，包含25,000名受试者和60,000次fMRI扫描，性能优异。

Conclusion: BrainGFM为脑图基础模型提供了统一框架，具有广泛适应性和泛化能力，为神经科学研究提供了新工具。

Abstract: As large language models (LLMs) continue to revolutionize AI research, there
is a growing interest in building large-scale brain foundation models to
advance neuroscience. While most existing brain foundation models are
pre-trained on time-series signals or region-of-interest (ROI) features, we
propose a novel graph-based pre-training paradigm for constructing a brain
graph foundation model. In this paper, we introduce the Brain Graph Foundation
Model, termed BrainGFM, a unified framework that leverages graph contrastive
learning and graph masked autoencoders for large-scale fMRI-based pre-training.
BrainGFM is pre-trained on a diverse mixture of brain atlases with varying
parcellations, significantly expanding the pre-training corpus and enhancing
the model's ability to generalize across heterogeneous fMRI-derived brain
representations. To support efficient and versatile downstream transfer, we
integrate both graph prompts and language prompts into the model design,
enabling BrainGFM to flexibly adapt to a wide range of atlases, neurological
and psychiatric disorders, and task settings. Furthermore, we employ
meta-learning to optimize the graph prompts, facilitating strong generalization
to previously unseen disorders under both few-shot and zero-shot learning
conditions via language-guided prompting. BrainGFM is pre-trained on 27
neuroimaging datasets spanning 25 common neurological and psychiatric
disorders, encompassing 2 types of brain atlases (functional and anatomical)
across 8 widely-used parcellations, and covering over 25,000 subjects, 60,000
fMRI scans, and a total of 400,000 graph samples aggregated across all atlases
and parcellations. The code is available at:
https://github.com/weixinxu666/BrainGFM

</details>


### [3] [Identifying interactions across brain areas while accounting for individual-neuron dynamics with a Transformer-based variational autoencoder](https://arxiv.org/abs/2506.02263)
*Qi Xin,Robert E. Kass*

Main category: q-bio.NC

TL;DR: GLM-Transformer模型结合深度学习和GLM，用于分析神经信号传输，解决了背景动态干扰问题。


<details>
  <summary>Details</summary>
Motivation: 大规模神经记录技术提供了研究神经回路信号传输的机会，但神经响应的变异性（如行为或内部状态变化）可能干扰功能耦合的识别。

Method: 开发了GLM-Transformer模型，结合Transformer变分自编码器和GLM，捕捉非稳态神经元动态并估计神经群体间的定向相互作用。

Result: 在合成数据和真实数据（Allen Institute视觉编码数据集）中，模型成功恢复已知耦合结构并识别视觉层级的前馈通路。

Conclusion: GLM-Transformer为神经群体相互作用的识别提供了改进方法，同时结合了深度学习的优势与可解释性。

Abstract: Advances in large-scale recording technologies now enable simultaneous
measurements from multiple brain areas, offering new opportunities to study
signal transmission across interacting components of neural circuits. However,
neural responses exhibit substantial trial-to-trial variability, often driven
by unobserved factors such as subtle changes in animal behavior or internal
states. To prevent evolving background dynamics from contaminating
identification of functional coupling, we developed a hybrid neural spike train
model, GLM-Transformer, that incorporates flexible, deep latent variable models
into a point process generalized linear model (GLM) having an interpretable
component for cross-population interactions. A Transformer-based variational
autoencoder captures nonstationary individual-neuron dynamics that vary across
trials, while standard nonparametric regression GLM coupling terms provide
estimates of directed interactions between neural populations. We incorporate a
low-rank structure on population-to-population coupling effects to improve
scalability. Across synthetic datasets and mechanistic simulations,
GLM-Transformer recovers known coupling structure and remains robust to shared
background fluctuations. When applied to the Allen Institute Visual Coding
dataset, it identifies feedforward pathways consistent with established visual
hierarchies. This work offers a step toward improved identification of neural
population interactions, and contributes to ongoing efforts aimed at achieving
interpretable results while harvesting the benefits of deep learning.

</details>


### [4] [Brain-Like Processing Pathways Form in Models With Heterogeneous Experts](https://arxiv.org/abs/2506.02813)
*Jack Cook,Danyal Akarca,Rui Ponte Costa,Jascha Achterberg*

Main category: q-bio.NC

TL;DR: 论文探讨了大脑如何通过特定约束动态组织成任务相关的通路，并提出了一种基于混合专家架构的新框架，揭示了三种生物相关的归纳偏置对通路形成的影响。


<details>
  <summary>Details</summary>
Motivation: 研究大脑如何动态组织成任务相关的通路，以理解其背后的机制。

Method: 使用扩展的混合专家架构（Heterogeneous Mixture-of-Experts），并引入三种归纳偏置：路由成本、任务性能相关的成本缩放和随机专家丢弃。

Result: 人工模型中的通路与大脑在学习不同难度任务时的通路使用方式一致。

Conclusion: 提出了一种新框架，揭示了归纳偏置对大脑通路形成的作用，可能使混合专家架构更具适应性。

Abstract: The brain is made up of a vast set of heterogeneous regions that dynamically
organize into pathways as a function of task demands. Examples of such pathways
can be seen in the interactions between cortical and subcortical networks
during learning. This raises the question of how exactly brain regions organize
into these dynamic groups. In this work, we use an extension of the
Heterogeneous Mixture-of-Experts architecture, to show that heterogeneous
regions do not form processing pathways by themselves, implying that the brain
likely implements specific constraints which result in reliable formation of
pathways. We identify three biologically relevant inductive biases that
encourage pathway formation: a routing cost imposed on the use of more complex
regions, a scaling factor that reduces this cost when task performance is low,
and randomized expert dropout. When comparing our resulting Mixture-of-Pathways
model with the brain, we observe that the artificial pathways match how the
brain uses cortical and subcortical systems to learn and solve tasks of varying
difficulty. In summary, we introduce a novel framework for investigating how
the brain forms task-specific pathways through inductive biases which may make
Mixture-of-Experts architectures in general more adaptive.

</details>


### [5] [Modelling the Effects of Hearing Loss on Neural Coding in the Auditory Midbrain with Variational Conditioning](https://arxiv.org/abs/2506.03088)
*Lloyd Pellatt,Fotios Drakopoulos,Shievanie Sabesan,Nicholas A. Lesica*

Main category: q-bio.NC

TL;DR: 提出了一种新的变分条件模型，通过学习健康动物和噪声暴露动物的听觉中脑神经活动，直接编码听力损失空间，参数化仅需6个自由参数，预测效果接近最优动物特定模型。


<details>
  <summary>Details</summary>
Motivation: 听觉大脑建模因复杂性高而难以手动构建，且缺乏直接训练DNN模型的数据集。现有模型假设听觉处理在所有大脑中相同，无法捕捉听力损失的广泛变化。

Method: 使用变分条件模型，从健康及噪声暴露动物的听觉中脑神经活动记录中学习听力损失空间，参数化仅需6个自由参数。

Result: 模型准确预测了正常听力动物62%和听力受损动物68%的可解释神经反应方差，接近最优动物特定模型。通过贝叶斯优化，仅需15-30次迭代即可拟合新动物数据。

Conclusion: 该模型为未来开发参数化听力损失补偿模型奠定了基础，可通过快速拟合新用户数据直接恢复听力受损大脑的正常神经编码。

Abstract: The mapping from sound to neural activity that underlies hearing is highly
non-linear. The first few stages of this mapping in the cochlea have been
modelled successfully, with biophysical models built by hand and, more
recently, with DNN models trained on datasets simulated by biophysical models.
Modelling the auditory brain has been a challenge because central auditory
processing is too complex for models to be built by hand, and datasets for
training DNN models directly have not been available. Recent work has taken
advantage of large-scale high resolution neural recordings from the auditory
midbrain to build a DNN model of normal hearing with great success. But this
model assumes that auditory processing is the same in all brains, and therefore
it cannot capture the widely varying effects of hearing loss.
  We propose a novel variational-conditional model to learn to encode the space
of hearing loss directly from recordings of neural activity in the auditory
midbrain of healthy and noise exposed animals. With hearing loss parametrised
by only 6 free parameters per animal, our model accurately predicts 62\% of the
explainable variance in neural responses from normal hearing animals and 68%
for hearing impaired animals, within a few percentage points of state of the
art animal specific models. We demonstrate that the model can be used to
simulate realistic activity from out of sample animals by fitting only the
learned conditioning parameters with Bayesian optimisation, achieving
crossentropy loss within 2% of the optimum in 15-30 iterations. Including more
animals in the training data slightly improved the performance on unseen
animals. This model will enable future development of parametrised hearing loss
compensation models trained to directly restore normal neural coding in hearing
impaired brains, which can be quickly fitted for a new user by human in the
loop optimisation.

</details>


### [6] [Non-invasive two-step strategy BCI: brain-muscle-hand interface](https://arxiv.org/abs/2506.02013)
*Sun Ye,Zuo Cuiming,Zhang Rui,Shi Bin,Pang Yajing,Gao Lingyun,Zhao Bowei,Wang Jing,Yao Dezhong,Liu Gang*

Main category: q-bio.NC

TL;DR: 提出了一种基于生物进化选择机制的两步策略脑-肌肉-手接口（BMHI），通过整合脑-肌肉和肌肉-手接口，验证了其可行性和优势。


<details>
  <summary>Details</summary>
Motivation: 传统脑机接口依赖非自然范式，限制了广泛应用。BMHI通过模仿自然神经运动通路，提升训练效率和预测精度。

Method: 提出BMHI（BM + MH），通过离线、对比和在线实验验证其有效性。

Result: BMHI预测精度0.79，训练时间减少18倍，在线实验成功完成日常任务。

Conclusion: BMHI的两步解码策略更自然高效，推动了脑机接口技术的自然交互模式发展。

Abstract: Brain-computer interface enables direct interaction between brain and device.
However, common brain-computer interfaces often employ one-step strategy that
rely on non-natural paradigms, such as SSVEP-BCI and MI-BCI, are limited to
specific scenarios, restricting their broader application. This paper first
proposes a two-step strategic brain-muscular-hand interface (BMHI) based on
biological evolutionary selection mechanism, by integrating the brain-muscle
(BM) interface with the muscle-hand (MH) interface through crosstalk ("BMHI =
BM + MH"). To verify the effectiveness of BMHI and the advantages of a two-step
strategy inspired by natural evolution, we conducted offline, comparison
(comparing BMHI (two-step) and brain-hand interface (one-step)), and online
experiments (using BMHI to control a virtual/machine hand for daily tasks). The
results show that: (1) BMHI is feasible and the prediction accuracy is 0.79;
(2) Unlike traditional multi-layer neural networks that attempt to establish a
direct brain-signal-to-action mapping through a single end-to-end process
(brain-hand interface), BMHI incorporates the neuro-muscular transmission
mechanisms evolved in biological systems as an intermediate constraint layer.
This phased decoding strategy can reduce training time by approximately 18-fold
and improve decoding accuracy; (3) In the online control experiment, both the
virtual hand and the manipulator were able to successfully complete tasks, like
moving objects such as boxes or plates and holding water glasses. The results
show that BMHI adopts a two-step decoding strategy that mimics natural human
neural motor pathways, improves training efficiency and prediction accuracy,
and promotes the development of BCI technology to a more natural interaction
mode.

</details>


### [7] [A Brain Graph Foundation Model: Pre-Training and Prompt-Tuning for Any Atlas and Disorder](https://arxiv.org/abs/2506.02044)
*Xinxu Wei,Kanhao Zhao,Yong Jiao,Lifang He,Yu Zhang*

Main category: q-bio.NC

TL;DR: 本文提出了一种基于图的新型预训练范式BrainGFM，用于构建大脑图基础模型，通过图对比学习和图掩码自编码器进行大规模fMRI预训练，支持跨异构脑图谱和任务的灵活迁移。


<details>
  <summary>Details</summary>
Motivation: 现有大脑基础模型多基于时间序列信号或ROI特征，而本文旨在通过图预训练范式提升模型在异构fMRI数据上的泛化能力。

Method: 采用图对比学习和图掩码自编码器进行预训练，结合图提示和语言提示设计，并利用元学习优化图提示。

Result: BrainGFM在27个神经影像数据集上预训练，涵盖25种神经和精神疾病，支持少样本和零样本学习条件下的泛化。

Conclusion: BrainGFM为大脑图基础模型提供了统一框架，显著提升了跨异构脑图谱和任务的泛化能力。

Abstract: As large language models (LLMs) continue to revolutionize AI research, there
is a growing interest in building large-scale brain foundation models to
advance neuroscience. While most existing brain foundation models are
pre-trained on time-series signals or region-of-interest (ROI) features, we
propose a novel graph-based pre-training paradigm for constructing a brain
graph foundation model. In this paper, we introduce the Brain Graph Foundation
Model, termed BrainGFM, a unified framework that leverages graph contrastive
learning and graph masked autoencoders for large-scale fMRI-based pre-training.
BrainGFM is pre-trained on a diverse mixture of brain atlases with varying
parcellations, significantly expanding the pre-training corpus and enhancing
the model's ability to generalize across heterogeneous fMRI-derived brain
representations. To support efficient and versatile downstream transfer, we
integrate both graph prompts and language prompts into the model design,
enabling BrainGFM to flexibly adapt to a wide range of atlases, neurological
and psychiatric disorders, and task settings. Furthermore, we employ
meta-learning to optimize the graph prompts, facilitating strong generalization
to previously unseen disorders under both few-shot and zero-shot learning
conditions via language-guided prompting. BrainGFM is pre-trained on 27
neuroimaging datasets spanning 25 common neurological and psychiatric
disorders, encompassing 2 types of brain atlases (functional and anatomical)
across 8 widely-used parcellations, and covering over 25,000 subjects, 60,000
fMRI scans, and a total of 400,000 graph samples aggregated across all atlases
and parcellations. The code is available at:
https://github.com/weixinxu666/BrainGFM

</details>


### [8] [Identifying interactions across brain areas while accounting for individual-neuron dynamics with a Transformer-based variational autoencoder](https://arxiv.org/abs/2506.02263)
*Qi Xin,Robert E. Kass*

Main category: q-bio.NC

TL;DR: 论文提出了一种名为GLM-Transformer的混合神经尖峰序列模型，用于在神经信号传输研究中减少背景动态干扰，提高功能耦合识别的准确性。


<details>
  <summary>Details</summary>
Motivation: 大规模记录技术的进步使得多脑区信号传输研究成为可能，但神经响应的试次间变异性（如动物行为或内部状态的变化）会干扰功能耦合的识别。

Method: 结合了深度潜变量模型和点过程广义线性模型（GLM），利用Transformer变分自编码器捕捉非稳态神经元动态，同时通过GLM耦合项估计神经群体间的定向交互。

Result: 在合成数据和模拟实验中，GLM-Transformer能准确恢复已知的耦合结构，并在Allen Institute视觉编码数据集中识别出与视觉层级一致的反馈通路。

Conclusion: 该模型为神经群体交互的识别提供了改进方法，同时结合了深度学习的优势与可解释性。

Abstract: Advances in large-scale recording technologies now enable simultaneous
measurements from multiple brain areas, offering new opportunities to study
signal transmission across interacting components of neural circuits. However,
neural responses exhibit substantial trial-to-trial variability, often driven
by unobserved factors such as subtle changes in animal behavior or internal
states. To prevent evolving background dynamics from contaminating
identification of functional coupling, we developed a hybrid neural spike train
model, GLM-Transformer, that incorporates flexible, deep latent variable models
into a point process generalized linear model (GLM) having an interpretable
component for cross-population interactions. A Transformer-based variational
autoencoder captures nonstationary individual-neuron dynamics that vary across
trials, while standard nonparametric regression GLM coupling terms provide
estimates of directed interactions between neural populations. We incorporate a
low-rank structure on population-to-population coupling effects to improve
scalability. Across synthetic datasets and mechanistic simulations,
GLM-Transformer recovers known coupling structure and remains robust to shared
background fluctuations. When applied to the Allen Institute Visual Coding
dataset, it identifies feedforward pathways consistent with established visual
hierarchies. This work offers a step toward improved identification of neural
population interactions, and contributes to ongoing efforts aimed at achieving
interpretable results while harvesting the benefits of deep learning.

</details>


### [9] [Brain-Like Processing Pathways Form in Models With Heterogeneous Experts](https://arxiv.org/abs/2506.02813)
*Jack Cook,Danyal Akarca,Rui Ponte Costa,Jascha Achterberg*

Main category: q-bio.NC

TL;DR: 论文提出了一种基于Heterogeneous Mixture-of-Experts架构扩展的模型，研究大脑如何通过特定约束形成动态任务路径。


<details>
  <summary>Details</summary>
Motivation: 探索大脑区域如何动态组织成任务相关的路径，以及这些路径形成的机制。

Method: 使用扩展的Heterogeneous Mixture-of-Experts架构，引入三种生物相关的归纳偏置（路由成本、任务性能低时的成本调整、随机专家丢弃）来模拟路径形成。

Result: 人工路径模型与大脑在学习和解决不同难度任务时的路径使用方式相匹配。

Conclusion: 提出了一种新框架，通过归纳偏置研究大脑任务路径形成，可能使Mixture-of-Experts架构更具适应性。

Abstract: The brain is made up of a vast set of heterogeneous regions that dynamically
organize into pathways as a function of task demands. Examples of such pathways
can be seen in the interactions between cortical and subcortical networks
during learning. This raises the question of how exactly brain regions organize
into these dynamic groups. In this work, we use an extension of the
Heterogeneous Mixture-of-Experts architecture, to show that heterogeneous
regions do not form processing pathways by themselves, implying that the brain
likely implements specific constraints which result in reliable formation of
pathways. We identify three biologically relevant inductive biases that
encourage pathway formation: a routing cost imposed on the use of more complex
regions, a scaling factor that reduces this cost when task performance is low,
and randomized expert dropout. When comparing our resulting Mixture-of-Pathways
model with the brain, we observe that the artificial pathways match how the
brain uses cortical and subcortical systems to learn and solve tasks of varying
difficulty. In summary, we introduce a novel framework for investigating how
the brain forms task-specific pathways through inductive biases which may make
Mixture-of-Experts architectures in general more adaptive.

</details>


### [10] [Modelling the Effects of Hearing Loss on Neural Coding in the Auditory Midbrain with Variational Conditioning](https://arxiv.org/abs/2506.03088)
*Lloyd Pellatt,Fotios Drakopoulos,Shievanie Sabesan,Nicholas A. Lesica*

Main category: q-bio.NC

TL;DR: 论文提出了一种新的变分条件模型，用于直接从健康动物和噪声暴露动物的听觉中脑神经活动中学习编码听力损失的空间。该模型仅需6个自由参数即可准确预测神经响应，并在未见过的动物上表现良好。


<details>
  <summary>Details</summary>
Motivation: 传统模型假设所有大脑的听觉处理相同，无法捕捉听力损失的多样性。本文旨在通过直接从神经活动中学习听力损失的编码空间，解决这一问题。

Method: 提出了一种变分条件模型，利用贝叶斯优化拟合学习到的条件参数，仅需6个参数即可预测神经响应。

Result: 模型在健康动物中预测了62%的可解释方差，在听力受损动物中预测了68%，接近最先进的动物特定模型。

Conclusion: 该模型为未来开发参数化听力损失补偿模型奠定了基础，可通过快速拟合新用户的参数恢复正常的神经编码。

Abstract: The mapping from sound to neural activity that underlies hearing is highly
non-linear. The first few stages of this mapping in the cochlea have been
modelled successfully, with biophysical models built by hand and, more
recently, with DNN models trained on datasets simulated by biophysical models.
Modelling the auditory brain has been a challenge because central auditory
processing is too complex for models to be built by hand, and datasets for
training DNN models directly have not been available. Recent work has taken
advantage of large-scale high resolution neural recordings from the auditory
midbrain to build a DNN model of normal hearing with great success. But this
model assumes that auditory processing is the same in all brains, and therefore
it cannot capture the widely varying effects of hearing loss.
  We propose a novel variational-conditional model to learn to encode the space
of hearing loss directly from recordings of neural activity in the auditory
midbrain of healthy and noise exposed animals. With hearing loss parametrised
by only 6 free parameters per animal, our model accurately predicts 62\% of the
explainable variance in neural responses from normal hearing animals and 68%
for hearing impaired animals, within a few percentage points of state of the
art animal specific models. We demonstrate that the model can be used to
simulate realistic activity from out of sample animals by fitting only the
learned conditioning parameters with Bayesian optimisation, achieving
crossentropy loss within 2% of the optimum in 15-30 iterations. Including more
animals in the training data slightly improved the performance on unseen
animals. This model will enable future development of parametrised hearing loss
compensation models trained to directly restore normal neural coding in hearing
impaired brains, which can be quickly fitted for a new user by human in the
loop optimisation.

</details>


<div id='cs.LG'></div>

# cs.LG [[Back]](#toc)

### [11] [Memorization to Generalization: Emergence of Diffusion Models from Associative Memory](https://arxiv.org/abs/2505.21777)
*Bao Pham,Gabriel Raya,Matteo Negri,Mohammed J. Zaki,Luca Ambrogioni,Dmitry Krotov*

Main category: cs.LG

TL;DR: 论文将扩散模型与Hopfield网络的联想记忆特性类比，揭示了在小数据和大数据量下扩散模型的记忆与泛化行为，并预测和验证了虚假状态的存在。


<details>
  <summary>Details</summary>
Motivation: 研究扩散模型在记忆编码和检索过程中的行为，探索其与Hopfield网络的相似性，特别是虚假状态的出现及其对生成样本的影响。

Method: 将扩散模型的训练和生成阶段分别类比为Hopfield网络的记忆编码和检索，分析不同数据量下的模型行为。

Result: 在小数据量下，扩散模型表现出强记忆行为；在大数据量下，生成样本的流形形成新的吸引子状态，虚假状态出现在过渡边界。

Conclusion: 通过联想记忆的视角，为扩散模型的记忆-泛化现象提供了新见解，并理论预测和实验验证了虚假状态的存在。

Abstract: Hopfield networks are associative memory (AM) systems, designed for storing
and retrieving patterns as local minima of an energy landscape. In the
classical Hopfield model, an interesting phenomenon occurs when the amount of
training data reaches its critical memory load $- spurious\,\,states$, or
unintended stable points, emerge at the end of the retrieval dynamics, leading
to incorrect recall. In this work, we examine diffusion models, commonly used
in generative modeling, from the perspective of AMs. The training phase of
diffusion model is conceptualized as memory encoding (training data is stored
in the memory). The generation phase is viewed as an attempt of memory
retrieval. In the small data regime the diffusion model exhibits a strong
memorization phase, where the network creates distinct basins of attraction
around each sample in the training set, akin to the Hopfield model below the
critical memory load. In the large data regime, a different phase appears where
an increase in the size of the training set fosters the creation of new
attractor states that correspond to manifolds of the generated samples.
Spurious states appear at the boundary of this transition and correspond to
emergent attractor states, which are absent in the training set, but, at the
same time, have distinct basins of attraction around them. Our findings
provide: a novel perspective on the memorization-generalization phenomenon in
diffusion models via the lens of AMs, theoretical prediction of existence of
spurious states, empirical validation of this prediction in commonly-used
diffusion models.

</details>


### [12] [Memorization to Generalization: Emergence of Diffusion Models from Associative Memory](https://arxiv.org/abs/2505.21777)
*Bao Pham,Gabriel Raya,Matteo Negri,Mohammed J. Zaki,Luca Ambrogioni,Dmitry Krotov*

Main category: cs.LG

TL;DR: 论文将扩散模型与Hopfield网络的联想记忆特性对比，揭示了扩散模型在小数据和大数据条件下的记忆与泛化行为，并预测和验证了虚假状态的存在。


<details>
  <summary>Details</summary>
Motivation: 研究扩散模型在记忆编码和检索中的行为，探索其与Hopfield网络的相似性，尤其是虚假状态的出现。

Method: 将扩散模型的训练和生成阶段分别类比为记忆编码和检索，分析其在小数据和大数据条件下的行为。

Result: 在小数据条件下，扩散模型表现出强记忆性；在大数据条件下，出现新的吸引子状态（虚假状态），验证了理论预测。

Conclusion: 通过联想记忆的视角，为扩散模型的记忆-泛化现象提供了新见解，并预测和验证了虚假状态的存在。

Abstract: Hopfield networks are associative memory (AM) systems, designed for storing
and retrieving patterns as local minima of an energy landscape. In the
classical Hopfield model, an interesting phenomenon occurs when the amount of
training data reaches its critical memory load $- spurious\,\,states$, or
unintended stable points, emerge at the end of the retrieval dynamics, leading
to incorrect recall. In this work, we examine diffusion models, commonly used
in generative modeling, from the perspective of AMs. The training phase of
diffusion model is conceptualized as memory encoding (training data is stored
in the memory). The generation phase is viewed as an attempt of memory
retrieval. In the small data regime the diffusion model exhibits a strong
memorization phase, where the network creates distinct basins of attraction
around each sample in the training set, akin to the Hopfield model below the
critical memory load. In the large data regime, a different phase appears where
an increase in the size of the training set fosters the creation of new
attractor states that correspond to manifolds of the generated samples.
Spurious states appear at the boundary of this transition and correspond to
emergent attractor states, which are absent in the training set, but, at the
same time, have distinct basins of attraction around them. Our findings
provide: a novel perspective on the memorization-generalization phenomenon in
diffusion models via the lens of AMs, theoretical prediction of existence of
spurious states, empirical validation of this prediction in commonly-used
diffusion models.

</details>


<div id='cs.CV'></div>

# cs.CV [[Back]](#toc)

### [13] [Quantifying task-relevant representational similarity using decision variable correlation](https://arxiv.org/abs/2506.02164)
*Yu,Qian,Wilson S. Geisler,Xue-Xin Wei*

Main category: cs.CV

TL;DR: 本文提出了一种新方法（DVC）来比较模型与猴子大脑在图像分类任务中的决策策略相似性，发现模型与猴子之间的相似性较低，且随着模型性能提升反而下降。


<details>
  <summary>Details</summary>
Motivation: 研究动机是解决先前关于大脑与深度神经网络在图像分类任务中表征相似性的争议，提出更准确的比较方法。

Method: 采用决策变量相关（DVC）方法，量化分类任务中解码决策的相关性，评估模型与猴子V4/IT记录的相似性。

Result: 发现模型间相似性与猴子间相似性相当，但模型与猴子相似性较低且随模型性能提升而下降；对抗训练和更大数据集预训练未能提升模型与猴子相似性。

Conclusion: 结果表明猴子V4/IT与图像分类模型在任务相关表征上存在根本差异。

Abstract: Previous studies have compared the brain and deep neural networks trained on
image classification. Intriguingly, while some suggest that their
representations are highly similar, others argued the opposite. Here, we
propose a new approach to characterize the similarity of the decision
strategies of two observers (models or brains) using decision variable
correlation (DVC). DVC quantifies the correlation between decoded decisions on
individual samples in a classification task and thus can capture task-relevant
information rather than general representational alignment. We evaluate this
method using monkey V4/IT recordings and models trained on image classification
tasks.
  We find that model--model similarity is comparable to monkey--monkey
similarity, whereas model--monkey similarity is consistently lower and,
surprisingly, decreases with increasing ImageNet-1k performance. While
adversarial training enhances robustness, it does not improve model--monkey
similarity in task-relevant dimensions; however, it markedly increases
model--model similarity. Similarly, pre-training on larger datasets does not
improve model--monkey similarity. These results suggest a fundamental
divergence between the task-relevant representations in monkey V4/IT and those
learned by models trained on image classification tasks.

</details>


### [14] [Explicitly Modeling Subcortical Vision with a Neuro-Inspired Front-End Improves CNN Robustness](https://arxiv.org/abs/2506.03089)
*Lucas Piper,Arlindo L. Oliveira,Tiago Marques*

Main category: cs.CV

TL;DR: EVNets结合VOneBlock和SubcorticalBlock，提升CNN的鲁棒性和生物视觉对齐，性能优于基准模型8.5%，与数据增强技术结合效果更佳。


<details>
  <summary>Details</summary>
Motivation: 解决CNN在视觉扰动和域外图像中的脆弱性，通过模仿生物视觉结构提升模型鲁棒性。

Method: 引入EVNets，结合VOneBlock和SubcorticalBlock，后者基于神经科学模型参数化，优化与生物视觉对齐。

Result: EVNets在V1对齐、形状偏好和鲁棒性基准测试中表现优异，性能提升8.5%。与数据增强技术结合时效果更佳。

Conclusion: EVNets展示了架构改进与数据增强的互补性，为提升CNN鲁棒性提供了新方向。

Abstract: Convolutional neural networks (CNNs) trained on object recognition achieve
high task performance but continue to exhibit vulnerability under a range of
visual perturbations and out-of-domain images, when compared with biological
vision. Prior work has demonstrated that coupling a standard CNN with a
front-end block (VOneBlock) that mimics the primate primary visual cortex (V1)
can improve overall model robustness. Expanding on this, we introduce Early
Vision Networks (EVNets), a new class of hybrid CNNs that combine the VOneBlock
with a novel SubcorticalBlock, whose architecture draws from computational
models in neuroscience and is parameterized to maximize alignment with
subcortical responses reported across multiple experimental studies. Without
being optimized to do so, the assembly of the SubcorticalBlock with the
VOneBlock improved V1 alignment across most standard V1 benchmarks, and better
modeled extra-classical receptive field phenomena. In addition, EVNets exhibit
stronger emergent shape bias and overperform the base CNN architecture by 8.5%
on an aggregate benchmark of robustness evaluations, including adversarial
perturbations, common corruptions, and domain shifts. Finally, we show that
EVNets can be further improved when paired with a state-of-the-art data
augmentation technique, surpassing the performance of the isolated data
augmentation approach by 7.3% on our robustness benchmark. This result reveals
complementary benefits between changes in architecture to better mimic biology
and training-based machine learning approaches.

</details>


### [15] [Quantifying task-relevant representational similarity using decision variable correlation](https://arxiv.org/abs/2506.02164)
*Yu,Qian,Wilson S. Geisler,Xue-Xin Wei*

Main category: cs.CV

TL;DR: 该论文提出了一种新方法DVC来比较模型和猴子大脑在图像分类任务中的决策策略相似性，发现模型与猴子之间的相似性较低且与模型性能成反比。


<details>
  <summary>Details</summary>
Motivation: 探讨深度神经网络与猴子大脑在图像分类任务中的决策策略相似性，以理解两者在任务相关表征上的差异。

Method: 使用决策变量相关（DVC）方法量化模型和猴子大脑在分类任务中的决策策略相似性，评估了猴子V4/IT记录和图像分类模型。

Result: 模型间相似性与猴子间相似性相当，但模型与猴子相似性较低且随模型性能提升而下降；对抗训练和更大数据集预训练未能改善模型与猴子相似性。

Conclusion: 猴子V4/IT与图像分类模型在任务相关表征上存在根本性差异，模型性能提升并未使其更接近猴子大脑的决策策略。

Abstract: Previous studies have compared the brain and deep neural networks trained on
image classification. Intriguingly, while some suggest that their
representations are highly similar, others argued the opposite. Here, we
propose a new approach to characterize the similarity of the decision
strategies of two observers (models or brains) using decision variable
correlation (DVC). DVC quantifies the correlation between decoded decisions on
individual samples in a classification task and thus can capture task-relevant
information rather than general representational alignment. We evaluate this
method using monkey V4/IT recordings and models trained on image classification
tasks.
  We find that model--model similarity is comparable to monkey--monkey
similarity, whereas model--monkey similarity is consistently lower and,
surprisingly, decreases with increasing ImageNet-1k performance. While
adversarial training enhances robustness, it does not improve model--monkey
similarity in task-relevant dimensions; however, it markedly increases
model--model similarity. Similarly, pre-training on larger datasets does not
improve model--monkey similarity. These results suggest a fundamental
divergence between the task-relevant representations in monkey V4/IT and those
learned by models trained on image classification tasks.

</details>


### [16] [Explicitly Modeling Subcortical Vision with a Neuro-Inspired Front-End Improves CNN Robustness](https://arxiv.org/abs/2506.03089)
*Lucas Piper,Arlindo L. Oliveira,Tiago Marques*

Main category: cs.CV

TL;DR: EVNets结合了VOneBlock和SubcorticalBlock，提高了CNN的鲁棒性和生物学对齐性，在对抗扰动和域偏移等任务中表现优异。


<details>
  <summary>Details</summary>
Motivation: 解决CNN在视觉扰动和域外图像中的脆弱性问题，通过模仿生物视觉系统提升模型鲁棒性。

Method: 提出EVNets，结合VOneBlock和SubcorticalBlock，后者基于神经科学计算模型设计。

Result: EVNets在V1对齐性、形状偏好和鲁棒性基准测试中表现优于基础CNN，提升8.5%。结合数据增强后性能进一步提升7.3%。

Conclusion: EVNets展示了架构改进与数据增强的互补性，为提升模型鲁棒性提供了新方向。

Abstract: Convolutional neural networks (CNNs) trained on object recognition achieve
high task performance but continue to exhibit vulnerability under a range of
visual perturbations and out-of-domain images, when compared with biological
vision. Prior work has demonstrated that coupling a standard CNN with a
front-end block (VOneBlock) that mimics the primate primary visual cortex (V1)
can improve overall model robustness. Expanding on this, we introduce Early
Vision Networks (EVNets), a new class of hybrid CNNs that combine the VOneBlock
with a novel SubcorticalBlock, whose architecture draws from computational
models in neuroscience and is parameterized to maximize alignment with
subcortical responses reported across multiple experimental studies. Without
being optimized to do so, the assembly of the SubcorticalBlock with the
VOneBlock improved V1 alignment across most standard V1 benchmarks, and better
modeled extra-classical receptive field phenomena. In addition, EVNets exhibit
stronger emergent shape bias and overperform the base CNN architecture by 8.5%
on an aggregate benchmark of robustness evaluations, including adversarial
perturbations, common corruptions, and domain shifts. Finally, we show that
EVNets can be further improved when paired with a state-of-the-art data
augmentation technique, surpassing the performance of the isolated data
augmentation approach by 7.3% on our robustness benchmark. This result reveals
complementary benefits between changes in architecture to better mimic biology
and training-based machine learning approaches.

</details>
