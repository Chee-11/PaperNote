# Meta-Learning

## Meta-Learning: Learning to Learn Fast [1]

There are three common approaches: 

1) Metric-based: learn an efficient distance metric;

2) Model-based: use (recurrent) network with external or internal memory;

3) Optimization-based: optimize the model parameters explicitly (明确地) for fast learning 

### 1. Metric-Based: 

The core idea in metric-based meta-learning is similar to nearest neighbors algorithms (i.e., k-NN classificer and k-means clustering) and kernel density estimation.
$$
P_{\theta}(y | \mathbf{x}, S)=\sum_{\left(\mathbf{x}_{i}, y_{i}\right) \in S} k_{\theta}\left(\mathbf{x}, \mathbf{x}_{i}\right) y_{i}
$$
To learn **a good kernel** is crucial to the success of a metric-based meta-learning model. Metric learning is well aligned with this intention, as it aims to learn a metric or distance function over objects.  The notion of a good metric is problem-dependent. It should represent the relationship between inputs in the task space and facilitate problem solving.

#### 1.1 Convolutional Siamese Neural Network

The Siamese Neural Network [2] is composed of two twin networks and their outputs are jointly trained on top with a function to learn the relationship between pairs of input data samples. The twin networks are identical, sharing the same weights and network parameters. 

Koch, Zemel & Salakhutdinov (2015) [3] proposed a method to use the siamese neural network to do one-shot image classification.

![siamese](https://lilianweng.github.io/lil-log/assets/images/siamese-conv-net.png)

*Fig. 1-1. The architecture of convolutional siamese neural network for few-shot image classification.*

1.1 Steps:

1.First, convolutional siamese network learns to encode two images into feature vectors via a embedding function *fθ*  which contains a couple of convolutional layers.

2.The L1-distance between two embeddings is |*fθ*(**x**i*)−*f**θ(**x*j*)|.

3.The distance is converted to a probability *p* by a linear feedforward layer and sigmoid. It is the probability of whether two images are drawn from the same class.

4.the loss is cross entropy because the label is binary.
$$
\begin{aligned} p\left(\mathbf{x}_{i}, \mathbf{x}_{j}\right) &=\sigma\left(\mathbf{W}\left|f_{\theta}\left(\mathbf{x}_{i}\right)-f_{\theta}\left(\mathbf{x}_{j}\right)\right|\right) \\ \mathcal{L}(B) &=\sum_{\left(\mathbf{x}_{i}, x, y_{i}, y_{j}\right) \in B} \mathbf{1}_{y_{i}-y_{j}} \log p\left(\mathbf{x}_{i}, \mathbf{x}_{j}\right)+\left(1-\mathbf{1}_{y_{i}-y_{j}}\right) \log \left(1-p\left(\mathbf{x}_{i}, \mathbf{x}_{j}\right)\right) \end{aligned}
$$


#### 1.2 Matching Networks

The task of Matching Networks  (Vinyals et al., 2016) [4] is to learn a classifier for any given (small) support set. This classifier defines a probability distribution over output labels *y* given a test example **x**. Similar to other metric-based models, the classifier output is defined as a sum of labels of support samples weighted by attention kernel *a*(**x**,**x**i).

![siamese](https://lilianweng.github.io/lil-log/assets/images/matching-networks.png)

*Fig. 1-2. The architecture of Matching Networks.*
$$
c_{S}(\mathbf{x})=P(y | \mathbf{x}, S)=\sum_{i=1}^{k} a\left(\mathbf{x}, \mathbf{x}_{i}\right) y_{i}, \text { where } S=\left\{\left(\mathbf{x}_{i}, y_{i}\right)\right\}_{i=1}^{k}
$$
The attention kernel depends on two embedding functions, *f* and *g*, for encoding the test sample and the support set samples respectively. The attention weight between two data points is the cosine similarity, cosine(.), between their embedding vectors, normalized by softmax:
$$
a\left(\mathbf{x}, \mathbf{x}_{i}\right)=\frac{\exp \left(\operatorname{cosine}\left(f(\mathbf{x}), g\left(\mathbf{x}_{i}\right)\right)\right.}{\sum_{j=1}^{k} \exp \left(\operatorname{cosine}\left(f(\mathbf{x}), g\left(\mathbf{x}_{j}\right)\right)\right.}
$$

#### 1.3 Relation Network

Relation Network (RN) (Sung et al., 2018) [5] is similar to siamese network but with a few differences:

1) The relationship is not captured by a simple L1 distance in the feature space, but predicted by a CNN classifier *gϕ*. The relation score between a pair of inputs, **x**i and **x**j, is *r*i*j*=*gϕ*([**x**i*,**x***j]) where [.,.] is concatenation.

2) The objective function is MSE loss instead of cross-entropy,  because conceptually RN focuses more on predicting relation scores which is more like regression, rather than binary classification.
$$
\mathcal{L}(B)=\sum_{\left(\mathbf{x}_{i}, \mathbf{x}_{j}, y_{i}, y_{j}\right) \in B}\left(r_{i j}-\mathbf{1}_{y_{i}=y_{j}}\right)^{2}
$$
![relation-network](https://lilianweng.github.io/lil-log/assets/images/relation-network.png)

*Fig. 1-3. Relation Network architecture for a 5-way 1-shot problem with one query example.*

#### 1.4 Prototypical Networks

Prototypical Networks (Snell, Swersky & Zemel, 2017) [6] use an embedding function fθ to encode each input into a M-dimensional feature vector. A prototype feature vector is defined for every class c∈C, as the mean vector of the embedded support data samples in this class.
$$
\mathbf{v}_{c}=\frac{1}{\left|S_{c}\right|} \sum_{\left(\mathbf{x}_{i}, y_{i}\right) \in S_{c}} f_{\theta}\left(\mathbf{x}_{i}\right)
$$
![prototypical-networks](https://lilianweng.github.io/lil-log/assets/images/prototypical-networks.png)

*Fig. 1-4. Prototypical networks in the few-shot and zero-shot scenarios.*

The distribution over classes for a given test input x is a softmax over the inverse of distances between the test data embedding and prototype vectors.
$$
P(y=c | \mathbf{x})=\operatorname{softmax}\left(-d_{\varphi}\left(f_{\theta}(\mathbf{x}), \mathbf{v}_{c}\right)\right)=\frac{\exp \left(-d_{\varphi}\left(f_{\theta}(\mathbf{x}), \mathbf{v}_{c}\right)\right)}{\sum_{c^{\prime} \in \mathcal{C}} \exp \left(-d_{\varphi}\left(f_{\theta}(\mathbf{x}), \mathbf{v}_{c^{\prime}}\right)\right)}
$$
where dφ can be any distance function as long as φ is differentiable. In the paper, they used the squared euclidean distance. The loss function is the negative log-likelihood: 
$$
\mathcal{L}(\theta)=-\log P_{\theta}(y=c | \mathbf{x})
$$



### 2. Model-Based

Model-based meta-learning models make no assumption on the form of *Pθ*(*y*|**x**). Rather it depends on a model designed specifically for fast learning — a model that updates its parameters rapidly with a few training steps.

#### 2.1 Memory-Augmented Neural Networks (MANN)

A family of model architectures use external memory storage to facilitate the learning process of neural networks, including Neural Turing Machines and Memory Networks. Note that recurrent neural networks with only *internal memory* such as vanilla RNN or LSTM are not MANNs.

Taking the Neural Turing Machine (NTM) as the base model, Santoro et al. (2016) [7] proposed a set of modifications on the training setup and the memory retrieval (检索) mechanisms.

As a quick recap, NTM couples a controller neural network with external memory storage. 

![NTM](https://lilianweng.github.io/lil-log/assets/images/NTM.png)

*Fig. 1-5. The architecture of Neural Turing Machine (NTM).* 

MANN for Meta-Learning:

The training described in Santoro et al., 2016 [7] happens in an interesting way so that the memory is forced to hold information for longer until the appropriate labels are presented later.

![NTM](https://lilianweng.github.io/lil-log/assets/images/mann-meta-learning.png)

*Fig. 1-6. Task setup in MANN for meta-learning*

#### 2.2 Meta Networks

Meta Networks (Munkhdalai & Yu, 2017) [8] short for MetaNet, is a meta-learning model with architecture and training process designed for *rapid* generalization across tasks.



![meta-net](https://lilianweng.github.io/lil-log/assets/images/meta-network.png)

*Fig.1-7. The MetaNet architecture.*

### 3. Optimization-Based

Is there a way to adjust the optimization algorithm so that the model can be good at learning with a few examples? This is what optimization-based approach meta-learning algorithms intend for.

#### 3.1 LSTM Meta-Learner

The optimization algorithm can be explicitly modeled. Ravi & Larochelle (2017) [9] did so and named it “meta-learner”, while the original model for handling the task is called “learner”. 

#### 3.2 Model-Agnostic Meta-Learning

MAML, short for Model-Agnostic Meta-Learning (Finn, et al. 2017) [10] is a fairly general optimization algorithm, compatible with any model that learns through gradient descent.

#### 3.3 First-Order MAML

To make the computation less expensive, a modified version of MAML omits second derivatives, resulting in a simplified and cheaper implementation, known as First-Order MAML (FOMAML).

#### 3.4 Reptile

Reptile (Nichol, Achiam & Schulman, 2018) [11] is a remarkably simple meta-learning optimization algorithm. It is similar to MAML in many ways, given that both rely on meta-optimization through gradient descent and both are model-agnostic.



[1] https://lilianweng.github.io/lil-log/2018/11/30/meta-learning.html

[2] Bromley, Jane, et al. "Signature verification using a" siamese" time delay neural network." *Advances in neural information processing systems*. 1994.

[3] Koch, Gregory, Richard Zemel, and Ruslan Salakhutdinov. "Siamese neural networks for one-shot image recognition." *ICML deep learning workshop*. Vol. 2. 2015.

[4] Vinyals, Oriol, et al. "Matching networks for one shot learning." *Advances in neural information processing systems*. 2016.

[5] Sung, Flood, et al. "Learning to compare: Relation network for few-shot learning." *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition*. 2018.

[6] Snell, Jake, Kevin Swersky, and Richard Zemel. "Prototypical networks for few-shot learning." *Advances in Neural Information Processing Systems*. 2017.

[7] Santoro, Adam, et al. "Meta-learning with memory-augmented neural networks." *International conference on machine learning*. 2016.

[8] Munkhdalai, Tsendsuren, and Hong Yu. "Meta networks." *Proceedings of the 34th International Conference on Machine Learning-Volume 70*. JMLR. org, 2017.

[9] Ravi, Sachin, and Hugo Larochelle. "Optimization as a model for few-shot learning." (2016).

[10] Finn, Chelsea, Pieter Abbeel, and Sergey Levine. "Model-agnostic meta-learning for fast adaptation of deep networks." *Proceedings of the 34th International Conference on Machine Learning-Volume 70*. JMLR. org, 2017.

[11] Nichol, Alex, Joshua Achiam, and John Schulman. "On first-order meta-learning algorithms." *arXiv preprint arXiv:1803.02999* (2018).



