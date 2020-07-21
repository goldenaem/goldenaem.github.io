---
title: "Information Theory : Mutual Information with Deep Learning"
date: 2020-07-21 00:06:28 -0400
categories: 표상학습 pre-train self-supervised mutual-information information-theory entropy cross-entropy kl-divergence MINE DIM CPC information-bottleneck
---

정보이론(entropy, cross-entropy, KL divergence, MI, MINE, DIM, CPC)

해당 포스트는 정보이론 그 중에서도 Mutual Information(MI)과 MI를 이용한 표상학습에 대한 연구 및 구현 코드를 정리하였습니다. 

정보이론에 대한 개념은 [(1)](https://en.wikipedia.org/wiki/Information_theory) [(2)](https://namu.wiki/w/%EC%97%94%ED%8A%B8%EB%A1%9C%ED%94%BC) [(3)](https://ratsgo.github.io/statistics/2017/09/22/information/)를 참고. 해당 포스트에서는 내용을 간략히 정리하고 이를 어떻게 deep learning 연구에 활용하는지에 초점을 맞추려고 한다.

간단하게 정리하자면, entropy는 한 메시지에 들어갈 정보량을 비트수로 표현한 값이며, $$H(X) = -\sum_{i=1}^{n}p(x_i)\log_{2}{p(x_i)}$$이다.(이떄 $$p(x_i)$$는 discrete random variable $$x_i$$를 입력으로 확률값을 나타내주는 Probability Mass Functiond이다.)

2개의 random variable $$X$$와 $$Y$$에서 두 random variable의 joint entropy(정보량의 합)은 $$H(X,Y) = \mathbb{E}_{X,Y}[-\log(p(x,y))]=-\sum_{x,y}p(x,y)\log(p(x,y))$$이다.

deep learning에서 자주쓰이는 cross-entropy loss는 두개의 probability distribution p와 q에 대하여 $$H(p,q) = -\sum_{x \in X}p(x) \log q(x)$$이다. 두 확률 분포 p, q를 구분하는데 필요한 정보량(평균 비트 수)라고 한다. joint entropy와의 차이점을 정리하자면 joint entropy는 두 개의 random variable에 대하여 동일한 probability measure에 대한 값이고, cross entropy는 동일 random variable에 대한 다른 probability measure에 대한 값이다.

이와 관련해서 두 확률분포사이의 거리(엄밀하게는 거리라고 할 수 없음)를 계산하는 함수로 Kullback-Leibler divergence(KLD)가 있다. $$D_{KL}(\mathbb{P}\|\|\mathbb{Q}) = \sum_{x}P(x) \log {P(x) \over Q(x)}$$로 표현하며, deep learning에서는 Variational AutoEncoder 등에서 다른 분포를 통해 이상적인 분포를 근사시켜 샘플링하는 방법으로 사용됨.

2개의 random variable중에 하나가 주어졌을 때, 다른 하나의 정보량을 나타내는 conditional entropy(조건부 정보량)은 $$H(X \| Y) = \mathbb{E}_{Y}[H(X \| y)] = -\sum_{y \in Y}p(y)\sum_{x \in X}p(x\|y)\log p(x\|y) = - \sum_{x,y}p(x,y)\log p(x\|y)$$로 표현할 수 있다. 

해당 포스트에서 중점적으로 다루고자하는 내용인 mutual information은 상호 정보량이며, 다른 random variable를 관찰하여 하나의 random variable에 대하여 얻을 수 있는 정보량을 말한다. 쉽게 말해 두 random variable의 정보량의 intersection(교집합), 공유하고 있는 정보량이라고 해석할 수 있다. $$I(X;Y) = \sum_{x,y}p(x,y) \log {p(x,y)\over p(x)p(y)}$$ (joint entropy에서 사용하는 comma와 mutual information의 semi-colon은 다른 의미임) 또한 correlation과 다르게 MI는 두 변수 사이의 non-linear한 통계적 dependency를 측정하므로 true dependence의 measure로 사용가능 하다.[관련연구](https://www.pnas.org/content/111/9/3354.short)

위의 내용들은 아래와 같은 여러 특성을 가진다.(Gray, Entropy and Information Theory 참조)
$$0 \leq H(X\|Y) \leq H(X)$$  
$$I(X;Y) = H(X) + H(Y) - H(X,Y) = H(X) - H(X\|Y) = H(Y) - H(Y\|X)$$  
$$I(X;Y) = D_{KL}(P_{X,Y} \|\| P_X \times P_Y)$$ X, Y의 MI는 X, Y와 joint와 product of marginal의 KLD  
$$0 \leq I(X;Y) \leq min(H(X), H(Y))$$  
$$I(f(X);g(Y)) \leq I(X;Y)$$  
$$I(f(X);g(Y)\|Z) \leq I(X;Y\|Z)$$  
$$H(f(X) \| X) = 0$$  
$$H(X, f(X)) = H(X)$$  
$$H(X) = H(f(X)) + H(X\|f(X))$$  
$$I(X;f(X)) = H(f(X))$$  
$$H(X\|g(Y)) \geq H(X\|Y)$$  
$$I(f(X) ; g(Y)) \leq I(X;Y)$$  
$$H(X, f(X,Y) \|Y) = H(X\|Y)$$  
$$H(X\|Y) \geq H(X\|Y,Z)$$  
$$I(X;Y\|Z) + I(Y;Z) = I(Y;(X,Z))$$  

위에 정리한 내용을 이해를 위해 벤다이어그램으로 그리면 아래와 같다.

![venn-diagram_xy](/assets/images/it-xy-venn.png)

discrete random variable X,Y,Z에 대하여 벤다이어그램을 그리면 아래와 같다.

![venn-diagram_xyz](/assets/images/it-xyz-venn.jpg)

기본적인 개념과 특성에 대하여 정리하였다. 이제 이 중에 Mutual Information을 어떤식으로 deep learning과 접목하였는지에 집중하여 다루고자 한다.

Mutual Information은 상호 정보량으로 두 변수 사이의 공유되는 정보량이라고 생각할 수 있다. 이는 random variable의 차원이 높아지면(continuous하고 high-dimensional setting에서는) 계산이 intractable하다고 한다.(정확한 계산은 summation이 정확히 계산되는 discrete variable과 probability distribution을 아는 제한된 문제일때만 가능) 

이 때, MINE은 $$I(X;Y) = D_{KL}(P_{X,Y} \|\| P_X \times P_Y)$$를 활용하여 KLD의 dual formulation을 사용하여 MI estimator를 사용하여 general-purpose parametric neural estimator를 만들어 학습시키는 방법을 제안하였다. 

$$I(X;Z) \geq I_{\Theta}(X,Z)$$로 I(X;Z)의 lower bound를 neural information measure로 다음과 같이 정의한다.
$$I_{\Theta}(X,Z) = \sup_{\theta \in \Theta}\mathbb{E}_{P_{XZ}}[T_\theta] - \log (\mathbb{E}_{P_X \otimes P_Z}[e^{T_{\theta}}])$$
$$T_{\theta}$$는 $$T_{\theta} : \mathcal{X} \times \mathcal{Z} \to \mathbb{R}$$인 neural network.

위의 식은 Donsker-Varadhan representation을 이용하여 MI의 lower bound를 설정한 것이며, 해당 lower bound를 maximize하여 MI를 높인다. 

<details><summary>증명</summary><p>


DV-representation이 f-divergence보다 tight한 lower bound이므로 해당 포스트에서 f-divergence는 생략함.
</p>
</details>

이런 방식을 GAN에 이용하면 

$$\min_{G} \max_{D} V(D,G) := \mathbb{E}_{\mathbb{P}_{X}}[D(X)] + \mathbb{E}_{\mathbb{P}_{Z}}[\log (1-D(G(Z)))]$$를 $$\arg \max_{G} \mathbb{E}[\log (D(G([\epsilon,c])))] + \beta I(G([\epsilon,c]);c)$$로 바꾸어 GAN을 학습하면 mode collapse를 완화시키는 목적함수를 제안할 수 있다.(이 때 $$Z = [\epsilon, c]$$, $$I(G([\epsilon, c];c) = H(G([\epsilon, c])) - H(G([\epsilon, c]) \| c)$$)

![GAN_with_MINE_result](/assets/images/gan_with_mine_result.jpg)

해당 포스트에서는 MINE이후 Mutual Information을 Maximize하여 표상학습(Representation Learning)에 사용한 논문 중에 DeepInfoMax(DIM)과 Contrastive Predictive Coding(CPC)에 대하여 정리하겠다.(디테일보단 MI를 표상학습에 어떤식으로 적용 시켰는지를 위주로 정리하곘음)

DIM

![DIM_model](/assets/images/dim_model_archi.png)

Infomax는 입력과 출력 데이터 간의 MI를 최대화하는 학습 방법론이다. 여기에 신경망 모델을 사용하여 표상학습을 수행한 연구가 DIM이다. 구조를 간단히 설명하면, positive input $$X$$와 negative input $$X'$$가 encoder $$C_{\psi}$$를 거친 후에, $$M \times M$$ feature map $$C_{\psi}(x) := \{ C_\psi^{(i)} \}_{i=1}^{M \times M}$$


위의 모델 구조를 설명하기 앞서, loss함수를 먼저 정리한 후에 모델을 살펴보겠다.


위 그림에서 positive input을 real이라고, negative input을 fake라고 하여 




DIM 논문에는 다양한 방법(local과 global의 )



유용한 representation을 찾아내는 것은 deep learning의 주요 과제중에 하나인데, DIM은 Local과 Global로 representation을 나누어 local은 해당 pa 


CPC

![CPC_model](/assets/images/cpc_model_archi.JPG)

CPC는 입력 sequence $$x_t$$에서 encoder를 통해 뽑아낸 feature $$z_t = g_{enc}(x_t)$$를 autoregressive model $$g_{ar}$$을 통해 context $$c_t = g_{ar}(Z_{\leq t}$$를 생성하여, 입력 $$x_{t+k}$$와 context $$c_t$$의 MI를 maximizing하는 방향으로 학습시킴. 즉 self-supervised learning으로 입력 데이터 내부적으로 다음 step의 정보를 맞춤으로써 데이터의 표상을 학습하도록 함. 이 때 $$x_t$$를 직접 predicting하는 것은 reconstruction task에 과적합되기도 하며, 논문에서 Mean Squared Error(MSE)나 cross-entropy를 사용하는게 효과적이지 않고, 어렵다는 주장을 한다.


Information Bottlenekc






***
  ref)

  https://en.wikipedia.org/wiki/Information_theory
  https://en.wikipedia.org/wiki/Cross_entropy
  https://namu.wiki/w/%EC%97%94%ED%8A%B8%EB%A1%9C%ED%94%BC
  https://ratsgo.github.io/statistics/2017/09/22/information/
  Martin, Elliot A., et al. "Network inference and maximum entropy estimation on information diagrams." Scientific Reports 7.1 (2017): 1-15.
  Gray, Robert M. Entropy and information theory. Springer Science & Business Media, 2011.
  Belghazi, Mohamed Ishmael, et al. "Mine: mutual information neural estimation." arXiv preprint arXiv:1801.04062 (2018).
  Kinney, Justin B., and Gurinder S. Atwal. "Equitability, mutual information, and the maximal information coefficient." Proceedings of the National Academy of Sciences 111.9 (2014): 3354-3359.