<!-- ---
title: "Mockingjay : Unsupervised Speech Representation Learning With Deep Bidirectional Transformer Encoders"
date: 2020-07-20 00:06:28 -0400
categories: 표상학습 pre-train self-supervised BERT Speech Mockingjay 
use_math : true
---

해당 포스트에서 정리할 논문은 [Mockingjay](https://arxiv.org/pdf/1910.12638.pdf)다. (2019년 10월 25일자 논문)
  "제이(Jay)는 한국어로 어치라고 불리는 새다. 앵무새, 구관조와 같이 사람의 목소리를 흉내낼 수 있다고 한다. Mock는 따라하면서 놀리다라는 뜻이니 Mockingjay 는 따라하면서 놀리는 어치. 실제로 있는 새는 아니고, 헝거 게임 세계관 내에서 만들어진 새다."[1]

NLP에서 사용되는 token기반의 기존 BERT를 speech domain에 적용시킨 사례는 이전에 vq-wav2vec이 있었다.(vq-wav2vec 추후에 정리할 예정) wav2vec라는 CPC기반의 self-supervised learning 방법론을 BERT와 접목하기 위해서 quantization(양자화)을 수행하여 continuous한 feature를 양자화한 token으로 변환시켜 BERT를 적용한 연구다.(wav2vec, CPC 추후에 정리할 예정) Mockingjay에서 이런 양자화 과정이 continuous한 speech의 특성에 위배된다고 주장하며 해당 연구를 제안하였다.

모델 구조는 아래 그림과 같다

![Mockingjay_model](/assets/images/mockingjay_model.jpg)

[이전 연구](https://arxiv.org/pdf/1803.09519.pdf)에 따라 acoustic feature에 positional encoding을 그대로 적용시키면 학습이 실패하는 경우를 보안하기 위하여, projection을 통한 후에 sinusoidal positional encoding 정보를 더한다. downsample은 길이가 긴 sequence를 모델이 수용하기 위해서 $$R_{factor}$$개의 연속된 frame들을 reshape하고, BERT의 output인 hidden에서 다시 원래 frame과 맞춰주는 과정을 통해 $$seq_len$$을 조정한다.

BERT의 구조를 그대로 가져와 사용하되, wave형태의 speech 데이터를 (mel) spectrogram으로 변환하고, 각 frame을 dynamic하게(preprocess에서 마스킹을 처리하지않고, trainiing단계에서 masking 결정) maksing하고 projection을 통하여  처리하여 사용하였다. 또한 maksing을 0부터 $$C_{num}$$개 연속적으로 적용하였다.

mask token대신 해당 frame값을 0으로 채워서 masking의 의미로 사용하며, BERT에서처럼 maksing 80%/random replace 10%/original 10% 전략을 취한다. 또한 BERT에서 사용한 Cross entropy loss대신 l1 reconstruction loss를 사용하며, NSP나 SOP같은 BERT의 inter-coherence loss는 오히려 학습에 방해되어 제외하였다고 한다.

코드는 BERT와 거의 동일하기 떄문에 추후 BERT code를 분석하는 내용을 정리하며 링크로 변환하겠다.


***
  ref)
  https://namu.wiki/w/%ED%97%9D%EA%B1%B0%20%EA%B2%8C%EC%9E%84:%20%EB%AA%A8%ED%82%B9%EC%A0%9C%EC%9D%B4
  Liu, Andy T., et al. "Mockingjay: Unsupervised speech representation learning with deep bidirectional transformer encoders." ICASSP 2020-2020 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP). IEEE, 2020.
  Sperber, Matthias, et al. "Self-attentional acoustic models." arXiv preprint arXiv:1803.09519 (2018).
  https://github.com/andi611/Self-Supervised-Speech-Pretraining-and-Representation-Learning
 -->
