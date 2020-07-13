---
title: "Signal Processing - Spectrogram"
date: 2020-07-14 00:06:28 -0400
categories: Signal_Processing spectrogram mel-spectrogram STFT Fourier waveform
---

신호처리에서 waveform은 소리 파동의 높이 값(float-type)을 정해진 sampling rate에 따라 기록한 배열임. 고로 sampling rate를 높이면 1초당 더 많은 element를 가진 디지털화 된 배열을 얻을 수 있다.

이때 Pitch는 음의 높낮이를 의미하는데, Linear한 수치가 아니라 Exponential하게 처리되어 인간의 audio 인식 시스템의 특성을 반영한 수치이다. 

Mel-scale은 pitch에서 발견한 사람의 audio 인지 기준을 반영시킨 scale변환 함수. <수식 : Mel(f) = 2595xlog(1+f/700)>

Mel Filter Bank는 mel-scale에서 Linear하게 구간을 N개로 나누어 구현한 triangular filter(window)를 말하는데, 주파수 영역이 Exponential하게 넓어짐
<Mel filter bank 사진>

Mel Spectrogram 추출 방법
1) WaveForm > Framing : Waveform(x축이 time-domain, y축이 신호의 세기)를 특정한 크기의 Window로 slicing하여(overlap stride)하여 frames을 생성
<사진> waveform이 slice되어 
2) Framing >  


***
	ref)
	 
	Chi, Po-Han, et al. "Audio ALBERT: A Lite BERT for Self-supervised Learning of Audio Representation." arXiv preprint arXiv:2005.08575 (2020).







