---
title: "Signal Processing - preprocessing"
date: 2020-07-14 00:06:28 -0400
categories: Signal_Processing spectrogram mel-spectrogram STFT Fourier waveform preprocessing
---

해당 포스트는 신호처리, 소리 데이터를 deep learning(pytorch)에 적용하기 위한 개념과 코드를 정리하고자 합니다.

신호처리에서 wave은 현실세계의 연속된 파형을 정해진 sampling rate에 따라 소리 파동의 값(float-type)으로 변환한 배열이다. 고로 sampling rate를 높이면 1초당 더 많은 element를 가진 배열을 얻을 수 있다. 또한
인간이 한번에 듣는 소리는 대략 0.25초 정도의 단위라고 한다.

Wave의 진폭을 amplitude(intensity는 amplitude와 다르게 절대적 크기만을 의미함)라고 하며, 둘 다 소리의 세기(loudness, decibel)와 연관된 값이라고 보면 된다.
Wave의 주기(period, 파동이 한번 진동하는데 걸리는 시간)에 따라 주파수(frequency, 초당 진동 횟수, Hz)가 결정되며, 주파수가 높으면(진동을 많이하면) 고음, 낮으면 저음의 소리를 가진다. 이와 연관된 값은 pitch인데, 이는 음의 높낮이를 의미함.
마지막으로 소리의 크기와 높이 이 외에, 흔히 음색이라고 부르는 값을 나타내는 정보를 tone, waveform, timbre라고 한다.
이를 정리하면 아래 그림과 같다.
![signal_quantities](/assets/images/signal_quantities.JPG)

그림에서 표현된 measured quantities와 perceive quantities의 차이는 관련된 수치를 Linaer하게 측정된 수치로 나타내는가 사람의 소리 인지 구조에 따라 Exponential하게 나타내는가의 차이이다. 따라서 perceive quantities는 사람의 소리 인지 시스템 자체가 exponential하기 때문에 그 시스템에 맞도록 변환된 값을 의미한다.  

![plt](/assets/images/pitch_loudness_timbre.JPG)

최근에 위와 같은 성질을 가진 소리 데이터를 1-dimensional 배열로 표현하여 Deep learning에 응용한 연구들이 있다.(PANNs,WaveNet,DeepVoice,Tacotron 추후에 정리할 예정) 그러나 이전에는 1-D time-domain의 wave를 소리의 특성을 더 compact하게 표현한 feature로 변환하기 위해 frequency-domain의 spectrogram으로 변환 및 전처리하여 사용하는 방법이 활발히 연구되었다.

time-domain의 데이터를 frequency-domain으로 바꾸는 방법으로 Fourier 변환을 사용한다. 보다 자세한 이론적 내용은 [링크](https://en.wikipedia.org/wiki/Fourier_transform), 시각적 이해를 위해선 [링크](https://www.youtube.com/watch?v=spUNpyF58BY&feature=youtu.be)를 참고.
쉽게 정리하면 시간 영역의 데이터를 주파수 성분으로 분리해내는 방법이다. 정해진 구간의 소리 데이터를 입력으로 받으면, 그 구간내에서 형성된 주파수 성분들을 골라주기 때문에 time-domain에서 쉽게 발견하기 어려웠던 frequency-domain에서 성질을 파악하기 쉬워진다. 여기서 한계점은 시간에 대한 연속성이 고려되지 않기 때문에(구간의 길이와 무관하게 주파수 성분을 계산하기 때문에) 주어진 wave에 대하여 한 번의 Fourier transform으로는 시간에 대한 정보를 반영하기 어렵다. 이를 해결하고자 Short-Time Fourier Transform(STFT)를 사용하여 최종적으로 deep learning에서 사용되는 spectrogram(mel-spectrogram)을 만드는 과정을 아래에 정리함.

1) wave를 특정 window_size로 slicing하며(overlap stride) frames을 생성한다.

![wav_slicing](/assets/images/wav_slicing.png)

2) 각 frame에 대하여 Hanning window를 적용한 후에 STFT를 적용하여 frequency-domain으로 변환한다.

![STFT](/assets/images/STFT_result.png)

밑에 코드에서 확인할 수 있듯이 예시의 파일을 STFT한 결과는 0부터 200정도의 범위에서 평균이 0.38정도를 가진 STFT결과의 figure이므로 위와 같은 그림이 그려진다. 이를 decibel 변환하면 아래와 같은 power spectrogram figure가 나온다. 이때의 spectrogram의 데이터 타입은 가로축 time, 세로축 frequency, 그리고 color-coding된 값은 해당 시간과 주파수에 대한 decibel 값을 나타내는 2차원 Matrix이다.

![powerSepc](/assets/images/power_spectrogram.png)

3) Mel-filterbank를 적용하여 spectrogram을 mel spectrogram을 변환한다. 이때 마찬가지로 decibel 변환을 해야 figure로 의미있는 결과를 받을 수 있다.

![melSpec](/assets/images/mel_spectrogram.png)

4) 마지막으로 y축에도 log-scale을 반영하여 log-mel-spectrogram을 구한다.(이런 방식이 deep learning에서 성능적인 면에 도움이 되는 경우도 존재함)

mel-filterbank에 대하여 잠깐 살펴보면, 우선 mel이란 mel-scale을 의미하며 위에서 언급한 음에 대한 perceive quantities처럼 사람의 소리 인지 시스템을 반영한 scale 변환 $$Mel(f) = 2595log(1+f/700)$$을 의미한다.

![mel-scale](/assets/images/mel-scale.png)

filterbank는 주파수에 따라 특정한 filter를 통해 해당 부분만 추출해내는 방법이다. mel-filterbank은 N개의 filterbank 구간들이 mel-scale로 Linaer하게 나눠 triangular filter를 적용하기 때문에 주파수 영역이 Exponential하게 넓어진다. 사람의 인지 구조가 저음역대에서 더 민감하게 반응하기 때문에 낮은 주파수 영역대에 filter를 세밀하게 가진다. 아래의 그림과 같이 색깔별 그래프를 가중치 곱하여 정보를 추출하기에, 노이즈가 제거된 정보를 얻을 수 있으며 filterbank간 overlap 되는 부분이 존재하므로 filtering된 결과 사이의 상관관계가 존재할 수 있다.

![mel-filterbank](/assets/images/Mel-filter-banks.jpg)

마지막으로 지금까지의 전처리 방식을 구현한 코드를 정리하고 이용하여 pytorch의 dataset, dataloader, 그리고 tranform을 구현하여 deep learning의 batch단위로 학습하는 방법을 정리하고자 한다.

	import numpy as np
	import matplotlib.pyplot as plt
	import librosa
	import torch

	audio_path = librosa.util.example_audio_file()
	y, sr = librosa.load(audio_path)
	x = np.arange(0, len(y), 1)

먼저 관련 package를 import한 후에, 예제 파일을 load한다. 이때 x는 y의 index용으로 사용하기 위한 변수.

	plt.plot(x,y)
	plt.xlable("Time")
	plt.ylabel("Amplitude")
	plt.show()

![wave](/assets/images/wav.png)

wave를 전체를 그려보면 위와 같다.

	start_point, n_fft = 3000, 2048
	plt.plot(x[start_point:start_point+n_fft], y[start_point:start_point+n_fft])
	plt.xlabel('Time')
	plt.ylabel('Amplitude')
	plt.show()

![wave_split](/assets/images/wav_split.png)

start_point ~ start_point+n_fft를 확대하여 그리면 위와 같이 파형으로 생긴 결과를 볼 수 있다.

	D = np.abs(librosa.stft(y))
	print(D.shape, np.max(D), np.min(D), np.mean(D))

	((1025, 2647), 216.45607, 5.8435834e-10, 0.38008934)

이번엔 wave 전체에 STFT를 적용하고 shape와 최대, 최소, 평균값을 구해보면 다음과 같다.


	D = np.abs(librosa.stft(y, n_fft=2048, hop_length=512))
	plt.plot(D[:,1])
	plt.xlabel("Frequency")
	plt.ylabel("Magnitude")

![short_freq](/assets/images/short_freq.png)

n_fft와 hop_length는 mel_spectrogram에서 자세히 정리. 위의 코드에서의 D는 (1025,2647) shape를 가지는 배열이고, 위에 그려진 결과는 그 중 2번째 frame에 대해 Fourier transform을 적용 시킨 결과이다. frequency가 낮은 (200이하) 영역의 주파수대에서 존재하는 wave라는 사실을 알 수 있다.


	librosa.display.specshow(D, y_axis='linear', x_axis='time')
	plt.title('Spectrogram')
	plt.colorbar(format="%+2.0f")
	plt.tight_layout()
	plt.show()

![STFT](/assets/images/STFT_result.png)

STFT의 결과를 그대로 그리면 Hz가 낮은 영역대에서만 조금씩 active 되는걸 확인할 수 있다.(자세히 보면 붉은 점이 보임) 이는 위의 1개의 window에 대해서만 적용된 결과에서 볼 수 있듯이 전반적으로 낮은 주파수 영역대에서 포진된 값이라고 볼 수 있다. 또한 1개의 frame에 STFT하여 나타낸 그래프를 시간대별로 적층해서 나타낸 결과이다.

	librosa.display.specshow(librosa.amplitude_to_db(D, ref=np.max), y_axis='linear', x_axis='time')
	plt.title('power Spectrogram')
	plt.colorbar(format="%+2.0f db")
	plt.tight_layout()
	plt.show()

![powerSepc](/assets/images/power_spectrogram.png)

decibel 변환($$L_B = 10 \log_{10} {B \over A}[dB] $$,$$A$$는 np.max)하여 그리면 다음과 같이 고루 active된 결과를 얻을 수 있다.

	mel_128 = librosa.filters.mel(sr=sr, n_fft=2048, n_mels=128)
	plt.figure(figsize=(15, 4));

	plt.subplot(1, 3, 1);
	librosa.display.specshow(mel_128, sr=sr, hop_length=512, x_axis='linear');
	plt.ylabel('Mel filter');
	plt.colorbar();
	plt.title('1. Our filter bank for converting from Hz to mels.');

	plt.subplot(1, 3, 2);
	mel_10 = librosa.filters.mel(sr=sr, n_fft=2048, n_mels=10)
	librosa.display.specshow(mel_10, sr=sr, hop_length=512, x_axis='linear');
	plt.ylabel('Mel filter');
	plt.colorbar();
	plt.title('2. Easier to see what is happening with only 10 mels.');

	plt.subplot(1, 3, 3);
	idxs_to_plot = [0, 9, 49, 99, 127]
	for i in idxs_to_plot:
		plt.plot(mel_128[i]);
	plt.legend(labels=[f'{i+1}' for i in idxs_to_plot]);
	plt.title('3. Plotting some triangular filters separately.');

	plt.tight_layout();

![melfilter](/assets/images/mel_filter.png)

첫번째 사진은 mel_filter의 갯수를 128개, 두번째는 10개를 설정한 그림이다. 위에 mel-filterbank 설명할때 보았던 triangular window mel-filterbank를 specshow하여 표현한 것이다(triangular window를 3차원으로 놓고 위에서 쳐다본다는 느낌으로 생각하면 된다.) 마지막 사진은 filter 중에 0, 9, 49, 99, 127 번 filter만을 나타낸 그림이다.

	plt.plot(D[:,1])
	plt.plot(mel_128.dot(D[:,1]))
	plt.legend(labels=['spectrum', 'mel-spectrum'])
	plt.show()   

![melspectrum](/assets/images/mel-spectrum.png)

D의 shape가 (1025, 2647)일 때, 2647이 time-domain에 대한 정보이고, 그 중 2번째 frame에 대하여 STFT결과(spectrum)과 mel-filterbank결과(mel-specturm)의 그래프를 나타낸 결과이다. mel-specturm의 길이가 128까지만 나타난 이유는 mel_128이 128개의 filter만을 사용하였기 때문이다.

	mel_spectrogram = librosa.feature.melspectrogram(y, sr=sr, n_mels=128)
	log_mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)

	plt.figure(figsize=(12,4))
	librosa.display.specshow(log_mel_spectrogram, sr=sr, x_axis='time', y_axis='mel')
	plt.title('mel power spectrogram')
	plt.colorbar(format="%+02.0f dB")
	plt.tight_layout()

![melSpec](/assets/images/mel_spectrogram.png)

최종적으로 2번쨰 frame에만 적용했던 mel-filter를 모든 frame에 적용하여 그리면 위와 같다. 이 역시 mel-spectrum을 frame(time-axis)에 따라 적층한 결과이다.

mel-spectrum에서 cepstral analysis를 적용하여 MFCC라는 coefficient를 추출하여 feature로 사용하는 방법론이 예전에는 자주 사용되었으나, 최근 연구에서는 mel-spectrogram이나 wave자체를 데이터로 사용해서 모델이 학습하도록 적용하는 경향이 늘고 있다.(MFCC에 대한 내용은 [여기]https://brightwon.tistory.com/11)를 참고)

이제 mel-spectrogram과 wave를 계산하는 방법을 살펴보았으니, [GTZAN](http://marsyas.info/downloads/datasets.html) 데이터셋을 사용하여 pytorch의 dataset, dataloader, transform를 사용한 generator를 만들어보자.

	root_dir = "./GTZAN"
	csv_list = []
	genres = []
	for genre in os.listdir(root_dir):
		if os.path.isdir(os.path.join(root_dir, genre)):
			genres.append(genre)
			for wav_file in os.listdir(os.path.join(root_dir, genre)):
				csv_list.append([os.path.join(root_dir, genre, wav_file), genre])
	df = pd.DataFrame(csv_list, columns=['path', 'genre'])
	df.to_csv(os.path.join(root_dir, "meta.csv"), index=False)

우선은 GTZAN데이터셋을 다운받고, 코드와 동일한 directory에 unzip한 후에 GTZAN의 파일 path와 genre를 csv로 저장한다. 이 부분이 없이 코드를 작성할 수 있으나 편의상 meta 정보를 담은 csv를 만들어서 사용하려함.

	genre_idx_dict = dict()
	g = np.array(genres)
	for genre in genres:
		# genre_idx_dict[genre] = np.argwhere(g==genre)[0] # for index
		genre_idx_dict[genre] = np.array(g==genre, dtype=np.int64) # for one-hot

genre_idx_dict은 label에 대한 정보를 처리할 때 편의를 위해 만든 dictionary이며, 모델의 loss 타입에 따라 index타입이나 one-hot타입을 사용하면 된다.

	class GTZANDataset(Dataset):
	"""
	GTZAN Dataset.
	"""
		def __init__(self, root_dir, csv_file, sample_rate=32000, wave_size = 32000*10, n_fft=2048, hop_length=512, win_length=2048, n_mels=128 transform=None):
			"""
			Args:
			  csv_file (string) : GTZAN의 meta csv가 저장된 path
			  transform (callable, optional) : 샘플에 적용될 optional transform
			  wave_size : default wave crop size
			  spec_size : default spectrogram crop size(according to time axis)
			"""
			self.path_genre_dict = pd.read_csv(csv_file)
			self.root_dir = root_dir
			self.transform = transform
			self.sample_rate = sample_rate
			self.wav_npy_path = os.path.join(self.root_dir, 'wavs.npy')
			self.spec_npy_path = os.path.join(self.root_dir, 'specs.npy')
			self.genre_npy_path = os.path.join(self.root_dir, 'genres.npy')

			# prepare(save/load) numpy array before batch iter
			if os.path.exists(self.wav_npy_path) and os.path.exists(self.spec_npy_path) and os.path.exists(self.genre_npy_path):
				self.wavs = np.load(self.wav_npy_path)
				self.specs = np.load(self.spec_npy_path)
				self.genres = np.load(self.genre_npy_path)
			else:
				self.wavs, self.specs, self.genres = [], [], []
				for i in range(len(self.path_genre_dict)):
					wav_name, genre = self.path_genre_dict.iloc[i]
					wav, sr = librosa.load(wav_name, sr=self.sample_rate)
					genre_label = genre_idx_dict[genre]
					spec = librosa.feature.melspectrogram(y=wav[:wav_size], sr=sr, n_fft=n_fft, hop_length=hop_length, win_length=win_length, n_mels=n_mels)
					spec = librosa.power_to_db(spec, ref=np.max)
					self.wavs.append(wav[:wav_size])
					self.specs.append(spec)
					self.genres.append(genre_label)
				self.wavs = np.array(self.wavs)
				self.specs = np.array(self.specs)
				self.genres = np.array(self.genres)
				np.save(self.wav_npy_path, self.wavs)
				np.save(self.spec_npy_path, self.specs)
				np.save(self.genre_npy_path, self.genres)

		def __len__(self):
			return len(self.path_genre_dict)

		def __getitem__(self, idx):
			if torch.is_tensor(idx):
				idx = idx.tolist()

			sample = {
				'wav' : self.wavs[idx],
				'spec' : self.specs[idx],
				'genre' : self.genres[idx],
				'sr' : self.sample_rate
			}

			if self.transform:
				sample = self.transform(sample)
			return sample

다음은 pytorch의 Dataset을 상속받아 GTZAN용 dataset 클래스를 선언하였다. GTZANDataset이 선언되면 초기화 method인 init에 의해 정해진 root에 wave와 mel-spectrogram이 numpy array형태(.npy)로 저장 되어있는지 체크한 후에, load하거나 새롭게 만들어 저장한다. (해당 과정은 데이터셋 크기에 따라 다소 시간이 소요 될 수 있음) len는 dataset의 길이를 나타내주는 method이며, getitem은 추후 dataloader에서 sampling할 때 데이터를 가져오는 method이다. 이떄 transform이 설정되어 있다면, transform을 거쳐 sampling하게 된다.

데이터 처리방법은 크게 2가지로 볼 수 있는데, 위의 코드와 같이 batch iter를 call하기 전 Dataset 인스턴스를 생성할 때 wave와 mel-spectrogram을 처리하여 메모리에 저장해두는 경우와 batch iter가 call되면 getitem method에서 wave파일을 읽어 librosa연산을 수행하는 방법이 있다. 전자는 컴퓨터의 메모리를 많이 소요하는 대신 CPU연산을 줄여 학습의 속도를 늘릴 수 있고, 후자는 대용량 데이터의 경우 batch iteration마다 file I/O를 통해 데이터를 읽어 오기에  컴퓨터 메모리가 부족한 상황에서도 처리가능하나 학습속도가 batch를 구성하는데 시간을 많이사용하여 현저히 느려지는 단점이 있다.(상황에 따라 하되, GPU/TPU를 사용하는 상황에서는 전자를 사용하는게 좋음)

	class RandomCrop(object):
		"""
		random crop from sample data
		mode = "wav" or "spec" or "comb"
		"""
		def __init__(self, wav_crop_size, spec_crop_size, mode):
			self.wav_crop_size = wav_crop_size
			self.spec_crop_size = spec_crop_size
			self.mode = mode

		def __call__(self, sample):
			"""
			Assuming that the sample is wave/mel-spectrogram, crop randomly
			"""
			try:
				wav, spec, genre = sample['wav'], sample['spec'], sample['genre']
				if self.mode == 'wav' or self.mode == 'comb':
					wav_len = len(wav)
					if wav_len > self.wav_crop_size:
						start_frame = np.random.randint(0, wav_len - self.wav_crop_size)
						wav = wav[start_frame:start_frame+self.wav_crop_size]
					elif wav_len == self.wav_crop_size:
						pass
					else :
						raise Exception("wave length is shorter than crop size!")

				if self.mode == 'spec' or self.mode == 'comb':
					spec_len = spec.shape[-1]
					if spec_len > self.spec_crop_size:
						start_frame = np.random.randint(0, spec_len - self.spec_crop_size)
						spec = spec[:, start_frame:start_frame+self.spec_crop_size]
					elif spec_len == self.spec_crop_size:
						pass
					else:
						raise Exception("spectrogram length is shorter than crop size!")

				return {'wav':wav, 'spec':spec, "genre":genre}

			except Exception as error:
				print("RandomCrop Transform error : " + repr(error))

예시로 customize crop transform 클래스를 만들면 위와 같다. 다소 복잡해보이지만, Dataset 클래스에서 getitem의 마지막에 transform이 존재하면 transform을 호출하게 되는데, 이때 미리정의한 transform 클래스의 call method가 호출된다. 따라서 sample이라는 입력을 어떻게 처리할지 자유롭게 transform 로직을 작성하여 사용할 수 있다. pitch를 변경하는 방식으로도 transform이 가능한데, 이떄 librosa.effects.pitch_shift 등을 사용하면 됨)


	gtzan_dataset = GTZANDataset(
		root_dir=root_dir, 
		csv_file=os.path.join(root_dir, 'meta.csv'),
		transform = transform.Compose([
				RandomCrop(wav_crop_size = 320000, spec_crop_size = 600, mode="comb"),
			])
		)
	dataset_size = len(gtzan_dataset)
	indices = list(range(dataset_size))
	split = int(np.floor(0.3 * dataset_size)) # 0.3 is test set ratio
	shuffle_dataset = True
	randome_seed = 42
	if shuffle_dataset:
		np.random.seed(random_seed)
		np.random.shuffle(indices)
	trn_indices, tst_indices = indices[split:], indices[:split]

	trn_gtzan_dataset = torch.utils.data.sampler.SubsetRandomSampler(trn_indices)
	val_gtzan_dataset = torch.utils.data.sampler.SubsetRandomSampler(val_indices)

    trn_gtzan_dataloader = DataLoader(gtzan_dataset, batch_size = 16, drop_last=True,sampler=trn_gtzan_dataset)
    val_gtzan_dataloader = DataLoader(gtzan_dataset, batch_size = 16, drop_last=True,sampler=val_gtzan_dataset)    	


dataset 인스턴스를 생성할 때 transform 클래스를 compose로 여러 transform을 조합하여 사용할 수 있다. (위의 예시에서는 1개만 사용함) 또한 **torchaudio.transform**를 이용하면 미리 정의된 transform함수를 편리하게 사용할 수 있다.(ex. torchaudio.transforms.FrequencyMasking, torchaudio.transforms.TimeMasking 등)~~(transform의 예시를 위해 만들었을 뿐 torch에서 미리 정의된 함수 맞춰 사용하는게 더 좋다.)~~ 

	for batch in trn_gtzan_dataloader : 
		wavs, specs, genres = batch['wav'], batch['spec'], batch['genre']
		print(wavs.shape, specs.shape, genres.shape)

dataloader의 사용법은 매우 단순하다. for loop에서 batch 단위로 sample들을 가져오면 위와 같이 미리 선언해둔 key를 통해 접근하여 사용하면 된다. 

관련된 코드의 정리판은 [링크](https://github.com/goldenaem/)에 jupyter notebook형태로 올려두었으니 참고. 

마지막으로 Wave나 mel-spectrogram등을 사용할때 전체 데이터에 대해서 normalize등을 적용하면 scale 관련 이슈를 해결할 수 있어서 사용해보는게 좋다.

***

	reference)

	Compadre.org. [online] Available at: <https://www.compadre.org/osp/EJSS/4485/270.htm> [Accessed 14 July 2020].
	onosokko.co.jp. [online] Available at : <https://www.onosokki.co.jp/English/hp_e/whats_new/SV_rpt/SV_7/sv7.htm> [Accessed 14 July 2020].
	wikipedia.org. [online] Available at: <https://en.wikipedia.org/wiki/Fourier_transform> [Accessed 14 July 2020].
	3Blue1Brown. "푸리에 변환이 대체 뭘까요? 그려서 보여드리겠습니다." Youtube. Youtube, 27 Jan 2018. 14 July 2020, https://www.youtube.com/watch?v=spUNpyF58BY&feature=youtu.be
	haythamfayek.com. [online] Available at: <https://haythamfayek.com/2016/04/21/speech-processing-for-machine-learning.html> [Accessed 14 July 2020].
	McFee, Brian, Colin Raffel, Dawen Liang, Daniel PW Ellis, Matt McVicar, Eric Battenberg, and Oriol Nieto. "librosa: Audio and music signal analysis in python." In Proceedings of the 14th python in science conference, pp. 18-25. 2015.
	SKplanet Tacademy. "토크ON74차. 디지털신호처리 이해 | T아카데미." PlayList. Youtube. Youtube, 14 May 2020. 14 July 2020.
	Brightwon.tistory.com. [online] Available at: <https://brightwon.tistory.com/11> [Accessed 14 July 2020].
	towardsdatascience.com. [online] Available at: <https://towardsdatascience.com/getting-to-know-the-mel-spectrogram-31bca3e2d9d0> [Accessed 14 July 2020].
	Sturm, Bob L. "The GTZAN dataset: Its contents, its faults, their effects on evaluation, and its future use." arXiv preprint arXiv:1306.1461 (2013).