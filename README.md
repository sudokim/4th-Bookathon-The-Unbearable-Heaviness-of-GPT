# [제4회 대학생 AIXBookathon 대회]

![image](https://user-images.githubusercontent.com/5147122/213642918-bf47b64a-1b8e-44da-8df8-b374baac11c8.png)


AI × Bookathon 대회는 작품 기획,  AI 활용, 글쓰기 등을 담당하는 3명 이상의 인원이 팀을 구성하여, 아이디어 도출, 데이터 수집, 머신 러닝, AI 글쓰기 및 문장 다듬기를 통해 한 편의 문학 작품을 완성하는 대회입니다.

이번 대회에서는 `담대한`이라는 주제를 활용해 약 2만자 분량의 글을 작성해야 했습니다.

<br>

<br>


## Overview

![image](https://user-images.githubusercontent.com/5147122/213643213-55271904-d67d-466f-850b-9b6669e9895e.png)

🥉 **팀 참을 수 없는 GPT의 묵직함([김영빈](https://github.com/ybkim3603), [김준영](https://github.com/ybkim3603), [김현수](https://github.com/sudokim))**

저희가 [**행복으로의 내디딤**]이라는 제목의 수필을 완성했고, 우수상을 수상했습니다.

발표 자료는 여기에서 확인하실 수 있습니다.

<br>

<br>



## Data

- 브런치 (감성 에세이, 산문, 수필, 에세이, 생각, 인생 등의 수필 관련 키워드를 가진 포스팅)
- 문학광장 (글틴 수필 카테고리, 글틴 명예의 전당)

<details>
<summary><b>포스팅 단위 전처리</b></summary>
<div markdown="1">
  
> 품질 낮은 포스팅을 학습에서 제외 시키기 위함

1. 한글 비율이 일정 이하인 포스팅 제거
2. 특정 키워드를 포함한 데이터 제거
3. 특정 저자의 포스팅 제거
4. 중국어와 일본어를 포함한 포스팅 제거
5. 길이가 너무 짧은 포스팅 제거
6. [kss](https://github.com/likejazz/korean-sentence-splitter)로 문장 분리
</div></details>

<details>
<summary><b>문장 단위 전처리</b></summary>
<div markdown="1">
  
> 문장에서 모델의 학습에 방해되는 요소를 제외 시키기 위함

1. Email, URL, hashtag(#), mention(@) 제거
2. 괄호 및 괄호 안의 내용 제거
3. 자음 혹은 모음만 있는 문자 제거
4. 반복되는 특수문자와 공백 제거
5. 개행문자 제거
6. 특이한 형태의 특수 기호를 일반적인 형태로 변경
</div></details>

<br>

<br>

## **Questions**

### 1. **자연스럽고 유의미한 문장을 생성하는 방법이 뭘까?**

[[A Contrastive Framework for Neural Text Generation](https://arxiv.org/abs/2202.06417)]과 [[Contrastive Search Is What You Need For Neural Text Generation](https://arxiv.org/abs/2210.14140)] 두 논문을 참고해 Contrastive Search를 활용해 문장을 생성했습니다.

<br>

### 2. **긴 문장을 생성하는 동안 어떻게 문맥을 유지할 수 있을까?**

해당 문제을 해결하기 위해 저희는 Sliding Window, Input-Target, Keyword 세 가지 방법을 생각했습니다.

Sliding Window는 긴 문서를 학습하기 위해 반드시 필요하다 생각했고, Input-Target과 Keyword의 유효성을 검증하기 위해 간단한 실험을 진행 후, Keyword를 학습 방식으로 선택하였습니다.


<details>
<summary><b>실험 및 학습 방법 디테일</b></summary>
<div markdown="1">

- Sliding Window

    GPU 메모리의 한계로 전체 1,024 토큰을 입력으로 사용할 수는 없었으며, 최대 384 토큰까지만 학습이 가능했습니다. 따라서, 문서의 처음부터 끝까지 학습하기 위해 Sliding Window를 사용하여 정해진 문장 수로 전체 문서를 나누어 학습을 진행하였습니다.
    
- Input-Target
    
    모델이 앞 문장이 주어졌을 때, 뒤 문장을 이어서 생성하는 방법을 학습할 수 있도록 하기 위해 전체 문장 중 앞 일부분은 입력 프롬프트로, 나머지 문장은 프롬프트가 주어졌을 때 생성해야 하는 결과로 설정하여 학습을 진행하였습니다. 이때, 입력 프롬프트에 해당하는 토큰은 손실 함수를 계산할 때 제외하였습니다.
    
- Keyword
    
    모델이 긴 글을 생성하는 동안 문맥을 일정하게 유지할 수 있도록 앞에 생성한 글에서 키워드를 추출하여 입력 문장 앞에 넣어주었습니다. 키워드를 추출하기 위해 그래프 기반 비지도 학습 단어 추출 툴킷인 [KR-WordRank](https://github.com/lovit/KR-WordRank)를 활용하였습니다.
    

빠르게 학습하기 위해 [skt/kogpt2](https://github.com/SKT-AI/KoGPT2) 모델 사용해 실험을 진행했습니다.

[[A Contrastive Framework for Neural Text Generation](https://arxiv.org/abs/2202.06417)] 논문을 참고해 기존 reference와 유사도를 비교하는 metric을 사용하지 않고 generation quality를 평가할 수 있는 metric을 사용했으며, 추가로 Human evaluation을 진행했습니다.

**Metric**

1. `rep-n` : $100-\left(1.0- {{|\text{unique n-grams}(\hat x)|} \over {|\text{total n-grams}(\hat x)|}}\right)$, 전체 문장에서 반복되는 $n$-gram의 수
2. `Diversity` : $\Pi^4_{n=2} \left(1.0-{\text{rep-n} \over 100} \right)$, `rep-2`, `rep-3`, `rep-4`를 모두 고려한 지표로, 생성된 토큰의 다양성을 의미
3. `Keyword Recall` : 생성된 문장에서 입력한 키워드의 재현율
4. `Coherence` : ${\text{BERT} \left( P \right) \cdot \text{BERT} \left( W \right)} \over { || \text{BERT} \left( P \right) || \times || \text{BERT} \left( W \right) ||}$, [Sentece-BERT](https://github.com/snunlp/KR-SBERT)를 활용한 프롬프트와 생성된 문장 사이 코사인 유사도
5. `Perplexity` : $\sqrt[N]{1 \over {P\left( w_1, w_2, \ ..., w_N \right)}}$, 모델이 생성한 토큰에 대한 확실함을 나타내는 지표

> $P$: 프롬프트 문장, $W=\{w_1, w_2, \ ..., w_N\}$: 생성된 문장

**Result**

|  | rep-2 | rep-3 | rep-4 | Diversity | Keyword Recall | Coherence | Perplexity |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Base | 3.3572 | 1.8210 | 1.4898 | 0.9520 | - | 0.5141 | 35.917 |
| Keyword | 1.9189 | 1.2608 | 1.1378 | 0.9707 | 0.9475 | 0.6803 | 25.600 |
| Input-Target | 2.3295 | 1.3471 | 1.2223 | 0.9639 | 0.9470 | 0.6923 | 26.238 |
| Keyword + Input-Target | 3.7116 | 2.1753 | 1.8742 | 0.9440 | - | 0.3531 | 21.548 |

</details>

<br>

### 3. **16GB GPU 위에서 어떻게 LLM을 학습시킬 수 있을까?**

저희는 1.2B parameters를 가진 [**ko-gpt-trinity**](https://huggingface.co/skt/ko-gpt-trinity-1.2B-v0.5)를 모델로 선택했습니다.

하지만 사용할 수 있는 GPU가 16GB T4로 제한되어 있어, 효과적으로 학습하기 위해 다음과 같은 방법을 사용했습니다.

1. Automatic Mixed Precision(AMP)
2. Gradient Accumulation

<br>

### 4. **모델이 `담대함`을 알까?**

`담대함`이라는 주제가 모델에게 생소할 것이라고 느껴 브런치에서 해당 주제어를 포함한 글 약 700개를 크롤링해 추가로 학습시켰습니다. 

<br>

<br>

## Writing process

1. **Initial prompt와 Initial Keyword를 입력**
2. **모델이 글을 생성**
3. **생성된 문장으로부터 추출된 키워드 + 앞선 n개의 문장 제공**
4. **이때 사람의 개입을 최소화하고 키워드를 추가하는 방식으로 글 방향을 조절**
5. **2~4 반복**
<details>
<summary>
<b>키워드로 글 방향 조절하는 예시</b>
</summary>

`Keyword`: 미래 희망

`Prompt`: 행동은 강력한 힘을 발휘한다.그 용기에 감탄하고 힘을 얻는다. 두려움을 이기기 힘들다면 두려워하지 말고 행동하면 된다.

> 그 행동을 통해 희망을 찾을 수 있다는 사실을 알아야 한다. 희망이 절망으로 변할까 말까 고민하는 사람들이 많은데, 희망은 긍정적인 단어라고 생각하기로 했다. 긍정의 힘이란 '희망'과 연관되어 있기 때문이다. 희망을 이야기한 글들을 읽다 보면 마치 내 마음을 대변하는 것 같아 눈물이 날 뻔했다. 그 감정이 무엇인지는 모르겠지만, 내가 원하는 일을 해냈다는 안도감이랄까... 나는 아직 살아있음을 느낀다. 내 인생의 여정 위에 서 있는 나 자신이 한 발자국씩 성장하고 있음에 감사하게 된다. 그 순간을 놓 치고 싶지 않아 발버둥 치며 하루하루를 견뎌야 하는 날들이 있을

<br> 

`Keyword`: 과거 실패

`Prompt`: 행동은 강력한 힘을 발휘한다.그 용기에 감탄하고 힘을 얻는다. 두려움을 이기기 힘들다면 두려워하지 말고 행동하면 된다.

> 실패를 두려워하는 사람이 있을테니. 그 두려움이라는 놈이 내 발목을 잡을지도 모른다는 생각이 드니 두렵기도  하구나 라고 생각하기로 했다. 실패는 성공의 어머니라는 말이 떠올랐다. 내가 원하는 것을 얻지 못했을 때 느끼는 좌절감이 두려워 도전하지 못했던 나를 반성하게 만든 것 같다. 그래서 다시 시작해보기로 했다. 내 인생 최고의 실패로 기억되는 순간을 꼽아보자면 초등학교 5학년 때였나 보다. 엄마와 외삼촌이 이혼을 하셨다. 어린 나이였지만 그 순간을 돌이켜보면 참 서글펐다. 엄마가 집을 나 간 지 얼마 되지 않았을 때 아빠는 술을 마시고 들어오신 적이 있었다. 그때 난 처음으로 엄마의 마음을 헤아

</details>

<details>
<summary>
<b>생성된 문장 예시</b>
</summary>
    
> 을지로에서 북촌 한옥마을까지 걷다 집으로 향했다. 걷는 동안 머릿속에 맴돌던 질문들을 정리하고 싶었기 때문이다. '지금까지와는 다른 삶을 살고 있는 나를 상상하면 어떤 기분이 들까?' '어떻게 살아야 잘 살 수 있을까?' '앞으로 어떻게 살아야 잘 사는 걸까' 이런 생각들이 꼬리를 물기 시작했다. 나는 이러한 질문들에 대해서 생각해 보려고 노력하고 있었다. 북촌의 좁은 골목 사이를 지나가는 사람들과 같은 나를 보며, 지금과는 다른 삶을 살 수 있는 기회는 있을까 하는 생각이 들었다. 한참을 걷다가 고개를 들어 하늘을 보았다. 구름 한 점 없이 푸르른 하늘과 햇무리가 내 눈을 사로잡았다.
> 

> '실패'가 두려워 아무것도 하지 않는 겁쟁이가 되지는 않았으면 좋겠다. 실패를 두려워하지 말고 일단 부딪혀보자. 그리고 그 실패로 무엇을 얻을지 상상해보는 거다. 실패에 익숙해져야 그다음 도전이 두렵지 않을 테니까. 두려움이라는 감정은 그냥 두면 사라질 감정이다. 그 감정이 올라올 때 어떻게 대처해야 할지는 오롯이 나에게 달려있다고 해도 과언이 아니다. 두려움과 걱정은 내 안에 존재해선 안 되는 감정인 것 같다. 나를 응원하는 마음가짐이 생겨나기 시작했고 결국 조금씩 성장해가는 나를 발견했다. 그래서 더 두려워도 담대하게 이겨낼 수 있었던 것 같다. 오늘도 어영부영 버티고 있는 나에게 박수를 보내고 싶다. 이 글을 읽는 당신도 그러길 바라면서. 담대하게, 포기하지 않길 바라며 응원한다.
> 
    
</details>

<br>

<br>

## Result

**[행복으로의 내디딤]**

화자는 부정적인 생각에서 벗어나고 진취적인 삶을 살기 위해

자신을 성찰하면서 행복해지는 삶의 방향과 방법에 대해 고민하고,

행복을 위해 한 발자국 내딛기로 마음 먹는다.

<details>
  <summary><b>[글의 구성]</b></summary>
  
  1. 행복해지는 방법
  2. 거울 속의 나를 바라보며
  3. 행복해지기 위해서는
      1. 도전은 두려운 일이 아니다
      2. 내 안의 두려움을 포용하자
      3. 실패에 대한 두려움을 극복하자
      4. 두려움을 극복하기 위해 행동하자
      5. 글을 쓰자
      6. 위기를 기회로
  4. 결론
  
</details>

글의 원문은 여기에서 확인할 수 있습니다.
    
<br>

<br>


## Usage

### Folder

```
┌── data
│   ├── preprocess.ipynb: 데이터셋 & 문장 단위 전처리 코드
│   ├── preprocess_dataset.py: Train/Valid/Test split
│   ├── scraping.ipynb: 데이터 크롤링 코드
│   └── split_sentences.py: 문장 분리 코드
├── src
│   ├── dataloader_collate.py: DataLoader 취합 및 tokenization 코드
│   ├── datamodule.py: 기본 프롬프트 DataModule
│   ├── datamodule_input_target.py: Input - Target DataModule
│   ├── datamodule_keyword.py: 키워드 사용 DataModule
│   ├── datamodule_sliding_window.py: Sliding Window DataModule
│   ├── keywords.py: 키워드 추출 도구
│   ├── model.py: GPT-2/GPT-Trinity 모델 코드
│   ├── tf-idf.py: (미사용) TF-IDF 계산 코드
│   └── utils.py: (미사용) 로깅 코드
├── COPYING: 라이센스
├── README.md
├── bookathon_writer.py: 긴 문장 작성 코드
├── evaluate.ipynb: 7개 metric 평가 코드
├── generate_gpt.py: GPT-2 대량 문장 생성 코드
├── generate_gpt_trinity.py: GPT-Trinity 대량 문장 생성 코드
├── requirements.txt
├── test_gpt.py: GPT-2 테스트 코드
├── test_gpt_trinity.py: GPT-Trinity 테스트 코드
├── train_gpt.py: GPT-2 학습 코드
├── train_gpt_trinity.py: GPT-Trinity 학습 코드
└── 참을-수-없는-GPT의-묵직함-발표자료.pdf
```

### Getting started

```
evaluate==0.4.0
nltk==3.8.1
notebook==6.5.2
numpy==1.23.5
pandas==1.5.2
python==3.10.8
pytorch-lightning==1.8.1
pytorch==1.13.1
rich==12.5.1
krwordrank==1.0.3
kss==4.4.0
scikit-learn==1.2.0
selenium==3.141.0
sentence-transformers==2.2.2
sentencepiece==0.1.97
soynlp==0.0.493
transformers==4.24.0
```

### Training

`python train_gpt.py -h` 또는 `python train_gpt_trinity.py -h`를 사용하여 매개변수의 도움말을 표시할 수 있습니다.

다음은 GPT-Trinity를 keywords로 학습하는 예시입니다.

```bash
python train_gpt_trinity.py \
    --datamodule keywords \
    --max_length 384 \
    --train_sep_token \
    --batch_size 1 \
    --gradient_accumulation 8
```

### Test/Generate

`python test_gpt.py -h` 또는 `python test_gpt_trinity.py -h`를 사용하여 매개변수의 도움말을 표시할 수 있습니다.

다음은 pre-trained GPT-2를 가져와서 문장을 생성하는 예시입니다.

```bash
python test_gpt.py \
    --checkpoint ./path/to/pretrained_model \
    --max_length 384 \
    --device cuda:1
```

### Writing

`python bookathon_writer.py -h`를 사용하여 매개변수의 도움말을 표시할 수 있습니다.

다음은 pre-trained GPT-Trinity를 가져와서 글을 작성하는 예시입니다.

```bash
python bookathon_writer.py \
    --checkpoint ./path/to/pretrained_model \
    --max_length 384 \
    --device cuda:0 \
    --num_last_sentences 4
```
