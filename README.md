#Mobilebert를 이용한 아마존 책 리뷰의 긍부정 분석
<img src ="https://blog.kakaocdn.net/dn/cd2MQ5/btqx0q65v5Y/mKwQKWKh0HNtslQkgsktE0/img.jpg" width="300" height="100">

# 1. 개요

  아마존닷컴은 미국의 종합 인터넷 플랫폼이다. 
  세계 최대 규모의 온라인 쇼핑몰과 클라우드 컴퓨팅 서비스를 제공하고 있다. 인터넷 서점으로 시작해 현재 미국 온라인 쇼핑몰 매출 1위, 미국 전체 온라인 소매 시장의 약 절반 가량을 차지하고 있다.

  아마존에는 운영기간이 오래 된 만큼 소비자의 긍부정 리뷰를 확인하기 위한 데이터셋 자료가 방대할것으로 생각하고 이것을 분석하고자 아마존을 선택하였다.

### - 1.1 문제정의

    책을 포함한 대부분의 리뷰를 볼 때, 별점이나 다른 요소들이 있지만 그 책에 긍정인지 부정인지 애매모호하게 표현될때가 많고
    소비자들의 만족도를 알기가 어렵기때문에 리뷰 데이터를 분석하여 쉽게 리뷰가 긍정인지 부정인지 한눈에 판단할 수 있는 인공지능 모델을 개발하고자 한다.
    프로젝트는 kaggle에서 제공하는 아마존 책 리뷰데이터를 활용할것이다.

### - 1.2 책 리뷰의 영향력

     1.구매 결정에 도움: 책 리뷰는 독자가 특정 책을 구매할지 여부를 결정하는 데 큰 영향을 미친다. 다른 독자들의 의견을 듣고 해당 책이 자신의 취향과 관심사에 부합하는지를 판단하는 데 도움된다.
     2.독서 경험 향상: 책 리뷰를 통해 독자는 책의 내용, 특징 및 강점에 대한 정보를 얻을 수 있다. 이를 통해 독서 경험을 미리 예측하고 책을 선택하는 데 더 많은 정보를 확보할 수 있다
     3.작가와 출판사 지원: 긍정적인 리뷰는 작가와 출판사에게 중요한 지원을 제공한다. 리뷰가 많이 모이면 책의 인기도와 판매량이 증가하고, 작가는 더 많은 독자들에게 알려지게 된다.
     4.커뮤니티 구축: 책 리뷰는 독자들 간의 커뮤니티를 형성하고 독서 관련 토론을 촉진하는 데 도움이 된다. 비슷한 취향과 관심을 가진 사람들이 함께 책을 읽고 리뷰를 공유함으로써 서로의 독서 경험을 보완하고 풍부한 토론을 이끌어낸다.

# 2. 데이터
 ### - 2.1 데이터 출처     
 Kaggle : https://www.kaggle.com/datasets/mohamedbakhet/amazon-books-reviews

 ### - 2.2 입력-모델-출력

 ```python
  import pandas as pd
  df = pd.read_csv('/content/drive/MyDrive/Books_rating.csv')
  df
 ```
| Id         | Title                | Price | User_id          | profileName                            | review/helpfulness | review/score | review/time | review/summary                                     | review/text                                                                                                                      |
|------------|----------------------|-------|------------------|----------------------------------------|--------------------|--------------|-------------|----------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------|
| 1882931173 | Its Only Art If Its Well Hung! | NaN   | AVCGYZL8FQQTD    | Jim of Oz "jim-of-oz"                 | 7/7                | 4.0          | 940636800   | Nice collection of Julie Strain images            | This is only for Julie Strain fans. It's a col...                                                                                 |
| 0826414346 | Dr. Seuss: American Icon        | NaN   | A30TK6U7DNS82R  | Kevin Killian                          | 10/10              | 5.0          | 1095724800  | Really Enjoyed It                                 | I don't care much for Dr. Seuss but after read...                                                                                 |
| 0826414346 | Dr. Seuss: American Icon        | NaN   | A3UH4UZ4RSVO82 | John Granger                           | 10/11              | 5.0          | 1078790400  | Essential for every personal and Public Library  | If people become the books they read and if "...                                                                                 |
| 0826414346 | Dr. Seuss: American Icon        | NaN   | A2MVUWT453QH61 | Roy E. Perry "amateur philosopher"     | 7/7                | 4.0          | 1090713600  | Phlip Nel gives silly Seuss a serious treatment  | Theodore Seuss Geisel (1904-1991), aka &quot;D...                                                                                 |
| 0826414346 | Dr. Seuss: American Icon        | NaN   | A22X4XUPKF66MR | D. H. Richards "ninthwavestore"       | 3/3                | 4.0          | 1107993600  | Good academic overview                           | Philip Nel - Dr. Seuss: American IconThis is b...                                                                                 |
| ...        | ...                  | ...   | ...              | ...                                    | ...                | ...          | ...         | ...                                                | ...                                                                                                                               |
| B000NSLVCU | The Idea of History            | NaN   | NaN              | NaN                                    | 14/19              | 4.0          | 937612800   | Difficult                                          | This is an extremely difficult book to digest,...                                                                                 |
| B000NSLVCU | The Idea of History            | NaN   | A1SMUB9ASL5L9Y  | jafrank                                | 1/1                | 4.0          | 1331683200  | Quite good and ahead of its time occasionally    | This is pretty interesting. Collingwood seems ...                                                                                 |
| B000NSLVCU | The Idea of History            | NaN   | A2AQMEKZKK5EE4 | L. L. Poulos "Muslim Mom"             | 0/0                | 4.0          | 1180224000  | Easier reads of those not well versed in histo... | This is a good book but very esoteric. "What i...                                                                                 |
| B000NSLVCU | The Idea of History            | NaN   | A18SQGYBKS852K | Julia A. Klein "knitting rat"         | 1/11               | 5.0          | 1163030400  | Yes, it is cheaper than the University Bookstore | My daughter, a freshman at Indiana University,...                                                                                 |
| B000NSLVCU | The Idea of History            | NaN   | NaN              | NaN                                    | 7/49               | 1.0          | 905385600   | Collingwood's ideas sink in a quagmire or verb... | The guy has a few good ideas but, reader, bewa...                                                                                 |

  ### - 데이터 구성
| 데이터       | 구분                                 |
|-------------|-------------------------------------|
| Id          | 책의 고유 식별자                    |
| Title       | 책의 제목                           |
| Price       | 책의 가격 (NaN은 가격이 없음을 의미)|
| User_id     | 사용자의 고유 식별자                |
| profileName | 사용자의 프로필 이름                |
| review/helpfulness | 리뷰의 도움말 수/총 도움말 요청 수 |
| review/score | 리뷰의 평점 (3점 이하는 부정 4점이상은 긍정으로 표기)     |
| review/time | 리뷰가 작성된 시간 (UNIX 시간 형식)|
| review/summary | 리뷰의 요약                        |
| review/text | 리뷰 내용                           |


책 데이터가 3억건이나 되는 거대한 데이터이므로 책의 데이터를 5만건정도로 줄이고,
긍 부정이 골고루 분포되어있는 책 위주로 데이터를 추출하였다.

 ### 3 분석 데이터

딥러닝 학습을 위해 원본 데이터를 가공하여 분석 데이터를 만든다.

가공한 방식:
  
(1) 학습하는 Review의 너무 길거나 짧은 문장들은 딥러닝 학습에 무의미한 데이터들이므로 제거하는 과정을 거침. 문장 길이가 20~855인 Reviews만 추출

(2) Rating 3.5의 리뷰는 중립적인 내용의 리뷰가 대부분. 학습 정확도를 높이기 위해 Rating 3.5의 리뷰는 제외. 

(3) 전처리한 데이터의 Rating이 4이상인 리뷰를 1(긍정), 3이하인 리뷰를 0(부정)으로 바꾸어서 이진 분류함.

### 4 학습 데이터
학습 데이터는 모델이 패턴과 특징을 학습하는 데 사용되는 데이터로, 이를 통해 모델은 입력과 출력 간의 관계를 학습하고 예측을 수행할 수 있게 됨. 학습 정확도를 높이기 위해 앞서 가공한 분석 데이터에서 발생하는 클래스 불균형을 해결
  
- 추출한 방식: 1(긍정), 0(부정)에서 임의로 각각 1,000건씩 추출하여 2,000건을 학습함.

# 5 결과

### 5-1 MOBILEBERT 학습결과

![546546](https://github.com/imsuckatcoding1/midterm/assets/132319927/22c32fb9-7aad-49dd-beab-5c2441c4497b)

![657567](https://github.com/imsuckatcoding1/midterm/assets/132319927/15f6b201-d4b3-47ec-b620-7dbce471ec94)

분석 데이터 전체에 대해 모델을 평가한 결과는 다음과 같다.

검증 데이터에 대한 평가 결과, 모델의 손실과 정확도는 다음과 같다:

Validation Loss: 7.766
Validation Accuracy: 0.490
이를 통해 모델이 분석 데이터 전체에 대해 예측한 결과를 시각적으로 나타낸 그래프는 다음과 같다.

이 결과로부터 우리는 다음과 같은 결론을 도출.

모델의 Validation Accuracy가 0.49, 즉 약 49%
이 결과로부터 현재 사용 중인 예측 모델이 분석 데이터에 대해 높지 않은 성능을 보인다는 것을 알 수 있다

### 결론


현재 훈련된 MobileBERT 모델은 훈련 데이터에 대해서는 점진적으로 향상된 성능을 보였으나, 검증 데이터에 대해서는 낮은 성능을 보였다. 이를 통해 모델이 더 나은 성능을 보이도록 하기 위해 추가적인 데이터 전처리, 하이퍼파라미터 튜닝, 더 많은 데이터로의 훈련 등이 필요함, 데이터의 양이 방대해서 줄이는것도 꽤나 복잡하였고 확실한 방향성을 잡지 못해서 난항을 겪었다. 다음엔 중간중간에 명확한 포인트를 잡고 mobilebert 학습포인트를 잡는게 중요하다고 생각한다.



