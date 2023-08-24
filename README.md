# 매장 판매-시계열 예측
머신러닝을 사용하여 식료품 판매 예측. 이 대회에서는 에콰도르에 위치한 Favorita 매장에서 판매되는 수천 개의 제품군에 대한 매출을 예측하게 됩니다.

# 데이터 세트 설명

이 대회에서는 에콰도르에 위치한 Favorita 매장에서 판매되는 수천 개의 제품군에 대한 매출을 예측하게 됩니다. 학습 데이터에는 날짜, 매장 및 제품 정보, 해당 항목의 판촉 여부, 판매 번호가 포함됩니다. 추가 파일에는 모델 구축에 유용할 수 있는 보충 정보가 포함되어 있습니다.

데이터 세트 출처(https://www.kaggle.com/competitions/store-sales-time-series-forecasting/data)

데이터 목록
* train.csv
* test.csv
* holidays_events.csv
* oil.csv
* sample_submission.csv
* stores.csv
* transactions.csv






| 번호 | 내용                                                                          |
|------|-------------------------------------------------------------------------------|
| 1  | [train.csv - 학습 데이터](#1)                                                    |
| 2  | - store_nbr: 상품 판매 상점 식별 번호.                                           |
| 3  | - family: 판매되는 제품 유형.                                                    |
| 4  | - sales: 특정 상점에서 특정 날짜에 판매된 제품 유형의 총 판매량.                  |
| 5  | - onpromotion: 특정 날짜의 상점에서 프로모션 중인 제품 유형의 총 수.              |
| 6  | [test.csv - 테스트 데이터](#2)                                                   |
| 7  | - 학습 데이터와 동일한 특징을 갖지만 예측 대상 판매량 포함.                       |
| 8  | [sample_submission.csv - 제출 샘플 파일](#3)                                     |
| 9  | [stores.csv - 상점 메타데이터](#4)                                               |
| 10 | - city, state, type 및 cluster를 포함한 메타데이터. 클러스터는 유사한 상점 그룹.  |
| 11 | [oil.csv - 일일 유가](#5)                                                        |
| 12 | - 학습 및 테스트 데이터 기간 동안의 값 포함.                                      |
| 13 | [holidays_events.csv - 휴일 및 이벤트](#6)                                       |
| 14 | - 휴일과 이벤트의 메타데이터 포함.                                                |

<!-- 목차 -->

# 차 례

| 번호 | 내용                                             |
|------|--------------------------------------------------|
| 1  | [데이터 로드](#1)                                  |
| 2  | [데이터 전처리](#2)                         |
| 3  | [데이터 모델링](#3)                         |
| 4  | [데이터 모델링 평가](#4)                            |
| 5  | [데이터 결과 예측](#5)                                |


<!-- intro -->
<div id="1">

# 1.데이터 로드

```python
import pandas as pd
import numpy as np

df_train = pd.read_csv('train.csv')
df_holidays_events = pd.read_csv('holidays_events.csv')
df_oil = pd.read_csv('oil.csv')
df_stores = pd.read_csv('stores.csv')
df_test = pd.read_csv('test.csv')
df_transactions = pd.read_csv('transactions.csv')

```



</div>

<div id="2">

# 2.데이터 전처리
## 2.1 특성 엔지니어링

시계열 데이터 분석시 날짜를 기준으로 분석을 진행하기 때문에 먼저 타임 스탬프 생성해놓으면 편리하다. 

### 2.1.1 타임 스탬프 생성



```python
# df_train의 타임스탬프 추출
df_train['date'] = pd.to_datetime(df_train['date'])
df_train['timestamp'] = (df_train['date'] - pd.Timestamp("1970-01-01")) // pd.Timedelta('1s')

# df_transactions의 타임스탬프 추출
df_transactions['date'] = pd.to_datetime(df_transactions['date'])
df_transactions['timestamp'] = (df_transactions['date'] - pd.Timestamp("1970-01-01")) // pd.Timedelta('1s')

# df_oil의 타임스탬프 추출
df_oil['date'] = pd.to_datetime(df_oil['date'])
df_oil['timestamp'] = (df_oil['date'] - pd.Timestamp("1970-01-01")) // pd.Timedelta('1s')

# df_holidays_events의 타임스탬프 추출
df_holidays_events['date'] = pd.to_datetime(df_holidays_events['date'])
df_holidays_events['timestamp'] = (df_holidays_events['date'] - pd.Timestamp("1970-01-01")) // pd.Timedelta('1s')

```
OutPut
```python
0          1356998400
1194       1356998400
1193       1356998400
1192       1356998400
1191       1356998400
              ...    
2999693    1502755200
2999692    1502755200
2999691    1502755200
2999702    1502755200
3000887    1502755200
Name: 타임스탬프, Length: 3000888, dtype: int64
```

### 2.1.2 데이터 프레임 열 변환, 이상값 제거

위에 열 정리를 해 놓았지만, 한국어로 진행하겠습니다

그리고 df_holidays_events 데이터 설명을 참고.

참고: 전송된 열에 특별한 주의를 기울이십시오. 이전된 휴일은 공식적으로 해당 날짜에 해당하지만 정부에 의해 다른 날짜로 변경되었습니다. 

['변경'] == 'TRUE' 이상값을 제거를 추가.


```ptyhhon
# df_train 열들을 한국어로 변경

df_train.drop(columns=['id'], inplace=True)
df_train.rename(columns={
    'date': '날짜',
    'store_nbr': '매장번호',
    'family': '제품유형',
    'onpromotion': '제품항목수',
    'sales': '판매수',
    'year': '연도',
    'month': '월',
    'day': '일',
    'timestamp': '타임스탬프'
}, inplace=True)

# df_holidays_events 열 한국어

df_holidays_events = df_holidays_events[df_holidays_events['transferred'] != True]
df_holidays_events.rename(columns={
    'date': '날짜',
    'type': '종류',
    'locale': '장소',
    'locale_name': '장소이름',
    'description': '설명',
    'transferred': '변경',
    '인
```python
contains_true = (df_holidays_events['변경'] == 'TRUE').any()

print(contains_true)

```

### 2.1.3 결측값 제거

```python
# df_oil결측값 1218에서 1175로 43개 줄어듬
missing_data = df_oil.isnull().sum()
print(missing_data[missing_data > 0])
df_oil.dropna(inplace=True) 
df_oil

유가    43
dtype: int64
```
before:
[1218 rows x 3 columns]
after
[1175 rows x 3 columns]

## 2.2 시계열 시각화 기본 분석

```python
df_train.sort_values(by='날짜', inplace=True)  # 날짜별 정렬
df_train['판매수'].fillna(method='ffill', inplace=True)  # 판매수 결측값 제
df_sales = df_train.groupby('날짜')['판매수'].sum().reset_index() # 날짜별 판매수 

```

* df_sales 데이터프레임의 '날짜' 열을 인덱스로 설정, 일별로 데이터를 집계해서 각 날짜별 총합을 계산

```python
# 날짜를 인덱스로 설정
df_sales.set_index('날짜', inplace=True)
# 일간 빈도로 리샘플링
df_sales = df_sales.resample('D').sum()
```

OutPut

```python
df_sales


판매수
날짜	
2013-01-01	2511.618999
2013-01-02	496092.417944
2013-01-03	361461.231124
2013-01-04	354459.677093
2013-01-05	477350.121229
...	...
2017-08-11	826373.722022
2017-08-12	792630.535079
2017-08-13	865639.677471
2017-08-14	760922.406081
2017-08-15	762661.935939
1688 rows × 1 columns
```

### 2.2.1 시계열 시각화(기본)

이중 차원 오류를 막기 위해 agg_sales 인덱싱 후 날짜별 판매수 시계열 시각화 진행

```python
import matplotlib.pyplot as plt

# '날짜'별로 '판매수' 합계를 계산
agg_sales = df_train.groupby('날짜')['판매수'].sum()

# 그래프 그리기
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.figure(figsize=(15, 7))
agg_sales.plot()
plt.title('판매수 시계열 데이터')
plt.xlabel('날짜')
plt.ylabel('판매수')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
```
OutPut

![image](https://github.com/plintAn/Store_Sales_Forecast/assets/124107186/a7ce3e95-0c8d-4ce6-bd48-406822e4b18c)

```python
agg_sales

날짜
2013-01-01      2511.618999
2013-01-02    496092.417944
2013-01-03    361461.231124
2013-01-04    354459.677093
2013-01-05    477350.121229
                  ...      
2017-08-11    826373.722022
2017-08-12    792630.535079
2017-08-13    865639.677471
2017-08-14    760922.406081
2017-08-15    762661.935939
Name: 판매수, Length: 1684, dtype: float64
```


## 2.3 '제품유형' 통합

날짜는 2013년부터 2017년까지 자료인데 많은 카테고리로 인해 [3000888 rows × 6 columns] 육박하므로, 날짜별, 매장번호 별 판매수로 묶어보겠습니다. 


```python
df_train_sales_date = df_train.groupby(['날짜', '매장번호']).agg({'판매수': 'sum', '제품유형': 'first'}).reset_index()

print(df_train_sales_date)

[90936 rows x 4 columns]
```

* before:[3000888 rows × 6 columns]
* after:[90936 rows x 4 columns]

## 2.4 날짜별 판매수 시각화(상세)

조금 더 알기쉽게 다음 시각화를 추가합니다.
* 이동 평균 7일
* 이동 평균 추세선

```python
import matplotlib.pyplot as plt
import numpy as np

# '날짜'별로 '판매수' 합계를 계산
agg_sales_category = df_train.groupby('날짜')['판매수'].sum()

# 이동 평균 (예: 7일) 계산
rolling_mean = agg_sales_category.rolling(window=7).mean()

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.figure(figsize=(15, 7))

# 원래의 시계열 데이터 그리기
agg_sales_category.plot(label='판매수', alpha=0.5)

# 이동 평균(추세선) 그리기
rolling_mean.plot(color='red', label='7일 이동 평균')

# 그래프 설정
plt.title('판매수 시계열 데이터와 추세선')
plt.xlabel('날짜')
plt.ylabel('판매수')
plt.xticks(rotation=45)
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.legend()
plt.tight_layout()

plt.show()

```
OutPut

![image](https://github.com/plintAn/Store_Sales_Forecast/assets/124107186/cdc39220-ce99-4c47-92f4-4910cc850146)


### 2.5.1 df_oil 유가 변동 추이 시각화 및 분석

'날짜'별로 '유가'의 변동 추이를 알아보기 위한 시각화를 진행하겠습니다. 일반적인 방법은 선 그래프를 사용하는 것입니다. 이는 시계열 데이터의 연속적인 변동 추이를 직관적으로 파악하기 좋기 때문입니다.

* 이동 평균 7일
* 이동 평균 200일
* 그리드(격자)

```python
# df_oil 유가 변동 추이 시각화 및 분석
import matplotlib.pyplot as plt

# 유가 데이터 추출
oil_prices = df_oil.set_index('날짜')['유가']

# 이동 평균 계산
short_rolling_oil = oil_prices.rolling(window=7).mean()  # 짧은 기간의 이동 평균 (예: 7일)
long_rolling_oil = oil_prices.rolling(window=200).mean() # 장기 이동 평균 (예: 200일)

# 그래프 그리기
plt.figure(figsize=(15, 7))

oil_prices.plot(label='원래 유가 데이터', alpha=0.8)
short_rolling_oil.plot(color='red', label='7일 이동 평균')
long_rolling_oil.plot(color='blue', label='200일 이동 평균')

plt.title('유가 변동 추이')
plt.xlabel('날짜')
plt.ylabel('유가')
plt.legend()
plt.grid(True)

plt.show()
```

OutPut

![image](https://github.com/plintAn/Store_Sales_Forecast/assets/124107186/8642f90b-30ae-415c-a76b-b03e54fc4eb9)


유가는 2015년 이후로 많이 떨어진 상태로 확인됩니다.

### 2.5.2 df_train 날짜별 판매수, df_oil의 날짜별 유가 병합 후 상관관계 분석

추후 머신러닝을 통한 학습시 가중치를 추가하기 위해 상관관계를 분석합니다

```python
# df_train 날짜별 판매수, df_oil의 날짜별 유가 병합 후 상관관계 분석

# '날짜'를 기준으로 두 데이터프레임 병합
merged_df = df_train.groupby('날짜')['판매수'].sum().reset_index().merge(df_oil[['날짜', '유가']], on='날짜', how='inner')

# Pearson 상관 계수 계산
correlation = merged_df['판매수'].corr(merged_df['유가'])

print(f"판매수와 유가의 상관 계수: {correlation:.2f}")
```

OutPut

```python
판매수와 유가의 상관 계수: -0.71
```

-0.71로 강한 음의 상관관계를 보여줍니다. 이는 유가가 오르면 판매수는 줄어들고, 유가가 떨어지면 판매수가 오른다는 것을 의미합니다.



### 2.6.1 데이터 배경 분석

사이트 데이터 설명시(https://www.kaggle.com/competitions/store-sales-time-series-forecasting/data)

* 추가 참고 사항
공공부문 임금은 2주에 한 번씩 15일과 말일에 지급된다. 이로 인해 슈퍼마켓 매출이 영향을 받을 수 있습니다.

다음과 같은 정보를 주었기에 15일, 말일과 평일의 판매수를 분석해보겠습니다. 이와 같은 자료는 유가 정보와 마찬가지로 머신 러닝 모델 학습 시 가중치에 참고 사항입니다.

### 2.6.2 데이터 배경 분석 시각화

박스 플롯을 통해 15일과 말일을 묶고, 평일 판매수와 분포를 비교해보겠습니다.

```python
import matplotlib.pyplot as plt

# 15일과 말일 데이터만 선택
filtered_sales = df_train[(df_train['날짜'].dt.day == 15) | (df_train['날짜'].dt.is_month_end)]

# 날짜별 판매수 합계
agg_sales = filtered_sales.groupby('날짜')['판매수'].sum()

# 시각화
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.figure(figsize=(15, 7))
agg_sales.plot(kind='line', marker='o')
plt.title('15일과 말일의 판매수 시계열 데이터')
plt.xlabel('날짜')
plt.ylabel('판매수')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
```
OutPut
https://www.kaggle.com/competitions/store-sales-time-series-forecasting/data

조금은 차이가 나는 것을 확인할 수 있는데 이를 수치로 환산하여 비교해보겠습니다.

```python
# 15일과 말일, 평일 판매수의 평균을 계산합니다.
avg_special_days_sales = np.mean(special_days_sales)
avg_weekday_sales = np.mean(weekday_sales)

# 두 평균 간의 차이를 퍼센트로 변환합니다.
percentage_difference = ((avg_special_days_sales - avg_weekday_sales) / avg_weekday_sales) * 100

print(f"15일과 말일의 판매수가 평일 판매수보다 약 {percentage_difference:.2f}% 더 높습니다.")
```

OutPut
```python
15일과 말일의 판매수가 평일 판매수보다 약 1.41% 더 높습니다.
```

생각보다 큰 차이를 보이진 않은 걸로 나타났습니다.

## 2.7 휴일(이벤트) 판매수와 평일 판매수 비교분석

df_holidays_events라는 데이터가 주어졌는데, 확인해보면 휴일에 이벤트 한 날이 기록돼 있습니다. 이는 판매수와 연관이 있을 것이라 보이므로 상관관계 분석을 진행해보겠습니다.

```python
# df_sales의 인덱스를 리셋
df_sales_reset = df_sales.reset_index()

# '날짜'를 기준으로 df_holidays_events와 조인하면서 '이벤트' 열 추가
df_sales_merged = pd.merge(df_sales_reset, df_holidays_events[['날짜']], on='날짜', how='left', indicator='이벤트')

# '이벤트' 열이 merge indicator이므로, 'both'면 겹치는 날짜이기 때문에 '0'으로 표시
df_sales_merged['이벤트'] = df_sales_merged['이벤트'].apply(lambda x: '0' if x == 'both' else '1')

# '이벤트' 열이 '0'인 날짜와 그렇지 않은 날짜의 판매수 평균을 계산
avg_sales_event_0 = df_sales_merged[df_sales_merged['이벤트'] == '0']['판매수'].mean()
avg_sales_event_1 = df_sales_merged[df_sales_merged['이벤트'] == '1']['판매수'].mean()

# 시각화
labels = ['이벤트 0', '이벤트 1']
values = [avg_sales_event_0, avg_sales_event_1]

plt.bar(labels, values)
plt.title('판매수 비교: 이벤트 0 vs 이벤트 1')
plt.ylabel('평균 판매수')
plt.show()

# 백분율 차이 계산
percentage_difference = ((avg_sales_event_0 - avg_sales_event_1) / avg_sales_event_1) * 100
print(f"'이벤트' 열이 '0'인 날짜의 판매수가 '이벤트 1' 날짜보다 약 {percentage_difference:.2f}% 차이납니다.")

```
OutPut

![image](https://github.com/plintAn/Store_Sales_Forecast/assets/124107186/3eeb5948-13fe-40bb-bc4c-93dd41c8b1fe)

```python
'이벤트' 열이 '0'인 날짜의 판매수가 '이벤트 1' 날짜보다 약 11.07% 차이납니다.
```

휴일(이벤트)와 평일 판매수의 비교 분석 결과 휴일(이벤트)가 약 11% 높은 판매수를 보이고 있습니다.


### 2.5.2


### 2.2.1 시계열 시각화 기본 분석


</div>

<div id="3">

# 3.데이터 모델링

```python

```

</div>

<div id="4">

# 4.데이터 모델링 평가

</div>

<div id="5">

# 5.데이터 결과 예측

</div>


