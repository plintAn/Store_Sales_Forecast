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
filtered_df_sales_oil_merged = df_sales_oil_merged[(df_sales_oil_merged['날짜'].dt.day == 15) | (df_sales_oil_merged['날짜'].dt.is_month_end)]

# 그 외의 날짜 데이터 선택
other_df_sales_oil_merged = df_sales_oil_merged[~((df_sales_oil_merged['날짜'].dt.day == 15) | (df_sales_oil_merged['날짜'].dt.is_month_end))]

# 날짜별 판매수 합계
agg_filtered_sales = filtered_df_sales_oil_merged.groupby('날짜')['판매수'].sum()
agg_other_sales = other_df_sales_oil_merged.groupby('날짜')['판매수'].sum()

# 이동평균 계산
rolling_filtered_sales = agg_filtered_sales.rolling(window=15).mean()
rolling_other_sales = agg_other_sales.rolling(window=15).mean()  # 15일 이동평균 적용

# 시각화
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.figure(figsize=(15, 7))

# 15일과 말일의 이동평균만 그래프에 표시
rolling_filtered_sales.plot(kind='line', color='blue', linewidth=2, label='15일 & 마지막일 (이동평균 15일)')

# 그 외의 날짜의 이동평균만 그래프에 표시
rolling_other_sales.plot(kind='line', color='orange', linewidth=2, label='그 외의 날짜 (이동평균 15일)')

plt.title('이동평균 시계열 데이터 비교')
plt.xlabel('날짜')
plt.ylabel('판매수')
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.show()


# 평균 판매수 차이 계산
avg_diff_filtered = agg_filtered_sales.mean()
avg_diff_other = agg_other_sales.mean()
print(f"15일 & 마지막일 평균 판매수: {avg_diff_filtered:.2f}")
print(f"그 외의 날짜 평균 판매수: {avg_diff_other:.2f}")
print(f"평균 판매수 차이: {abs(avg_diff_filtered - avg_diff_other):.2f}")

```
OutPut
![image](https://github.com/plintAn/Store_Sales_Forecast/assets/124107186/a95ebfb9-bd5e-4d62-aae4-ef2851dc2888)


다음은 박스 플롯 분성 시각화

```python
import matplotlib.pyplot as plt

# 데이터 준비
data = [filtered_df_sales_oil_merged['판매수'].values, other_df_sales_oil_merged['판매수'].values]

# 시각화
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.figure(figsize=(10, 6))

# 박스플롯 생성. whis 값을 크게 설정하여 이상치의 범위를 확장하고, boxprops로 박스 색상을 지정.
bp = plt.boxplot(data, vert=False, patch_artist=True, labels=['15일과 말일', '평일'], whis=[5, 95], boxprops=dict(facecolor="skyblue"))

plt.title('15일과 말일 vs 평일 판매수 분포 비교')
plt.xlabel('판매수')
plt.tight_layout()
plt.show()

```
OutPut
![image](https://github.com/plintAn/Store_Sales_Forecast/assets/124107186/33e059b3-f9e2-4c0c-9474-1b7f44653a6b)



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


### 2.8 데이터 병합 및 정렬

1.df_sales & df_oil 결합

```python
# '날짜'를 기준으로 df_sales와 df_oil을 병합 (left join)
merged_df = df_sales.merge(df_oil[['날짜', '유가']], on='날짜', how='left')

# 병합 결과를 df_sales에 반영
df_sales_oil = merged_df
```
결합 후 결측값을 다음날 유가로 대체.
```python
# 결측값 다음값으로 변경
df_sales_oil['유가'] = df_sales_oil['유가'].fillna(method='bfill')
df_sales_oil
```

2.1.df_sales_oil & df_holidays_events 결합

```python
df_sales_oil_holidays = df_sales_oil.merge(df_holidays_events[['날짜', '변경']], on='날짜', how='left')
df_sales_oil_holidays
```

### 'df_sales_oil_holidays' 의 '변경' 열의 값이 Nan값이면 'O', Nan값이 아니면 'X'로 저장하는 코드

```python
import numpy as np

df_sales_oil_holidays['변경'] = np.where(df_sales_oil_holidays['변경'].isnull(), 'O', 'X')
df_sales_oil_holidays
```

결합 후 보니 중복값이 생겨서 중복값 제거

```python
df_sales_oil_holidays = df_sales_oil_holidays.drop_duplicates(subset='날짜', keep='first')
df_sales_oil_holidays
```

타임스탬프 추가

```python
df_sales_oil_holidays['타임스탬프'] = (df_sales_oil_holidays['날짜'] - pd.Timestamp("1970-01-01")) // pd.Timedelta('1s')
df_sales_oil_holidays
```

최종 데이터 
OutPut

```python
날짜	판매수	유가	변경	타임스탬프
0	2013-01-01	2511.618999	93.14	X	1356998400
1	2013-01-02	496092.417944	93.14	O	1357084800
2	2013-01-03	361461.231124	92.97	O	1357171200
3	2013-01-04	354459.677093	93.12	O	1357257600
4	2013-01-05	477350.121229	93.20	X	1357344000
...	...	...	...	...	...
1683	2017-08-11	826373.722022	48.81	X	1502409600
1684	2017-08-12	792630.535079	47.59	O	1502496000
1685	2017-08-13	865639.677471	47.59	O	1502582400
1686	2017-08-14	760922.406081	47.59	O	1502668800
1687	2017-08-15	762661.935939	47.57	X	1502755200
1688 rows × 5 columns
```


### 2.8 스케일링

모델링의 성능을 높이기 위해 표준화(0,1)를 진행하였습니다.

변수는 다음 결과를 참고했습니다.

* '이벤트' 열이 '0'인 날짜의 판매수가 '이벤트 1' 날짜보다 약 11.07% 차이납니다.
* 판매수와 유가의 상관 계수: -0.71

```python
# '변경' 열을 이진수로 변환
df_sales_oil_holidays['변경'] = df_sales_oil_holidays['변경'].map({'O': 1, 'X': 0})

# '유가'와 '판매수' 특성 스케일링
scaler_oil = StandardScaler()
scaled_oil = scaler_oil.fit_transform(df_sales_oil_holidays[['유가']])

scaler_sales = StandardScaler()
scaled_sales = scaler_sales.fit_transform(df_sales_oil_holidays[['판매수']])

# 스케일링된 결과를 새로운 데이터프레임에 저장
df_sales_oil_holidays_scaled = df_sales_oil_holidays.copy()
df_sales_oil_holidays_scaled['유가'] = scaled_oil
df_sales_oil_holidays_scaled['판매수'] = scaled_sales

print(df_sales_oil_holidays_scaled.head)
```
OutPut
```python
	날짜	판매수	유가	변경	타임스탬프
0	2013-01-01	-2.683254	0.984187	0	1356998400
1	2013-01-02	-0.592754	0.984187	1	1357084800
2	2013-01-03	-1.162968	0.977572	1	1357171200
3	2013-01-04	-1.192622	0.983409	1	1357257600
4	2013-01-05	-0.672135	0.986521	0	1357344000
...	...	...	...	...	...
1683	2017-08-11	0.806111	-0.740639	0	1502409600
1684	2017-08-12	0.663196	-0.788108	1	1502496000
1685	2017-08-13	0.972417	-0.788108	1	1502582400
1686	2017-08-14	0.528900	-0.788108	1	1502668800
1687	2017-08-15	0.536268	-0.788886	0	1502755200
1688 rows × 5 columns
```

</div>

<div id="3">

# 3.데이터 모델링

## 3.1 ARIMA 모델링
* ARIMA는 자동회귀통합이동평균 모델로, 시계열 데이터의 트렌드와 계절성을 모델링하려고 합니다.
* ARIMA 모델은 피처를 요구하지 않기 때문에 '판매수'만 사용하여 모델링 합니다.

로드
```python
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
import numpy as np

# 데이터 로드
# df_sales_oil_holidays_scaled & df_test_sales

X = df_sales_oil_holidays_scaled.drop('판매수', axis=1)
y = df_sales_oil_holidays_scaled['판매수']


```
## 3.1.1 최적의 파라미터 그리드 서치
```python
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
from math import sqrt

# 최적의 파라미터를 찾기 위한 그리드 서치
p_values = range(0, 8)
d_values = range(0, 3)
q_values = range(0, 5)

best_rmse = float("inf")
best_order = None

for p in p_values:
    for d in d_values:
        for q in q_values:
            order = (p,d,q)
            try:
                model = ARIMA(train, order=order)
                model_fit = model.fit()
                forecast = model_fit.forecast(steps=len(test))
                rmse = sqrt(mean_squared_error(test, forecast))
                if rmse < best_rmse:
                    best_rmse = rmse
                    best_order = order
            except:
                continue

print('Best ARIMA Order:', best_order)
print('Best RMSE:', best_rmse)

# 최적의 파라미터를 사용하여 다시 학습 및 예측
model = ARIMA(train, order=best_order)
model_fit = model.fit()
forecast = model_fit.forecast(steps=len(test))
forecast_series = pd.Series(forecast, index=test.index)

```
실행

결과
```python
1178    0.955537
1179    0.752744
1180    1.062633
1181    0.925266
1182    0.960328
1183    0.954992
1184    0.873538
1185    0.983819
1186    0.926973
1187    0.950540
1188    0.947603
1189    0.916762
1190    0.956434
1191    0.932764
1192    0.944948
1193    0.943278
Name: predicted_mean, dtype: float64
```
이를 다시 스케일링을 돌린다.

```python
forecast_original_scale = scaler_sales.inverse_transform(np.array(forecast).reshape(-1, 1))

# 예측된 넘파이 배열을 다시 df_test_sales 값에 대
df_test_sales['판매수'] = forecast_original_scale

```
최종 OutPut

```python
	날짜	판매수
0	2017-08-16	761072.557089
1	2017-08-17	721419.167524
2	2017-08-18	782013.860175
3	2017-08-19	755153.655275
4	2017-08-20	762009.529833
5	2017-08-21	760966.156758
6	2017-08-22	745038.916458
7	2017-08-23	766602.851257
8	2017-08-24	755487.334311
9	2017-08-25	760095.607461
10	2017-08-26	759521.287370
11	2017-08-27	753490.635172
12	2017-08-28	761248.132994
13	2017-08-29	756619.654973
14	2017-08-30	759002.040122
15	2017-08-31	758675.508767
```


## 3.2 LSTM 모델

* LSTM은 RNN의 변형으로 시계열 데이터에 잘 작동하는 딥러닝 모델입니다.
* 데이터를 시퀀스로 변환하여 LSTM에 적용해야 합니다.

## 3.2.1 필요한 라이브러리 임포트
```python
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
```
## 3.2.2 데이터 준비 및 전처리
```python
# 데이터 세트를 학습 및 테스트 데이터로 분할
train = df_sales_oil_holidays_scaled['판매수'].values
test = df_test_sales['판매수'].values

# 데이터를 LSTM 입력 형태에 맞게 변형
def create_dataset(dataset, look_back=1):
    X, Y = [], []
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back)]
        X.append(a)
        Y.append(dataset[i + look_back])
    return np.array(X), np.array(Y)

look_back = 1
trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)

# 입력을 [samples, time steps, features]로 재구성
trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
```

## 3.2.3 LSTM 모델을 구축하고 학습을 진행

```python
model = Sequential()
model.add(LSTM(50, input_shape=(1, look_back)))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(trainX, trainY, epochs=100, batch_size=1, verbose=2)
```
loading ...

![forcasting](https://github.com/plintAn/Store_Sales_Forecast/assets/124107186/aca1815a-2040-46e5-baed-97dbcba27baa)


</div>

<div id="4">

# 4.데이터 모델링 평가

4.1 ARIMA 모델 평가

```python
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
from math import sqrt

# 1. 데이터 준비
# df_sales_oil_holidays_scaled에서 '판매수'만 사용
data = df_sales_oil_holidays_scaled['판매수']

# 학습 및 테스트 데이터 분리 (예시로 마지막 16개를 테스트 데이터로 사용)
train, test = data[:-16], data[-16:]

# 2. 모델 설정 및 학습
model = ARIMA(train, order=(5, 2, 3))
model_fit = model.fit()

# 3. 예측
forecast = model_fit.forecast(steps=len(test))

# 예측된 결과의 인덱스를 테스트 데이터의 인덱스로 설정
forecast_series = pd.Series(forecast, index=test.index)

# 4. 평가
rmse = sqrt(mean_squared_error(test, forecast_series))
print('Test RMSE:', rmse)

```

OutPut
```python
Test RMSE: 0.4591690951740325
```

4.2 ARIMA 모델 평가

```python
# 스케일링된 값을 원래의 스케일로 변환
forecast_original = scaler_sales.inverse_transform(np.array(forecast).reshape(-1, 1))
test_original = scaler_sales.inverse_transform(test.values.reshape(-1, 1))

# RMSE 계산
rmse_original_scale = np.sqrt(mean_squared_error(test_original, forecast_original))
print('Original Scale Test RMSE: %.2f' % (rmse_original_scale))

```
OutPut

```
Original Scale Test RMSE: 0.46
```


</div>

<div id="5">

# 5.데이터 결과 테스트 및 예측

## 5.1 ARIMA 모델 예측

```python
forecast_original_scale = scaler_sales.inverse_transform(np.array(forecast).reshape(-1, 1))

# 예측된 넘파이 배열을 다시 df_test_sales 값에 대
df_test_sales['판매수'] = forecast_original_scale

```
최종 OutPut

```python
	날짜	판매수
0	2017-08-16	761072.557089
1	2017-08-17	721419.167524
2	2017-08-18	782013.860175
3	2017-08-19	755153.655275
4	2017-08-20	762009.529833
5	2017-08-21	760966.156758
6	2017-08-22	745038.916458
7	2017-08-23	766602.851257
8	2017-08-24	755487.334311
9	2017-08-25	760095.607461
10	2017-08-26	759521.287370
11	2017-08-27	753490.635172
12	2017-08-28	761248.132994
13	2017-08-29	756619.654973
14	2017-08-30	759002.040122
15	2017-08-31	758675.508767
```


```python
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)

# 스케일링된 값을 원래의 스케일로 변환
trainPredict_original = scaler_sales.inverse_transform(trainPredict)
trainY_original = scaler_sales.inverse_transform([trainY])
testPredict_original = scaler_sales.inverse_transform(testPredict)
testY_original = scaler_sales.inverse_transform([testY])

# RMSE 계산
trainScore = np.sqrt(mean_squared_error(trainY_original[0], trainPredict_original[:,0]))
print('Train RMSE: %.2f' % (trainScore))
testScore = np.sqrt(mean_squared_error(testY_original[0], testPredict_original[:,0]))
print('Test RMSE: %.2f' % (testScore))
```

OutPut
```python
37/37 [==============================] - 1s 1ms/step
1/1 [==============================] - 0s 22ms/step
Train RMSE: 84771.14
Test RMSE: 51838.95
```
### 평가

* ARIMA 모델을 사용하여 예측한 값들은 2017-08-16부터 2017-08-31까지의 기간 동안의 판매수를 나타낸다.
* 예측된 판매수 값은 721,419부터 782,014 사이에서 변동
* ARIMA 모델로 예측한 결과의 RMSE는 스케일링 된 데이터에 대해 약 0.46

* LSTM 모델은 딥 러닝 기반의 시계열 예측 모델로, 장기적인 패턴을 포착에 유리하다.
* 학습 데이터에 대한 RMSE는 약 84,771.14
* 테스트 데이터에 대한 RMSE는 약 51,838.95

## 결론: LSTM 모델의 RMSE가 ARIMA 모델보다 낮다면 LSTM이 더 나은 예측 성능을 보여주는 것으로 판단

</div>


