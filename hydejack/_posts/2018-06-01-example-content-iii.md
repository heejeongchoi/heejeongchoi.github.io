---
layout: post
title: Example Content III


---

이번 포스팅에서는 지도 차원 축소 방법 중 Forward Selection, Backward Elimination, Stepwise Selection, 그리고 Genetic Algorithm에 대해 살펴보겠습니다. 해당 포스팅은 고려대학교 산업경영공학과 강필성 교수님의 강의를 참고했음을 밝힙니다.

# **차원 축소**

차원 축소는 왜 해야 할까요?

저희가 일반적으로 접하는 데이터들은 대부분 고차원의 데이터입니다. 고차원의 데이터는 모델의 성능을 저하시키는 노이즈를 가지고 있을 확률이 높고, 모델을 학습하는데 있어 높은 계산 복잡도를 가집니다. 또한, 모델의 일반화를 위해 많은 개수의 데이터를 필요로 한다는 문제점도 존재합니다. 이러한 고차원 데이터의 문제점을 해결하기 위해 도입된 해결 방안 중 하나가 바로 차원 축소입니다.

차원 축소는 여러 개의 변수가 존재하는 변수 집합에서 모델을 가장 잘 학습시킬 수 있는 변수의 부분집합을 찾거나 새로운 변수의 집합을 추출하는 것입니다. 이러한 차원 축소는 크게 지도 방법과 비지도 방법으로 나누어집니다. 먼저, 지도 차원 축소(supervesed demension reduction)은 모델로부터 평가받은 차원 축소의 결과를 반영해 반복적으로 차원 축소를 진행함으로써 최종적으로 모델에 가장 적합한 변수의 부분집합을 찾는 방법입니다. 아래 그림의 Learning Algorithm에서 Supervised Feature Selection으로 이어지는 화살표가 바로 모델의 피드백을 반영해 다시 차원 축소를 진행하는 것을 나타냅니다. 따라서 지도 차원 축소는 모델의 평가를 기반으로 차원 축소를 진행하므로 동일한 데이터일지라도 모델에 따라 각각 다른 차원 축소 결과를 도출합니다.

![Supervised Feature Selection](http://h-doong.github.io/assets/img/blog/supervised_feature_selection.jpg){:data-width="1920" data-height="1200"}
Supervised Feature Selection
{:.figure}

반면 비지도 차원 축소(unsupervesed demension reduction)는 모델의 피드백 없이 분산, 거리 등과 같이 데이터로부터 추출할 수 있는 성질을 보존하는 저차원의 좌표계를 찾는 과정을 통해 새로운 변수의 집합을 추출하는 방법입니다. 따라서 지도 차원 축소와 다르게 데이터의 성질만을 바탕으로 차원 축소가 진행되므로 동일한 데이터에 대해 동일한 차원 축소 결과를 도출합니다.

![Unsupervised Feature Selection](http://h-doong.github.io/assets/img/blog/unsupervised_feature_selection.jpg){:data-width="1920" data-height="1200"}
Unsupervised Feature Selection
{:.figure}

지금부터는 위에서 소개한 지도 차원 축소 방법 중 Forward Selection, Backward Elimination, Stepwise Selection, 그리고 Genetic Algorithm에 대해 차례대로 살펴보겠습니다. 앞의 4가지 지도 차원 축소 방법은 고차원의 데이터에서 변수를 선택함으로써 데이터의 차원을 축소하는 변수 선택법에 해당함을 먼저 알려드립니다.

- - -

## **Forward Selection (전진 선택법)**
전진 선택법은 이름처럼 변수를 선택해 나가는 방법으로 변수가 없는 모델에서부터 시작해 유의미한 변수를 하나씩 추가하는 과정을 통해 변수를 선택합니다. 예를 들어, 아래 그림처럼 p개의 변수가 존재할 때, 가장 먼저 하나의 변수를 선택하는 과정을 거칩니다. 이 과정에서 선택되는 변수는 하나의 단일 변수로 모델을 학습시키거나 적합했을 때 가장 좋은 결과를 보여야겠죠? 그러므로 p개의 변수 각각에 대해 하나의 변수만을 이용한 p개 모델의 결과를 얻은 후, 그 중 가장 좋은 결과를 도출한 변수가 유의미한 변수라면 그 변수를 최종적으로 선택합니다. 그림에서는 3번째 변수가 선택되었습니다. 다음으로 이전 단계에서 선택된 변수는 고정시키고 해당 변수를 제외한 p-1개의 변수를 하나씩 추가해보며 위와 같은 과정을 거쳐 변수를 선택합니다. 즉, 총 두개의 변수로 학습한 p-1개의 결과 중 가장 좋은 결과를 보인 변수가 유의미하면 그 변수를 추가적으로 선택합니다. 그림에서는 5번째 변수가 두번째로 선택되었습니다. 위와 같은 과정을 반복해 선택되지 않은 변수들 중 가장 좋은 결과를 보이는 유의미한 변수를 추가적으로 선택함으로써 차원을 축소하는 방법이 바로 전진 선택법입니다.

![Foward Selection](http://h-doong.github.io/assets/img/blog/forward_selection.jpg){:data-width="1920" data-height="1200"}
Foward Selection
{:.figure}

이론적으로는 위와 같은 과정을 모든 변수가 선택되는 마지막 단계까지 진행할 수 있으므로 총 p번의 반복수행이 가능합니다. 하지만 변수 선택 과정에서 유의미한 변수를 선택해야 하므로 선택되지 않은 변수들 중 가장 좋은 결과를 보이는 변수가 유의미하지 않다면 해당 단계에서 전진 선택법이 종료됩니다.

전진 선택법은 앞에서 설명한 것처럼 이전 단계까지 선택된 변수는 고정시키고 해당 변수들을 제외한 다른 변수들을 하나씩 추가해보고 비교하는 과정을 통해 변수를 선택하는 방법입니다. 따라서 한 번 선택된 변수는 절대 제거될 수 없다는 한계점을 가집니다.

지금부터는 전진 선택법의 구현을 통해 알고리즘을 더 자세히 알아보겠습니다. 해당 구현에서는 여러 모델 중 회귀 모델을 기반으로 전진선택법을 진행했음을 알려드립니다.

먼저, 전진 선택법의 구현에 앞서 변수 선택과 변수의 유의미성 판단의 기준이 될 통계량에 대해 알아보도록 하겠습니다. 해당 코드는 회귀 모델을 기반으로 구현되었으므로 일반적으로 회귀 분석에서 변수 선택의 성능 지표로 주로 사용되는 Akaike Information Criteria (AIC)와 수정 결정계수, 그리고 변수의 유의미성 판단의 기준으로 사용되는 p-value를 이용하겠습니다.

먼저, 변수 선택의 기준이 되는 AIC는 잔차제곱항 SSE에 변수의 개수 k에 대한 penalty term을 추가한 지표로 아래와 같이 계산되며, AIC 값이 작을수록 좋은 모델이라고 볼 수 있습니다.

$$
AIC=n\cdot \ln { \left( \frac { SSE }{ n }  \right)  } +2k
$$

다음으로 수정 결정계수는 종속변수의 전체 변동 SST 중 회귀식에 의해 설명되는 변동 SSR(=SST-SSE)의 비율인 결정계수에 변수의 개수 k에 대한 penalty term을 추가한 지표로 아래와 같이 계산되며, AIC와 반대로 값이 클수록 좋은 회귀 모델이라고 볼 수 있습니다.

$$
{ R }_{ adj }^{ 2 }=1-\frac { n-1 }{ n-k-1 } \frac { SSE }{ SST }
$$

변수 선택의 기준인 AIC와 수정 결정계수의 계산식을 구현하면 아래와 같습니다.
~~~python
SSE = np.sum((self.y - y_pred) ** 2)
SST = np.sum((self.y - np.mean(self.y)) ** 2)

AIC = self.n * np.log(SSE / self.n) + 2 * len(used_vars)
adj_R_sq = 1 - (self.n - 1) / (self.n - len(used_vars) - 1) * SSE / SST
~~~

위에서 소개한 두 변수 선택 기준에 이어 회귀 모델에서 변수의 유의미성을 판단하는 기준이 되는 p-value에 대해 살펴보겠습니다. p-value란 가설 검정에서 귀무가설을 지지하는 확률을 나타내는 지표입니다. 따라서, p-value가 작을수록 귀무가설(변수의 계수=0)을 지지하는 확률이 낮으므로 변수가 유의미하다고 볼 수 있습니다. 아래 그림에서 p-value는 빨간색 부분에 해당하며, 이를 구현한 코드는 다음과 같습니다.

![p-value](http://h-doong.github.io/assets/img/blog/p_value.jpg){:data-width="1920" data-height="1200"}
p-value
{:.figure}
~~~python
MSE = (sum((self.y - y_pred) ** 2)) / (len(const_X) - len(const_X.columns))

var_b = MSE * (np.linalg.inv(np.dot(const_X.T, const_X)).diagonal())
sd_b = np.sqrt(var_b)
ts_b = params / sd_b

pvalue = [2 * (1 - stats.t.cdf(np.abs(i), (len(const_X) - 1))) for i in ts_b]
~~~

위에서 소개한 세가지 평가 지표를 이용해 전진 선택법을 구현하겠습니다. 전진 선택법은 크게 선택되지 않은 변수들 중 추가되었을 때 가장 좋은 성능을 보이는 변수를 선택하는 과정과 선택된 변수의 유의미성을 판단하고 해당 변수가 유의미하면 최종적으로 선택하는 과정으로 나누어집니다.

먼저, 첫번째 과정은 설명변수 *X*, 종속변수 *y*, 현재까지 선택된 변수의 리스트 *selected_vars*, 그리고 변수 선택의 기준 *eval_metric*을 바탕으로 현재 단계에서 추가되었을 때 가장 모델의 성능이 좋은 변수를 선택하는 과정입니다. candidate_vars에 속하는 각 변수와 selected_vars를 이용해 기준값을 도출한 후 가장 좋은 성능을 보이는 변수를 찾습니다. 아래 구현에서는 기준이 AIC의 경우 낮을수록 좋은 모델이므로 AIC가 최솟값인 변수를, 수정 결정계수의 경우 높을수록 좋은 모델이므로 최댓값인 변수를 선택합니다.
~~~python
candidate_vars = list(set(self.all_vars) - set(selected_vars))

candidate_vars_crt, pvalues = [], []
for i in range(len(candidate_vars)):
    used_vars = selected_vars + [candidate_vars[i]]

    candidate_var_crt, fitted_model = self.metric(used_vars)
    candidate_vars_crt.append(candidate_var_crt[self.eval_metric])

    pvalue = self.p_value(fitted_model, used_vars)
    pvalues.append(pvalue)

if self.eval_metric == 'AIC':
    selected_idx = np.argmin(candidate_vars_crt)
elif self.eval_metric == 'adj_R_sq':
    selected_idx = np.argmax(candidate_vars_crt)

selected_var = candidate_vars[selected_idx]
selected_pvalue = pvalues[selected_idx][-1]
~~~

다음으로 두번째 과정은 현재까지 선택된 변수의 리스트 *selected_vars*, 위의 첫번째 과정을 구현한 함수 *forward_cell*, 변수의 유의미성 판단 지표인 p-value의 유의수준 alpha를 바탕으로 첫번째 과정에서 선택된 변수를 최종적으로 선택할 지 판단합니다. 첫번째 과정에서 1차적으로 선택된 변수의 p-value값인 selected_pvalue를 기준으로 그 값이 유의수준보다 작아 선택된 변수가 유의하면 최종적으로 선택하고, 유의하지 않으면 변수 선택을 종료합니다.
~~~python
selected_var, selected_pvalue = self.forward_cell(selected_vars)

if selected_pvalue <= alpha:
    selected_vars.append(selected_var)
else:
	break
~~~


[mm]: https://guides.github.com/features/mastering-markdown/
[ksyn]: https://kramdown.gettalong.org/syntax.html
[ksyntab]:https://kramdown.gettalong.org/syntax.html#tables
[ksynmath]: https://kramdown.gettalong.org/syntax.html#math-blocks
[katex]: https://khan.github.io/KaTeX/
[rtable]: https://dbushell.com/2016/03/04/css-only-responsive-tables/
