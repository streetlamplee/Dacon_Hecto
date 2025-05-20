## Hecto 차량 분류 모델 git

### TODO


### IDEA
1. 데이터를 보니 차량을 다양한 각도에서 촬영한 데이터<br>low feature 에서 차량의 모습을 얻어내고, 이를 토대로 high feature에 대한 가중치를 더하는 방식으로 진행을 해보고자 함<br>low feature와 high feature를 concat 후, 이를 토대로 cls 분류를 해 나가면 다양한 차량의 촬영모습에 대응할 수 있을 것 같음
2. 사람이 차량을 분류할 때에 집중적으로 보는 부분이 있음 (ex. 바퀴, 후미등, 전조등, 마크 등) 따라서 ai 학습시에도 이 부분을 적용하기 위해 attention을 적용해보면 어떨까?

