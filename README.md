# Pose_Estimation


기간: 2020.03.01 ~ 2020.06.30 

[역할]
- 3D Pose에 적합한 모델 조사
- Python을 이용한 모델링 코드 구현
***********

[목표]
간단히 요약하면 다음과 같습니다. Pose Estimation은 머신 러닝을 사용하는데, 유독 정확도가 낮은 몇가지 Pose가 있습니다. 
정확도가 낮은 이유들 중, 학습했던 Data에 자주 등장하지 않은 Rare Pose 때문인 경우가 있었습니다. 
학습 데이터 속 Rare Pose를 알아낼 수 있다면 이를 집중 학습시켜 정확도를 향상시킬 수 있을 것이라 생각하였습니다. 
따라서, 이 프로젝트는 많은 양의 Data 속 Rare Pose를 Detection하는 방법에 대한 연구입니다.

***********
[설명]
- Pose Data는 MPI Human Pose 이용 
- 사람의 Pose를 수십 개의 Vector으로 구분.
- 각 부위 별로 3차원 Vector를 모델링함.
- 3차원 벡터를 분포화시킬 수 있는 FB8-Distribution을 사용.



