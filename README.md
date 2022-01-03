# Churn-User-Analysis
Bachelor's Thesis in Statistics using WOWAH Dataset :mortar_board:

> 논문 자료는 아래 링크에서 읽기 모드로만 확인 가능함 (Bachelor's thesis can be found in the link below)
> https://drive.google.com/file/d/1fv5NRgfbePqeSdrBlRp2ac4NtlREt-Yu/view?usp=sharing


- 1_Parser : Myles O'Neill의 parser 코드를 일부 수정하여 RAR 포맷의 WoWAH 데이터를 하나의 CSV 포맷으로 변환
- 2_Preprocessing : WoWAH 데이터가 Raw Data이기 때문에 기본적인 전처리 진행
- 3_BaseData : 전처리 완료한 데이터를 연구목적에 맞게 캐릭터별 최고 레벨 달성 여부를 제공하는 데이터셋으로 변환
- 4_FeatureEngineering, 5_FeatureEngineering : base 데이터에 파생 변수 생성
- 6_Visualization : 탐색적 데이터 분석을 위한 다양한 그래프 시각화
- 7_StatisticalTest : 변수 선택을 위한 이탈유저와 잔존유저 간에 설명변수의 평균 차이 통계 검정
- 8_Modeling : 논문에서 사용된 모델링을 담당하는 코드로, 베이스 모델부터 Full 모델가지 모델 적합 결과 출력

