# v2
모델 버전: v2

모델 종류: 다층 퍼셉트론 (MLPClassifier)

지점 수: 29

캘리브레이션: 하드아이언( Hard-iron ), 소프트아이언( Soft-iron ) 보정 적용

입력 : 'Mag_X', 'Mag_Y', 'Mag_Z', 'Ori_X', 'Ori_Y', 'Ori_Z', 'Mag_abs' (7개)

은닉층 구조: 512 → 256 → 128

활성화 함수: tanh

최적화 알고리즘: Adam

반복 횟수: 500

학습 데이터셋 : 242,749
