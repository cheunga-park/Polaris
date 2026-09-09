# π0.5 (pi05_droid_jointpos_polaris) on MuJoCo — run `pi05_v2`

업스트림 코드(`scripts/eval.py`, 루브릭, IC, 정책 클라이언트) 무수정. 시뮬레이터·렌더 전경만 MuJoCo/OpenGL (PLAN §8.5).

| 환경 | n | 성공 | 성공률 (Wilson 90 %) | 평균 진행도 | 업스트림 진행도: 논문 / issue #24 (성공률) | 비교 가능 |
|---|---|---|---|---|---|---|
| DROID-BlockStackKitchen | 30 | 1 | 3 % (1–14) | 0.362 | 0.54 / 0.50 (2 %) | 예 |
| DROID-FoodBussing | 30 | 0 | 0 % (0–8) | 0.450 | 0.58 / 0.58 (17 %) | 예 |
| DROID-TapeIntoContainer | 30 | 1 | 3 % (1–14) | 0.122 | 0.80 / 0.34 (23 %) | 아니오 (업스트림 리셋 상태 파손) |

진행도 = 루브릭 기준 도달 비율(max-ever), 에피소드 450 스텝·15 Hz, IC 는 허브의 `initial_conditions.json` 순환. 성공률의 이항 구간은 Wilson 90 %; 진행도 구간은 부트스트랩으로 따로 계산할 것.
