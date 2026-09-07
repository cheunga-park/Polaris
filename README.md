# Polaris — PolaRiS 식 Video-to-Sim (MuJoCo)

핸드폰 영상 → 2D Gaussian Splatting 스플랫 + 메시 → 브라우저에서 씬 구성 → PolaRiS 호환 환경 폴더 → MuJoCo 에서 π0.5 평가.
논문: [PolaRiS: Scalable Real-to-Sim Evaluations for Generalist Robot Policies](https://arxiv.org/abs/2512.16881).

- 계획·근거·결정: [PLAN.md](PLAN.md)
- 업스트림은 `third_party/` 서브모듈로 핀 박혀 있고 **수정하지 않는다**. MuJoCo 이식은 import 후킹으로 한다(PLAN §8).

## 상태

단계 0(저장소·툴체인) 진행 중. 아래 표는 각 단계의 게이트가 통과될 때 갱신된다.

| 단계 | 게이트 | 상태 |
|---|---|---|
| 0 툴체인 | nvcc·COLMAP·ffmpeg·node, 2DGS 커널 빌드, 허브 내려받기 | 진행 중 |
| 1 예제 갤러리 | 허브 6 환경이 브라우저에서 스플랫+메시로 보인다 | — |
| 2 배경 재구성 | 폰 영상 1편 → splat.ply + mesh.usdz | — |
| 3 물체·씬 구성 | 물체 2개 + GUI → 업스트림 검증 통과 폴더 | — |
| 4 폰 업로드·문서 | 폰 브라우저에서 업로드 → 결과 | — |
| 5 MuJoCo 평가 | 업스트림 eval.py 무수정 실행, 2 환경 × 50 롤아웃 | — |
