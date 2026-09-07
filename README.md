# Polaris — PolaRiS 식 Video-to-Sim (MuJoCo)

핸드폰 영상 → 2D Gaussian Splatting 스플랫 + 메시 → 브라우저에서 씬 구성 → PolaRiS 호환 환경 폴더 → MuJoCo 에서 π0.5 평가.
논문: [PolaRiS: Scalable Real-to-Sim Evaluations for Generalist Robot Policies](https://arxiv.org/abs/2512.16881).

- 계획·근거·결정: [PLAN.md](PLAN.md)
- 업스트림은 `third_party/` 서브모듈로 핀 박혀 있고 **수정하지 않는다**. MuJoCo 이식은 import 후킹으로 한다(PLAN §8).

## 실행

```bash
source scripts/env.sh          # 툴체인 PATH (micromamba 환경 polaris-tools + .venv)
uv sync --extra dev            # 파이썬 환경
scripts/build_kernels.sh       # 업스트림 CUDA 커널 2종 (editable + JIT)
scripts/run.sh                 # 콘솔 http://localhost:8080
```

허브 예제: `uvx --from huggingface_hub hf download owhan/PolaRiS-Hub --repo-type=dataset --local-dir data/hub`

## 상태

아래 표는 각 단계의 게이트가 통과될 때 갱신된다.

| 단계 | 게이트 | 상태 |
|---|---|---|
| 0 툴체인 | nvcc·COLMAP·ffmpeg·node, 2DGS 커널 빌드, 허브 내려받기 | **통과** 2026-09-08 — sudo 없이 micromamba 환경으로; 커널 2종 JIT 빌드·import 확인 |
| 1 예제 갤러리 | 허브 6 환경이 브라우저에서 스플랫+메시로 보인다 | **통과** 2026-09-08 — `scripts/shoot.py` 스크린샷 6장, 콘솔 오류 0. 로봇 FK 탭은 5.2 로 |
| 2 배경 재구성 | 폰 영상 1편 → splat.ply + mesh.usdz | — |
| 3 물체·씬 구성 | 물체 2개 + GUI → 업스트림 검증 통과 폴더 | — |
| 4 폰 업로드·문서 | 폰 브라우저에서 업로드 → 결과 | — |
| 5 MuJoCo 평가 | 업스트림 eval.py 무수정 실행, 2 환경 × 50 롤아웃 | — |
