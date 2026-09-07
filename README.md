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


## 핸드폰 촬영 가이드

**준비물**: 폰 1대, 콘솔의 [보드 PNG](http://localhost:8080/api/board.png)를 **A3 가로, 100 %** 로 인쇄한 ChArUco 보드(7×5 칸, 385×275 mm), 테이프.

**카메라 설정**: 1080p 또는 4K, 30 fps. **노출·초점 고정**(화면 길게 눌러 AE/AF lock). 손떨림 보정 켜기, HDR 끄기. 조명은 일정하게.

**배경 씬 (2–5 분)**
1. 보드를 작업대 위, 로봇이 설 자리 앞에 평평하게 테이프로 고정한다. 보드가 씬의 원점·스케일·수평이 된다.
2. 처음 20 초는 보드가 화면에 크게 들어오도록 여러 각도에서 찍는다.
3. 그 다음 팔 높이·눈높이·낮은 각도 3 궤도로 씬을 **천천히** 돈다. 한 바퀴 30 초 이상. 빨리 흔들면 블러 프레임이 버려진다.
4. 표면(상판, 벽, 물체 뒷면)을 빠짐없이. 유리·거울·검은 광택면은 천으로 덮는다.
5. 촬영 중 아무것도 움직이지 않는다.

**물체 (각 30–60 초, 단계 3)**: 보드 위에 물체 하나만 놓고 위·옆·낮은 각도 궤도. 뒤집어 한 번 더.

**업로드**: 같은 Wi-Fi 에서 폰 브라우저로 `http://<PC IP>:8080/scan.html` → 이름 입력 → 영상 선택 → 업로드하고 시작. 진행 상황이 같은 페이지에 뜨고, 끝나면 뷰어에서 열린다. 파일이 크면 USB 로 `data/scans/<이름>/upload/video.mp4` 에 복사한 뒤 페이지에서 re-run.

**자주 실패하는 경우**: 보드 미검출(휘었거나 너무 작음) → "NOT metric" 경고와 함께 스케일 없이 진행. 등록 프레임 비율이 낮음 → 너무 빨리 찍었거나 텍스처 없는 벽만 찍힘.

## 상태

아래 표는 각 단계의 게이트가 통과될 때 갱신된다.

| 단계 | 게이트 | 상태 |
|---|---|---|
| 0 툴체인 | nvcc·COLMAP·ffmpeg·node, 2DGS 커널 빌드, 허브 내려받기 | **통과** 2026-09-08 — sudo 없이 micromamba 환경으로; 커널 2종 JIT 빌드·import 확인 |
| 1 예제 갤러리 | 허브 6 환경이 브라우저에서 스플랫+메시로 보인다 | **통과** 2026-09-08 — `scripts/shoot.py` 스크린샷 6장, 콘솔 오류 0. 로봇 탭: 링크 스플랫 19개를 MuJoCo FK 로 움직임(`/robot.html`) |
| 2 배경 재구성 | 폰 영상 1편 → splat.ply + mesh.usdz | **통과(합성 영상)** 2026-09-08 — 합성 스캔(허브 스플랫을 업스트림 렌더러로 궤도 렌더, 24 s) → 프레임 114 → COLMAP 104 등록·재투영 0.55 px → 2DGS 30k iter 16 분·85,813 가우시안 → TSDF 메시 400k 면 → `env/assets/<id>_static/{splat.ply, mesh.usdz, mesh.glb, config.yaml}`. ChArUco 는 합성 왕복 0.5 mm 로 검증, 실제 보드 영상은 아직 없음(그래서 스케일·방향은 COLMAP 임의 좌표계) |
| 3 물체·씬 구성 | 물체 2개 + GUI → 업스트림 검증 통과 폴더 | — |
| 4 폰 업로드·문서 | 폰 브라우저에서 업로드 → 결과 | — |
| 5 MuJoCo 평가 | 업스트림 eval.py 무수정 실행, 2 환경 × 50 롤아웃 | **S5.1–5.3 통과** 2026-09-08 — `python -m polaris_mujoco.run third_party/polaris/scripts/eval.py --environment DROID-FoodBussing --policy.client Fake` 가 450 스텝 1 에피소드를 끝내고 mp4·CSV 를 쓴다(업스트림 파일 수정 0, import 후킹만). 스플랫 배경 + MuJoCo 전경 합성, 텍스처 물체, 손목/외부 카메라. 로봇은 업스트림 기본값대로 스플랫으로 그려지고(USD 에서 유도한 링크 프레임 오프셋, 팔·그리퍼 실루엣 일치), `scripts/oracle_lift.py` 오라클이 잡기·들기로 루브릭 0.33 도달(S5.4). 남은 것: π0.5 서버 50 롤아웃(S5.5), 결과 웹(S5.6) |
