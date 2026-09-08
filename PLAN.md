# Polaris — PolaRiS 식 Video-to-Sim 재구현 계획

작성 2026-09-07. 대상 논문: **PolaRiS: Scalable Real-to-Sim Evaluations for Generalist Robot Policies**
(arXiv 2512.16881, Jain·Zhang·…·Pertsch, 2025-12). 코드 `github.com/arhanjain/polaris`, 자산 HF `owhan/PolaRiS-Hub`,
씬 구성 GUI `github.com/polaris-evals/compose-environments`(MIT).

표기: **확인** = 이 세션에서 파일·API 로 직접 봄 / **추정** = 근거 있는 추론 / **결정 필요** = 사용자 판단.

---

## 0. 요약

| 질문 | 답 |
|---|---|
| "이미 되어 있나?" | **절반.** `~/urdf-to-simulater` 의 `tosim/v2s` 는 **다른 방식**(SAM2 로 물체 찾고 자산 라이브러리에서 골라 조립)이고, PolaRiS 방식(COLMAP → 2D Gaussian Splatting → TSDF 메시 → 물체 TRELLIS → 로봇 스플랫 링크 앵커 → USD 씬)은 **없다**. 그 저장소의 `docs/polaris/PLAN.md`(2026-09-06) 도 "스캔 입구는 2027 로 보류" 라고 적어 두었다 — 확인 |
| 논문 저자 코드에 재구성 파이프라인이 있나? | **없다.** 공개 코드는 Isaac Lab 위의 **평가 런타임**(스플랫 렌더러 + DROID 환경 + 루브릭)뿐이고, `docs/custom_environments.md` 는 "COLMAP 돌리고 2DGS 저장소 따라 하라" 두 줄이다 — 확인. 즉 **비디오→씬 파이프라인은 우리가 조립해야 한다**(부품은 전부 공개: COLMAP·2DGS·SAM2·TRELLIS) |
| 예제는? | HF `PolaRiS-Hub` 6 환경(1.77 GB): 환경마다 `scene.usda` + `assets/<물체>/mesh.usdz` + `assets/<배경>/{splat.ply, mesh.usdz, config.yaml}` + `initial_conditions.json`. 그리고 `nvidia_droid/`: Franka 로봇 스플랫 `splat.ply` + **링크별 ply 18개**(`SEGMENTED/panda_link0…7, Robotiq 손가락`) + USD — 확인. 예제 웹 뷰는 이걸 그대로 쓴다 |
| 이 머신으로 되나? | RTX 4080 16 GB, WSL2 RAM 12 GB(`.wslconfig`), 20 코어 — 2DGS 학습·메시 추출·SAM2 는 충분(**확인**). 없는 것: `nvcc`, `colmap`, `ffmpeg`, `node`. TRELLIS 는 VRAM 16 GB 경계선(**추정**, 폴백 있음) |
| 시뮬 평가(π0.5 실행)까지? | **MuJoCo 로 간다(§8).** 업스트림 평가 런타임은 서브모듈로 핀 박아 한 줄도 고치지 않고, `isaaclab`/`isaacsim`/`omni` 자리에 우리 모듈을 끼워(import 후킹) 그 위에서 돌린다. 스파이크 3개로 핵심 가정을 확인했다(§8.1) |

**최종 산출물**: 새 GitHub 저장소 하나. `폰 영상 업로드 → 자동 재구성 → 브라우저에서 스플랫/메시/씬 확인·물체 배치 → PolaRiS 호환 환경 폴더(scene.usda + splat + mesh + IC) 내보내기`. 예제 6종은 첫 화면 갤러리.

---

## 1. 논문 파이프라인 ↔ 우리 구현 대응표

논문 §3 (확인, arXiv HTML) 과 저자 코드 기준. "동일하게" 의 뜻을 단계마다 고정한다.

| # | 논문이 하는 일 | 논문의 도구 | 우리 구현 | 동일성 |
|---|---|---|---|---|
| 1 | 2–5 분 단안 영상. 배경 / 로봇 / 물체 **따로** 촬영. ChArUco 보드로 스케일·방향 | ZED 카메라(파이프라인은 카메라 무관이라 명시) | **핸드폰** 영상. ffmpeg 로 프레임 추출 + 블러 프레임 제거 | 같음 (카메라만 다름) |
| 2 | 전 프레임 카메라 추정, ChArUco 원점을 기준 좌표로 `model_aligner` | COLMAP | COLMAP(도커 이미지) + OpenCV ChArUco 검출 → PnP 로 얻은 카메라 위치를 `model_aligner --ref_images_path` 에 넣어 Sim(3) 정렬 | 같음 |
| 3 | 2DGS 학습(광도 손실 + 깊이 왜곡·법선 정규화) | 2DGS(hbb1) | 2DGS 원본 저장소 그대로(`train.py`, 기본 30k iter, `lambda_normal 0.05`) | 같음 |
| 4 | 스플랫 래스터 깊이 → TSDF 융합 → marching cubes 메시 | 2DGS `render.py` | 같은 스크립트(`--voxel_size/--depth_trunc`), 출력 `fuse_post.ply` → GLB → **USDZ** (저자 폴더 규약 `mesh.usdz`) | 같음 |
| 5 | 물체: SAM2 로 분할한 다뷰 이미지 → TRELLIS → 스플랫(외관) + 메시(형상), 텍스처 굽기 | SAM2 + TRELLIS | SAM2.1 + TRELLIS-image-large. **폴백**: 물체 단독 2DGS 스캔(논문도 "특정 물체 재구성법에 묶이지 않는다" 명시) | 같음 / 폴백은 명시 |
| 6 | 로봇 스플랫을 링크별로 앵커, FK 로 움직임 | 저자 코드 `splat_renderer.transform_many` | 허브의 `nvidia_droid/SEGMENTED/*.ply` 를 그대로 쓰고, 브라우저에서 URDF FK 로 링크별 스플랫을 움직여 **보여준다** | 같음 (로봇 실물 스캔은 우리에게 로봇이 없어 허브 자산 사용) |
| 7 | 웹 GUI 로 물체 배치·초기조건·명령문 → `scene.usda` + `initial_conditions.json` | compose-environments(Preact+three+USDZ wasm) | 같은 GUI 를 **벤더링**하고 스플랫 레이어(gaussian-splats-3d) 를 추가 — 원본 GUI 는 메시만 보여 준다(확인) | 같음 + 확장 |
| 8 | Isaac Lab 에서 스플랫 배경 + RTX 전경을 **시맨틱 마스크**로 합성, DROID 관절위치 15 Hz, 루브릭 진행도 | 저자 코드 | **단계 5(선택)**: 저자 코드를 그대로 실행(윈도우 Isaac) 또는 MuJoCo 이식 | §6 D3 |

정직한 경계: 논문 부록의 수치(ChArUco 규격, 2DGS 반복 수, TSDF voxel) 는 arXiv HTML 본문에 없다(확인). 2DGS 저장소 기본값을 쓰고 우리가 정한 값은 `configs/` 에 적어 둔다.

---

## 2. 저장소 구조 (제안: `github.com/cheunga-park/Polaris`)

```
Polaris/
├── README.md                 # 한 줄 실행, 폰 촬영 가이드 링크
├── PLAN.md                   # 이 문서
├── docker/
│   ├── recon.Dockerfile      # nvidia/cuda:12.8-devel + COLMAP + 2DGS + SAM2 + TRELLIS (nvcc 여기서만)
│   └── compose.yaml          # recon 워커 + web
├── polaris_v2s/              # 파이썬 패키지 (uv 관리, py3.12)
│   ├── cli.py                #   polaris-v2s {frames|colmap|align|splat|mesh|objects|pack|serve}
│   ├── frames.py             #   ffmpeg 추출 + Laplacian 블러 점수 + 창별 최선 프레임
│   ├── charuco.py            #   보드 PDF 생성, 검출, PnP, model_aligner 입력 작성
│   ├── colmap.py             #   feature/matcher/mapper/undistort 래퍼 (도커 호출)
│   ├── splat.py              #   2DGS train/render 래퍼, 진행률 파싱
│   ├── mesh.py               #   fuse_post.ply → 정리(연결요소·데시메이션) → GLB → USDZ
│   ├── objects.py            #   SAM2 분할 → TRELLIS → mesh.usdz + splat.ply (+폴백)
│   ├── pack.py               #   PolaRiS-Hub 폴더 규약으로 묶기 + 업스트림 validate 재사용
│   ├── hub.py                #   HF 허브 6 환경 내려받기·색인
│   └── jobs.py               #   단계별 잡 큐(재시작 가능, 단계 파일 = 진실)
├── web/
│   ├── server/               #   FastAPI: 업로드(폰에서 직접), 잡 상태, 자산 서빙
│   └── app/                  #   compose-environments 벤더 + 스플랫 뷰어 + 갤러리 + 로봇 FK 탭
├── third_party/              #   2d-gaussian-splatting, compose-environments (핀 박은 커밋, 서브모듈)
├── configs/                  #   phone.yaml(프레임 fps·해상도), splat.yaml(iter·lambda), mesh.yaml(voxel)
├── data/                     #   gitignore: hub/, scans/<id>/{frames,colmap,splat,mesh,objects,env}
└── tests/                    #   합성 씬(블렌더 없이 three.js 렌더 or 허브 자산) 으로 단계별 스모크
```

---

## 3. 단계별 계획

각 단계는 **게이트**(넘어야 다음 단계) 와 소요 추정을 가진다. 총 **4–6 주**(1인, 이 머신), 단계 5 제외.

### 단계 0 — 저장소·툴체인 (2–3 일)
- GitHub 저장소 생성(공개, MIT + 서드파티 라이선스 표), uv 프로젝트, 서브모듈 핀.
- `docker/recon.Dockerfile`: CUDA 12.8 devel + COLMAP(CUDA) + 2DGS 커널 빌드 + SAM2 + TRELLIS. **nvcc 는 컨테이너 안에만** 둔다(WSL 호스트는 그대로).
- node(LTS) 설치 → compose-environments 빌드 확인.
- 게이트: 컨테이너에서 `colmap -h`, `import diff_surfel_rasterization`, 2DGS 공식 샘플 1 씬 100 iter, TRELLIS 1 이미지 → GLB. 실패 시 폴백은 §5.

### 단계 1 — 예제 갤러리 웹 (1 주)
- 허브 6 환경 + `nvidia_droid` 내려받기(1.77 GB), 색인.
- 웹: 갤러리(환경 6개 썸네일) → 뷰어. 뷰어 레이어: ① 배경 `splat.ply`(gaussian-splats-3d, 2DGS ply 는 scale 2 → 3 패딩 변환 필요) ② `scene.usda` 를 파싱해 물체 usdz 를 제 자리에 ③ `initial_conditions.json` 의 포즈 100개 순환 ④ 로봇 탭: 링크별 ply 18개 + Franka URDF FK 슬라이더.
- 게이트: 6 환경 모두 브라우저에서 스플랫+메시 정합이 눈으로 맞는다. 스크린샷을 논문 그림과 나란히 저장.

### 단계 2 — 배경 재구성 파이프라인 (1.5 주)
- `frames`: ffmpeg 2–4 fps 추출, 1440px 장변 리사이즈, 블러 제거, 300–600 장 목표.
- `charuco`: 보드 PDF 생성(A3, 7×5 칸, DICT_5X5_100 — 우리 선택, configs 에 기록), 검출·PnP.
- `colmap`: SIFT(GPU) → sequential+vocab-tree 매칭(영상이라 exhaustive 불필요) → mapper → `model_aligner`(ChArUco 카메라 위치) → undistort.
- `splat`: 2DGS 30k iter(4080 기준 20–40 분 **추정**), `mesh`: TSDF → `fuse_post.ply` → 정리 → GLB/USDZ + `config.yaml`(저자 규약).
- 게이트: 폰으로 찍은 책상 1 씬이 `assets/<scene>/{splat.ply,mesh.usdz,config.yaml}` 로 나오고 단계 1 뷰어에서 보인다. ChArUco 한 칸 길이가 메시에서 ±3 % 안.

### 단계 3 — 물체·씬 구성·내보내기 (1 주)
- `objects`: 물체 영상 → 프레임 → SAM2 마스크(클릭 1회 프롬프트는 웹에서) → TRELLIS → `mesh.usdz + splat.ply`. 스케일은 ChArUco 위에 놓고 찍어 메시 바운딩박스로 맞춘다.
- 웹 GUI(벤더)에 우리 자산 폴더를 바로 여는 버튼, 스플랫 배경 위에서 배치, 명령문·초기조건 저장 → `scene.zip` → 서버가 풀어 `data/scans/<id>/env/` 로.
- `pack`: 업스트림 `upload_env_to_hf.py` 의 validate 로직으로 폴더 검증(dry-run).
- 게이트: 내 씬 + 물체 2개로 만든 폴더가 업스트림 검증을 통과.

### 단계 4 — 폰 업로드 UX·문서 (3–4 일)
- 폰 → 웹 업로드(같은 LAN, `.wslconfig` 가 mirrored 라 `http://<윈도우IP>:<port>` 로 바로 열림 — 확인). 진행률·단계 로그·실패 시 원인(프레임 부족, 보드 미검출, COLMAP 등록 실패율).
- README 의 촬영 가이드(§4) 와 원커맨드 `./run.sh`.

### 단계 5 — 시뮬 평가 (선택, 결정 D3, 2–3 주)
- **경로 A(논문 그대로)**: 윈도우 호스트 Isaac Sim 5.1 + Isaac Lab 2.3 + 저자 코드. 커널 2종을 MSVC+nvcc 로 빌드. π0.5 정책 서버는 WSL(openpi, JAX) 에 두고 포트로 연결. VRAM 16 GB 에 Isaac + 스플랫 렌더 + π0.5 동시는 빠듯(저자는 3090 24 GB) → 정책 서버를 원격 박스로.
- **경로 B(이식)**: MuJoCo + gsplat, `~/urdf-to-simulater` 의 기존 PLAN 노선. "동일" 은 아니고 "규약 이식".

---

## 4. 핸드폰 촬영 가이드 (README 에 들어갈 내용)

**준비물**: 폰 1대, A3(또는 A4 2장) 로 인쇄한 ChArUco 보드(웹의 "보드 PDF 받기" 버튼), 테이프.

**설정**: 1080p 또는 4K, 30 fps. 카메라 앱에서 **노출·초점 고정**(AE/AF lock, 화면 길게 누르기). 손떨림 보정 켜기. HDR 끄기. 조명은 일정하게(창가 역광 피함).

**배경 씬(2–5 분)**
1. 보드를 작업대 위, 로봇이 있을 자리 앞에 평평하게 붙인다(움직이지 않게). 보드는 씬의 원점이 된다.
2. 처음 20 초는 보드가 화면 가운데 크게 나오게 여러 각도에서 찍는다.
3. 그 뒤 팔 높이·눈높이·낮은 각도 3 궤도로 씬을 천천히 돈다. 한 바퀴 30 초 이상. **빨리 흔들면 블러가 생겨 프레임이 버려진다.**
4. 표면(책상 위, 벽, 물체 뒤편)을 빠짐없이. 유리·거울·검은 광택면은 재구성이 안 되니 천으로 덮는다.
5. 촬영 중 아무것도 움직이지 않는다(사람 포함).

**물체(각 30–60 초)**
- 보드 위에 물체 하나만 놓고, 물체를 화면 가득 담아 위·옆·낮은 각도 궤도. 뒤집어 한 번 더 찍으면 바닥면도 나온다.

**업로드**: 폰 브라우저에서 `http://<PC IP>:8000` → "새 스캔" → 영상 선택(배경 1개 + 물체 N개) → 이름 입력 → 제출. 진행률이 뜨고 끝나면 뷰어로 이동. 파일이 크면 같은 Wi-Fi 에서 USB 케이블 또는 `data/scans/<id>/upload/` 에 직접 복사도 된다.

**자주 실패하는 경우**: 보드가 안 잡힘(휘었거나 작음) → 스케일이 안 맞아 "unscaled" 경고. 등록 프레임 60 % 미만 → 너무 빨리 찍었거나 텍스처 없는 벽만 찍힘.

---

## 5. 리스크와 폴백

| 리스크 | 근거 | 폴백 |
|---|---|---|
| WSL RAM 12 GB 에서 COLMAP/2DGS OOM | `.wslconfig memory=12GB`(확인). 2DGS 는 이미지를 GPU 에 올리므로 CPU 는 여유, COLMAP 매칭이 문제 | 프레임 ≤ 400, sequential 매칭, 필요 시 `.wslconfig` 를 20 GB 로(호스트 31.7 GB) |
| TRELLIS 설치·VRAM | flash-attn·spconv 빌드, 16 GB 경계(추정) | 물체 단독 2DGS 스캔 → 메시 (논문이 허용한 대체). 또는 Hunyuan3D-2 |
| 2DGS 래스터라이저 라이선스 | INRIA Gaussian-Splatting 라이선스(비상업 연구) | 상업 필요 시 gsplat(Apache-2.0, `rasterization_2dgs`) 로 교체. 저장소 README 에 표기 |
| 원본 GUI 가 USDZ 만 읽음 | `AssetLoader.ts`(확인): usdz/usd/glb | 우리 파이프라인이 USDZ 를 만들고, 스플랫은 우리가 추가하는 레이어 |
| 2DGS ply → 웹 뷰어 | 2DGS 는 scale 2개(디스크), 웹 뷰어는 3개 기대 | pack 단계에서 `splat_web.ply` 로 변환(세 번째 scale 을 작은 값으로 패딩) |
| Isaac 평가 경로가 이 머신에 안 맞음 | RT 코어 있으나 VRAM 16 GB, WSL 에서 Isaac 불가 | 단계 5 를 선택으로 분리, 정책 서버 원격 |

---

## 6. 결정 필요 (사용자)

- **D1 저장소 이름·공개 여부**: 기본값 `cheunga-park/Polaris` 공개. (업스트림과 이름이 같아 헷갈리면 `polaris-v2s`.)
- **D2 물체 재구성 1차 방법**: TRELLIS(논문 그대로, 설치 리스크) vs 물체 2DGS 스캔(간단, 논문 허용). 기본값: TRELLIS 시도 → 2일 안에 안 되면 스캔.
- **D3 단계 5 시뮬 평가 경로**: **결정됨(2026-09-08) — MuJoCo, 업스트림 코드 무수정 + 후킹.** 상세는 §8.
- **D4 tosim 과의 관계**: 별도 저장소로 시작하고, 나중에 `tosim` 의 Video to Sim 탭이 이 저장소를 부르게 할지는 그때 정한다.

---

## 7. 완료 기준

1. `git clone && ./run.sh` 로 웹이 뜨고, 허브 예제 6 개가 스플랫+메시+초기조건으로 보인다.
2. 폰 영상 1 편(배경) + 2 편(물체) 을 업로드하면 사람 손 없이 `env/` 폴더가 나오고, 업스트림 검증 스크립트를 통과한다.
3. 그 폴더가 뷰어에서 열리고, GUI 로 물체를 옮겨 초기조건 10 개를 저장·재불러오기 할 수 있다.
4. README 만 읽고 다른 사람이 자기 폰으로 같은 결과를 낸다.

---

## 8. D3 — MuJoCo 이식 계획: 업스트림 무수정, 후킹으로

작성 2026-09-08. 질문은 둘이었다: **가능한가**, 그리고 **원본을 안 건드리고 후킹만으로 되는가**. 답은 둘 다 **예**이고, 아래 스파이크가 근거다.

### 8.1 스파이크 결과 (이 세션, 이 머신에서 실행 — 확인)

| # | 가정 | 실행한 것 | 결과 |
|---|---|---|---|
| S-A | 업스트림 `polaris` 패키지를 **한 줄도 안 고치고** `isaaclab*`·`isaacsim*`·`omni*`·CUDA 커널 자리에 스텁 모듈을 끼우면 import 된다 | `sys.meta_path` 파인더 1개(약 60줄) + 진짜 `usd-core`(pxr) 에 `Semantics` 속성만 붙임. `import polaris.environments` | **성공.** gym 레지스트리에서 6 환경의 `Rubric` 객체(기준 7/6/3/3/3/3개, 진짜 클로저)가 그대로 나옴 |
| S-B | 업스트림 설정 상수는 스텁이 **기록**해 두면 런타임에 읽을 수 있다(우리 쪽에 숫자 복제 불필요) | 스텁 생성자가 kwargs 를 속성으로 보관 | `wrist_cam` 1280×720, focal 2.8, aperture 5.376×3.024, 오프셋 pos/rot; PD 400/80; 초기 관절 7개 — 전부 업스트림 파일에서 읽힘 |
| S-C | 업스트림 `SceneCfg.dynamic_setup()`(USD 파싱) 이 Isaac 없이 `usd-core` 로 돈다 | 허브 `food_bussing/scene.usda`(usdz 페이로드 포함) 에 그대로 호출 | **성공.** 강체 7개(bowl, battery1/2, cup, ice_cream_, grapes, 배경) + `external_cam` 포즈 획득 |
| S-D | 업스트림 루브릭 체커 3종이 MuJoCo 모양의 가짜 `env.scene` 위에서 돈다 | `.data.root_pos_w` 등을 torch 텐서로 주는 덕타입 객체 + `omni.usd.get_context().get_stage()` 를 메모리 스테이지(`/World/envs/env_0/scene` 이 scene.usda 참조) 로 후킹 | `reach/lift/is_within_xy` 정답. `get_bbox` 가 usdz 에서 bowl 0.18×0.18×0.094 m 계산(scene.usda 의 scale 0.18 과 일치) |
| S-E | WSL 에서 MuJoCo 오프스크린 렌더가 GPU 로 된다 | `GALLIUM_DRIVER=d3d12 MUJOCO_GL=egl` | **RTX 4080 D3D12.** 1280×720 rgb 4.9 ms, 세그먼테이션 22 ms/프레임(llvmpipe 는 88 ms — 그것도 가능) |
| S-F | VRAM 16 GB 에 정책 서버 + 스플랫 렌더 + MuJoCo 가 같이 들어간다 | π0.5 서버 실측 9.0 GiB(`~/urdf-to-simulater/docs/openpi/README.md:122`) | 남는 ~7 GB. 스플랫 렌더러(배경 110 MB ply) 2–3 GB **추정** — S5.0 에서 잰다. 부족하면 정책 서버를 다른 박스로 |

업스트림 런타임 3,717줄 중 Isaac 을 import 하는 파일은 7개, 그 가운데 **런타임에 Isaac 객체를 실제로 호출하는 표면**은 `manager_based_rl_splat_environment.py` 의 `self.scene[...]`, `self.scene.rigid_objects/sensors`, `GeometryPrim.get_world_poses`, `get_current_stage`, `math.matrix_from_quat` 와 `checkers.py` 의 `get_context().get_stage()` 뿐이다(확인). 나머지(스플랫 렌더러 700줄, 정책 클라이언트, 루브릭, 설정, eval 루프) 는 Isaac 과 무관하다.

### 8.2 구조 — 세 층, 업스트림은 서브모듈로 핀

```
third_party/polaris/            업스트림 그대로 (git submodule, 커밋 핀, 수정 0)
polaris_mujoco/
├── hooks/                      H 층: import 후킹 — "Isaac Lab 모양의 창구"
│   ├── finder.py               sys.meta_path 파인더: isaaclab*, isaaclab_tasks*, isaacsim*, omni* 를 우리 모듈로
│   ├── recording.py            설정 클래스 스텁(kwargs 를 속성으로 보관; @configclass = 항등)
│   ├── facade_env.py           isaaclab.envs.ManagerBasedRLEnv  ← 우리 MuJoCo 기반 env (업스트림 클래스의 부모가 된다)
│   ├── facade_scene.py         scene[name].data.root_pos_w / root_state_w / joint_pos / target_pos_w / write_root_pose_to_sim
│   ├── facade_camera.py        isaaclab.sensors.camera.camera.Camera  ← isinstance 통과 + data.output["rgb"|"semantic_segmentation"]
│   ├── facade_prims.py         isaacsim.core.prims.GeometryPrim(prim_paths_expr) → 링크 xpos/xquat
│   ├── facade_stage.py         get_current_stage / omni.usd.get_context → 메모리 pxr 스테이지 (S-D)
│   ├── facade_math.py          isaaclab.utils.math.matrix_from_quat 등 3–4 함수
│   ├── app.py                  isaaclab.app.AppLauncher = no-op, isaaclab_tasks.utils.parse_env_cfg = 우리 것
│   └── patches.py              (최후 수단) 메서드 단위 몽키패치 — 패치마다 업스트림 소스 해시 테스트 1개
├── backend/                    M 층: MuJoCo 본체
│   ├── stage.py                scene.usda → MJCF: usdz→OBJ(pxr+trimesh), CoACD 볼록 분해, freejoint, 초기 포즈
│   ├── robot.py                menagerie panda_nohand + robotiq_2f85 조립, 위치 액추에이터 kp400/kv80, gravcomp=1, 자기충돌 off
│   ├── control.py              dt 1/120, decimation 8 → 15 Hz, 450 스텝, 그리퍼 이진(0 / π/4 ↔ driver joint)
│   ├── cameras.py              업스트림 내참(focal/aperture → fovx) + opengl 오프셋 → MuJoCo 카메라, D3D12 EGL
│   └── segmentation.py         geom→body→{robot, raytraced 물체} 마스크 (업스트림 `mask >= 2` 규약)
└── run.py                      E 층: 후킹 설치 후 업스트림 scripts/eval.py 를 runpy 로 그대로 실행
```

**핵심 아이디어**: 업스트림 `ManagerBasedRLSplatEnv(ManagerBasedRLEnv)` 의 부모 `ManagerBasedRLEnv` 를 우리 `facade_env` 가 대신한다. 업스트림 클래스의 메서드(`reset`, `step`, `custom_render`, `transform_sim_to_splat`, `render_splat`, `setup_splat_robot`, `_evaluate_rubric`) 는 **그대로** 우리 객체 위에서 실행되고, 그것들이 읽는 속성만 우리가 공급한다. 스플랫 렌더러(`SplatRenderer`, CUDA 커널 2종) 도 업스트림 것을 그대로 빌드해 쓴다.

실행 명령은 업스트림과 같다:
```
python -m polaris_mujoco.run third_party/polaris/scripts/eval.py --environment DROID-FoodBussing --policy.port 8000 --run-folder runs/mj
```

### 8.3 창구 계약 (업스트림이 읽는 속성 전부 — 테스트가 이 표를 강제)

| 업스트림이 부르는 것 | 어디서 | 우리가 주는 것 |
|---|---|---|
| `ManagerBasedRLEnv.__init__(cfg)`, `.reset()`, `.step(a)`, `.close()`, `.max_episode_length`, `.device`, `.sim.render()`, `.scene.update(0)`, `.observation_manager.compute()` | splat env | `facade_env`: MJCF 적재, 스텝, 관측 `{"policy": {"arm_joint_pos", "gripper_pos"}}` |
| `scene[name].data.root_state_w/root_pos_w/root_quat_w/default_root_state`, `.write_root_pose_to_sim(pose)` | splat env, checkers | `facade_scene`: 강체 body 의 xpos/xquat, qpos 쓰기 |
| `scene["robot"].data.joint_pos/joint_names` | droid_cfg obs, checkers | 관절 qpos, 이름 목록(`finger_joint` 포함) |
| `scene["ee_frame"].data.target_pos_w` | checkers.reach | Robotiq base_link site 위치 |
| `scene.rigid_objects`, `scene.sensors`, `isinstance(sensor, Camera)` | splat env | dict 2개, `facade_camera` 가 스텁 `Camera` 를 상속 |
| `cam.data.output["rgb"|"semantic_segmentation"]`, `.image_shape`, `.data.pos_w/quat_w_world`, `._sensor_prims[0].Get{Horizontal,Vertical}ApertureAttr().Get()/GetFocalLengthAttr()` | splat env | 렌더 결과, 내참을 업스트림 cfg 값으로 되돌려 줌 |
| `cfg.scene.robot.spawn.usd_path` → `parent/SEGMENTED/*.ply` | setup_splat_robot | 허브 `nvidia_droid/` 경로 |
| `GeometryPrim(prim_paths_expr="/World/envs/env_0/robot/<link>/...").get_world_poses(usd=False)` | transform_sim_to_splat | ply 이름 → MJCF body 매핑표 18행 |
| `get_current_stage()`, `Semantics.SemanticsAPI.Apply` | setup_splat_world | no-op (마스크는 세그먼테이션으로) |
| `omni.usd.get_context().get_stage()` + `pxr.Usd/UsdGeom/Gf` | checkers.is_within_xy | S-D 의 메모리 스테이지 |
| `isaaclab.utils.math.matrix_from_quat` | splat env | 순수 torch 구현 |
| `isaaclab.app.AppLauncher`, `isaaclab_tasks.utils.parse_env_cfg` | scripts/eval.py | no-op / 우리 것 |

`tests/test_facade_contract.py` 가 업스트림 소스를 정규식으로 훑어 `self.scene[...]`·`.data.<attr>` 사용을 모으고, 우리 창구가 그 속성을 전부 제공하는지 확인한다. 업스트림 핀을 올릴 때 이 테스트가 먼저 깨진다.

### 8.4 후킹의 세 단계 — 어디까지 가면 "원본을 건드린 것"인가

| 단계 | 방법 | 원본 수정 | 쓰는 곳 |
|---|---|---|---|
| H1 import 후킹 | 모듈 이름을 가로채 우리 것을 준다 | 0 | 기본. §8.2 전부 |
| H2 언바운드 호출 | 부모 교체가 너무 얽히면 우리 env 가 업스트림 메서드를 `ManagerBasedRLSplatEnv.render_splat(self)` 처럼 빌려 쓴다 | 0 | 폴백 1 |
| H3 메서드 몽키패치 | `patches.py` 에서 특정 메서드만 런타임 교체. 패치 1개 = 업스트림 소스 해시 테스트 1개 + 사유 주석 | 0 (런타임) | 최후. 예상 후보: `setup_splat_world_and_robot_views` 의 `_sensor_prims` 접근이 창구로 못 덮일 때 |

원본 파일 편집(H4)은 하지 않는다. 필요해지면 업스트림에 PR 을 내고 그 커밋으로 핀을 올린다.

### 8.5 논문·업스트림과의 차이 (정직하게 적는 표 — 결과 보고서에 그대로 들어간다)

| 항목 | 업스트림(Isaac Lab) | 우리(MuJoCo) | 영향 |
|---|---|---|---|
| 물리 | PhysX, dt 1/120, 위치반복 64 | MuJoCo, dt 1/120(또는 1/500 + 서브스텝), Newton 솔버 | 접촉 거동 차이. 잡기 성공률에 영향 가능 |
| 로봇 충돌 메시·관성 | NVIDIA DROID USD | menagerie panda + robotiq_2f85(BSD-3) | 자기충돌 off 라 영향 작음 |
| 그리퍼 | `finger_joint` 0 / π/4 이진 | 2f85 driver joint 0 / 0.8 rad, tendon 커플링 | 열림 판정 `open_finger_threshold=0.1` 은 driver joint 로 매핑 |
| 전경 셰이딩 | RTX 돔 라이트 1000 | OpenGL Phong, 그림자 없음 | 정책 입력 픽셀 차이. 논문 ablation 상 렌더 차이는 Δr≈0.15 |
| 배경·합성 | 스플랫 + 시맨틱 마스크 | **동일 코드** | 없음 |
| 루브릭·IC·행동 규약·15 Hz·450 스텝 | — | **동일 코드/값** | 없음 |
| 물체 충돌 | PhysX convexDecomposition | CoACD(threshold 0.05, ≤24 hull) | 유사 |
| 정적 배경 충돌 | PhysX 삼각형 메시(`meshSimplification`) | 작업공간 크롭 후 6 cm 격자 볼록 조각(약 2,000개) — CoACD 는 상판이 2.5 cm 꺼졌음 | 평면은 정확, 곡면은 셀 안에서 근사 |
| 구름 마찰 | PhysX 기본 0 (접촉 면 다각형이 감쇠) | `condim=6`, torsional 0.005 / rolling 0.001 | 원통(건전지)이 낙하 뒤 굴러가지 않게 함 |
| 로봇 외형 | 스플랫(`robot_splat=True` 기본) | **동일** — 링크 스플랫을 USD 유도 오프셋으로 앵커 | 없음 |
| 전경 조명 | RTX 돔 라이트 | 헤드라이트 ambient 0.6 / diffuse 0.5, 그림자 없음 | 텍스처 물체의 음영 차이 |
| 마스크 | 시맨틱 세그먼테이션(`>=2`) | geom→body 분류(로봇=스플랫 1, 스플랫 없는 물체=2) | 규약 동일 |

### 8.6 단계와 게이트 (2.5–3.5 주, 단계 1–4 이후 또는 병렬)

| 단계 | 하는 일 | 게이트 | 기간 |
|---|---|---|---|
| S5.0 툴체인 | 업스트림 커널 2종(`diff-surfel-rasterization`, `simple-knn`) 무수정 빌드. nvcc 는 호스트 `cuda-toolkit-12-8`(apt) — GPU GL 이 호스트에서 되므로 단일 프로세스가 가장 단순. 도커는 재현용 2안 | 허브 `g60` 배경 스플랫 1280×720 렌더 < 50 ms, VRAM 측정치 기록(S-F 확정) | 2일 |
| S5.1 후킹 | 스파이크를 패키지로. 창구 계약 테스트. `eval.py` 를 물리 없는 널 백엔드 + `FakeClient` 로 끝까지 실행 | CSV 1행이 나온다(호출 그래프 증명) | 3일 |
| S5.2 씬 | scene.usda → MJCF, 로봇 조립, IC 적용. 허브 6 환경 전부 적재 | 6/6 적재, 정지 상태 루브릭 progress 0, 물체가 IC 포즈에서 1 초 안에 안정(드리프트 < 5 mm) | 4일 |
| S5.3 카메라·합성 | 창구 카메라 rgb+seg, 업스트림 `custom_render` 로 합성 | 업스트림 `docs/images/*.png` 와 나란히 놓고 시점·스케일 일치(픽셀 판독) | 3일 |
| S5.4 제어 | 관절위치 목표, 15 Hz, 그리퍼 이진. 스크립트 오라클(접근→잡기→들기) | 오라클이 `lift` 기준을 넘겨 progress 가 오른다; 같은 IC → 같은 궤적 해시(결정론) | 3일 |
| S5.5 정책 | openpi π0.5 polaris 체크포인트 서버(tosim `scripts/openpi/serve.sh` 재사용, `pi05_droid_jointpos_polaris` 설정은 openpi 업스트림에 있음), 2 환경 × 50 롤아웃 | 업스트림 공개 수치(issue #24 3행) 와 Wilson 90 % 구간 비교, §8.5 표와 함께 보고 | 4일 |
| S5.6 웹 | 환경별 실행 결과(CSV·mp4·progress 히스토그램) 를 갤러리에 | 결과 페이지 1장 | 2일 |

### 8.7 남은 결정

- **D3a nvcc 위치**: 호스트 apt(기본) vs 도커 devel. 기본값 호스트 — S5.0 에서 둘 다 시도하고 되는 쪽.
- **D3b 로봇 모델**: menagerie(기본) vs 허브 USD 에서 메시 추출. 기본값 menagerie.
- **D3c 윈도우 Isaac 경로(A)**: 유지하지 않는다. 비교가 필요해지면 그때 별도.

### 8.8 물체 생성(TRELLIS)과 씬 구성 GUI 는 MuJoCo 와 무관하다 — 그대로 쓴다

논문 §3 "Object Creation with Generative Models" 와 그림 3 의 GUI 는 **시뮬레이터 앞단**이다. 시뮬레이터가 받는 것은 파일뿐이다.

| 산출물 | 만드는 쪽 | Isaac 이 쓰는 법 | MuJoCo 가 쓰는 법 (우리) |
|---|---|---|---|
| 물체 메시(텍스처 구운 GLB → `mesh.usdz`) | SAM2 → TRELLIS → 베이크 | 시각: RTX 렌더 / 충돌: PhysX convexDecomposition | 시각: OpenGL 텍스처 메시 / 충돌: CoACD 볼록 분해(`backend/stage.py`). TRELLIS 메시는 watertight 라 분해가 깨끗하다 |
| 물체 스플랫(`splat.ply`, 선택) | TRELLIS 의 GS 디코더 | `assets/<name>/splat.ply` 가 있으면 업스트림 렌더러가 강체 포즈를 따라 옮겨 그린다 | **같은 업스트림 코드**(`transform_sim_to_splat`). 허브 6 환경은 물체 스플랫 없이 메시만 배포한다(확인) — 우리 스캔은 둘 다 만들 수 있다 |
| `scene.usda` (물체별 translate/orient/**scale**, kinematic 플래그) | GUI(브라우저) | USD 스테이지 적재 | `usd-core` 로 파싱(S-C 확인) → MJCF: `<mesh scale>`, freejoint / kinematic 이면 고정 body, 질량은 밀도 1000 kg/m³ 기본(PhysX 기본과 동일 — S5.2 에서 대조) |
| `initial_conditions.json` (명령문 + 포즈 100개) | GUI | `reset(object_positions=…)` | **같은 업스트림 코드** — 창구의 `write_root_pose_to_sim` 이 qpos 를 쓴다 |
| DROID 로봇 자동 배치(원점) | GUI 가 참고용으로 띄움(내보내기 제외) | `nvidia_droid` USD | menagerie panda+2f85 를 같은 원점 규약으로 |

즉 TRELLIS 와 GUI 는 §3 단계 3 그대로이고, MuJoCo 이식은 "그 파일들을 어떻게 읽느냐"(§8.2 `backend/stage.py`) 에만 닿는다. TRELLIS 설치 리스크(D2) 는 그대로 남으며 시뮬레이터 선택과 무관하다.

---

## 9. 용량 점검 (2026-09-08 실측)

### 9.1 디스크

| 위치 | 전체 | 사용 | 여유 | 비고 |
|---|---|---|---|---|
| WSL 루트(`/`) | 1007 GB | 158 GB | **799 GB** | 가상 디스크라 실제 상한은 아래 C: 여유 |
| Windows C: | 931 GB | 811 GB | **119 GB (88 %)** | WSL 의 `ext4.vhdx` 가 여기 있음, 현재 **180 GB**. WSL 에 쓰는 만큼 C: 가 줄어든다 |

필요량(추정, 근거는 각 행):

| 항목 | 크기 | 근거 |
|---|---|---|
| PolaRiS-Hub 6 환경 + 로봇 | 1.8 GB | HF tree API 합산 |
| TRELLIS-image-large + DINOv2-large | 5.7 GB | HF API 3.3 + 2.4 |
| SAM2.1 | 0.3 GB(캐시에 이미 있음) / large 1.8 GB | HF |
| CUDA toolkit 12.8(apt) | ~3.5 GB | 배포판 일반값 |
| COLMAP 도커 이미지 | ~3 GB | 일반값 |
| 파이썬 venv(torch cu128 등) | ~7 GB | tosim `.venv` 7.2 GB 와 동급, torch 는 uv 캐시에 있어 실제 추가는 더 작음 |
| TRELLIS 빌드 의존(flash-attn, spconv, kaolin, nvdiffrast) | ~3 GB | 추정 |
| π0.5 polaris 체크포인트 | **12.4 GB** | GCS 목록 27 파일 합산 |
| 스캔 1건(프레임·COLMAP·2DGS·메시·물체) | 3–5 GB/건 | 추정 |
| **고정 합계** | **≈ 40 GB** + 스캔당 4 GB | |

판정: **된다.** 다만 C: 여유 119 GB 에서 vhdx 가 40–60 GB 더 자라므로 60–80 GB 남는다. 여유를 만들려면(권장, 사용자 결정):
- `uv cache prune` — uv 캐시 44 GB
- `docker system prune` — 회수 가능 이미지 8 GB + 빌드 캐시 9.5 GB
- HF 캐시 20 GB 중 안 쓰는 데이터셋(lerobot droid 등) 정리
- vhdx 압축(`wsl --shutdown` 후 `Optimize-VHD`/diskpart compact) — WSL 안에서 지운 공간은 이걸 해야 C: 로 돌아온다
- **`C:\u2s` 47 GB**(윈도우 Isaac Sim 가상환경 4벌) — D3 가 MuJoCo 로 정해져 더 쓰지 않는다면 C: 에서 바로 회수되는 가장 큰 덩어리. 단 tosim 의 Isaac 워커가 여기를 쓰므로 그쪽을 완전히 접을 때만

### 9.2 VRAM (RTX 4080 16 GB)

**지금 6.1 GB 를 Windows 쪽이 쓰고 있다**(Xwayland·데스크톱·Edge WebView, GPU 25 %). WSL 이 실제로 쓸 수 있는 건 현재 **~10 GB**.

| 작업 | 필요 | 판정 |
|---|---|---|
| 2DGS 학습(이미지 장변 1000–1440 px, 30k iter) | 6–10 GB | 됨. 크면 해상도 1000 px 로 |
| TRELLIS-image-large | 공식 권장 16 GB, fp16 축소판 ~8 GB | **경계.** 실행 중 Windows GPU 앱을 닫아 기저를 1–2 GB 로 내리거나, fp16 경로(TRELLIS-BOX 류)를 쓰거나, D2 폴백(물체 2DGS 스캔) |
| 평가: π0.5 서버 9.0 GB(실측) + 스플랫 렌더 2–3 GB(추정) + MuJoCo GL < 1 GB | ≈ 12–13 GB | **기저 ≤ 3 GB 일 때만 한 장에 들어간다.** 안 되면 정책 서버를 다른 박스로(업스트림도 서버·클라이언트 분리 구조) |

### 9.3 RAM

WSL 상한 12 GB(`.wslconfig`, Isaac 윈도우 워커 여유 때문에 낮춘 값) + 스왑 32 GB, 호스트 31.7 GB. COLMAP(순차 매칭, ≤ 600 프레임)·2DGS·TRELLIS 모두 12 GB 안에서 돈다(추정, TRELLIS 피크 ~8 GB). **D3 를 MuJoCo 로 정했으니 윈도우 Isaac 여유가 더 필요 없다 → `memory=20GB` 로 올리는 것을 권장**(사용자 결정, `.wslconfig` 편집 + `wsl --shutdown`).

---

## 10. 진행 기록 (2026-09-08, 계획 → 실제)

| 단계 | 상태 | 확인된 사실 |
|---|---|---|
| 0 툴체인 | 통과 | sudo 없음 → micromamba `polaris-tools`(nvcc 12.8, COLMAP 4.2 CUDA, ffmpeg 9, node 26). 업스트림 커널 2종은 editable + JIT. 2DGS 학습은 **별도 venv**(`.venv-2dgs`, 원본 hbb1 커널 — polaris 포크는 `near_n/far_n` 인자가 달라 한 환경에 공존 불가). gcc 13 대응은 `NVCC_APPEND_FLAGS="--pre-include cstdint --pre-include cfloat"` 로, 체크아웃 무수정 |
| 1 갤러리 | 통과 | 6 환경 스크린샷, 텍스처 물체(usdz 내 텍스처 + MDL `inputs:texture` 경로 둘 다), 로봇 FK 탭. 렌더 검증은 Windows Edge 헤드리스를 CDP 로 붙여서(WSL Chromium 은 시스템 라이브러리 없음) |
| 2 재구성 | 통과(합성) | 합성 스캔으로 전 단계 통과. COLMAP 4.x 는 옵션명이 바뀌었고(`FeatureExtraction.use_gpu`) 어휘 트리를 스스로 받는다. ChArUco: OpenCV 5 는 `estimatePoseCharucoBoard` 가 없어 `matchImagePoints`+`solvePnP`; OpenCV 보드 프레임은 y-아래·z-안쪽이라 z-up 으로 뒤집음(합성 왕복 0.5 mm). **실제 폰 영상은 아직 없음** |
| 3 물체·GUI | 진행 | compose GUI 는 무수정 빌드해 `/compose-environments/` 에 마운트. SAM 2 마스크 동작(합성 그릇 5/6). TRELLIS 환경 설치 중 |
| 4 업로드 | 완료 | 폰 → `/scan.html` 업로드 → 잡 → 뷰어. 물체 업로드 폼 포함 |
| 5 MuJoCo | S5.1–5.4 통과 | 업스트림 `eval.py` 무수정 실행. 창구 계약 테스트. 발견한 규약: (a) 업스트림 기본 `robot_splat=True` → 로봇은 스플랫으로 그려지고 시뮬레이터 마스크에 없음 (b) 링크 프레임: panda 링크는 menagerie 와 동일, Robotiq 링크는 USD(rest pose) 와 menagerie 를 수치로 대조해 상수 오프셋 유도(`data/cache/robot_link_offsets.json`) (c) 정적 배경 충돌은 CoACD 대신 **작업공간 크롭 + 6 cm 격자 볼록 조각**(CoACD 는 상판이 2.5 cm 꺼짐) (d) 허브 IC 는 물체를 상판 5 cm 위에 두어 리셋 직후 낙하 — Isaac 도 같음 (e) 물체에 구름 마찰 추가(MuJoCo 기본 0 이면 원통이 계속 구름) — §8.5 표에 추가할 델타 |

**S5.5 진행(2026-09-08 02:30~)**: π0.5 polaris 체크포인트(12.4 GB, gcsfs 익명 다운로드 4 분) 를 tosim 의 openpi 환경으로 서빙(:8100, VRAM 9.3 GB). 스모크 2 에피소드: 진행도 0.33, 0.17(업스트림 issue #24 의 Isaac 재현은 평균 0.58). 본 배치 `scripts/eval_batch.sh pi05 8100 50 DROID-FoodBussing DROID-TapeIntoContainer` 실행 중, 이어서 PanClean·BlockStackKitchen 예약. **공정한 비교 대상은 UW 3 환경**(FoodBussing, PanClean, BlockStackKitchen) — 저자가 Princeton 3 환경의 리셋 상태가 공개판에서 깨졌다고 밝힘(issue #24).

**보드 없는 스캔의 임시 정렬**: 카메라 상향 벡터 → 중력, 희소점 하위 5 % → 바닥 z=0, 카메라 높이 0.6 m 가정 → 스케일(합성 스캔에서 s=0.086, 실제 스케일과 일치). 경고 "NOT metric" 은 그대로 남는다. 실제 폰 영상에서는 ChArUco 가 이 자리를 대신한다.

**배치 중 관찰(2026-09-08 04시)**: FoodBussing 50 완료(진행도 0.327, 성공 0/50; 업스트림 Isaac 재현 0.58, 17 %). PanClean·BlockStack·Tape 는 정책 서버 웹소켓 keepalive(20 s) 타임아웃으로 중간에 끊겨 재개 큐에 넣음 — 원인은 WSL RAM 12 GB 상한에서 스왑(5 GB)이 도는 메모리 압박(같은 기계에서 tosim 테스트도 돌고 있었음). `polaris_mujoco.run` 이 websockets 의 ping_timeout 을 600 s 로 넓히는 후킹을 추가. **`.wslconfig memory=20GB` 권장은 그대로**(§9.3).

**S5.5 1차 결과(2026-09-08 05시, `docs/REPORT_pi05.md`)**: UW 3 환경 × 50 롤아웃. 진행도 BlockStack 0.249 / FoodBussing 0.327 / PanClean 0.340, 성공 0/150 (Wilson 90 % 상한 5 %). 업스트림 Isaac 재현(issue #24, 100 에피소드)은 0.501 / 0.580 / 0.750, 성공 2 / 17 / 37 %. 진행도 분포를 보면 reach·lift 기준은 넘고 마지막 `is_within_xy`(그리퍼 연 채 물체가 용기 XY 안 80 %)가 한 번도 성립하지 않는다 — 그리퍼 열림 판정(열림 시 driver 0.017 rad → finger_joint 0.017 < 0.1) 은 정상. 배치 뒤 `scripts/diagnose_release.py` 가 놓는 순간의 물체 위치·겹침·프레임을 기록한다. 후보: (a) 놓은 뒤 물체가 튀어 나감(접촉 물성) (b) 정책이 잡은 채로 끝냄(손목 시야·조명 차이로 행동이 다름) (c) 용기 bbox 판정의 프레임 차이.

**진단 결과(2026-09-08 23시)**: 4 에피소드 모두 그리퍼가 물체에 **16–17 cm(베이스 기준, 패드 끝은 물체 위 2 cm)** 까지만 내려간 뒤 허공에서 닫힘. 놓기 문제가 아니라 접근이 짧은 것. 물리 단독 낙하 실험은 정상(스펀지가 팬 안에 안착, 업스트림 체커 True). 업스트림 사이트의 Isaac 롤아웃 영상과 손목 카메라를 비교하니 **우리 손목 뷰에 손가락이 전혀 없었다**. 원인 두 가지, 모두 우리 이식의 오류:
1. 허브 링크 스플랫의 파일명은 링크가 아니라 **메시 프림 경로**이고 스플랫은 그 프레임에 있다. 링크 프레임으로 유도한 오프셋은 panda(메시 프림 = 링크) 는 맞았지만 Robotiq 메시 프림은 회전·위치가 달라 손가락 스플랫이 15 cm 어긋났다 → 메시 프림 경로 기준으로 오프셋 재유도(`robot_link_offsets.json` v3, 키 = ply stem).
2. 스캔된 손가락 스플랫이 menagerie 손가락과 **45° 방위** 차이 → DROID 는 Robotiq 을 플랜지에 −45° 돌려 장착. `robot.build()` 에 `GRIPPER_YAW=-π/4`. 여닫는 방향과 손목 카메라 방위가 함께 맞춰짐(손가락 스플랫 중심 (0, ±0.049, 0.139) = menagerie 패드).
수정 후 손목 뷰는 업스트림 프레임과 같은 구도(손가락 양쪽 아래). 재평가 `runs/pi05_fix`(10 롤아웃 × 2) 예약됨.

**재평가 `pi05_fix`(PanClean 30, 2026-09-09 00시)**: 성공 3/30 = 10 %, 진행도 0.478 (수정 전 0/50, 0.340; 업스트림 0.75, 37 %). 분포 0.33×18 / 0.67×8 / 1.0×3 / 0×1. 진단: "닿았지만 못 듦" 이 다수. 측정: menagerie 2F-85 텐던 액추에이터(kp 100, ±5 N)는 닫는 데 **1.2 s**, Isaac 의 finger_joint 드라이브(속도 5 rad/s, 힘 200)는 ~0.2 s → 게인 ×16(닫힘 0.39 s, 열림 0.28 s)으로 재평가 `pi05_fix2` 진행 중.

단계 3: 합성 그릇 영상 → SAM 2 → TRELLIS 가 통과(4,830면 텍스처 메시). 스케일은 `upload/objects/bowl.json` 의 size_m 으로 다음 실행 때 적용.

`pi05_fix2`(그리퍼 ×16): 성공 2/30, 진행도 0.389 — 속도만으로는 안 됨. 물리 단독 실험(오라클): 평평한 평면에서는 짧은 축 집기로 +165 mm 들지만, 스캔 지지면 위에서는 스펀지가 낙하 후 **30–57° 기울어져** 어떤 접촉 파라미터(마진·마찰·solimp·강성)로도 안 잡힘. 원인: TSDF 메시가 스토브 격자의 얇은 막대를 잃어 버너 우물이 열려 있음(스플랫·메시 표면 일치 ±5 mm, IC 는 표면 2.4–4.5 cm 위 — 세 환경 공통, Isaac 도 같은 낙하). 조치: 정적 지지면을 **높이맵(1.5 cm, 지지 밴드 z≤0.15)** 으로 바꾸고, **IC 배치 영역(볼록 껍질 +5 cm)을 그 영역의 p90 높이(격자 상단 0.062)로 채움**. 결과: 스펀지 기울기 0°, 오라클 4/4 IC 에서 +150 mm. MuJoCo 의 sdflib 플러그인은 wheel 에 없어(분석 형상만) 정확한 비볼록 메시 충돌은 불가. 재평가 `pi05_fix3` 진행 중.

**남은 게이트**: `pi05_fix3` 결과 → 목표(진행도 0.75, 성공 37 %) 대비 판단 → 나머지 환경 50 롤아웃, 그리고 **실제 폰 영상**.
