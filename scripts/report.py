"""S5.5 report: MuJoCo (ours) vs upstream Isaac (paper / issue #24) per environment, as Markdown.
usage: .venv/bin/python scripts/report.py [--run pi05] > docs/REPORT_pi05.md"""
import argparse, math
from polaris_mujoco import results

def main(run):
    rows = [r for r in results.list_runs() if r["run"].startswith(run + "/")]
    print(f"# π0.5 (pi05_droid_jointpos_polaris) on MuJoCo — run `{run}`\n")
    print("업스트림 코드(`scripts/eval.py`, 루브릭, IC, 정책 클라이언트) 무수정. 시뮬레이터·렌더 전경만 MuJoCo/OpenGL (PLAN §8.5).\n")
    print("| 환경 | n | 성공 | 성공률 (Wilson 90 %) | 평균 진행도 | 업스트림 진행도: 논문 / issue #24 (성공률) | 비교 가능 |")
    print("|---|---|---|---|---|---|---|")
    for r in rows:
        env = r["run"].split("/")[-1]; ref = r.get("reference") or {}
        lo, hi = r["wilson90"]
        print(f"| {env} | {r['episodes']} | {r['successes']} | {100*r['success_rate']:.0f} % ({100*lo:.0f}–{100*hi:.0f}) | {r['mean_progress']:.3f} | "
              f"{ref.get('paper_progress', float('nan')):.2f} / {ref.get('issue24_progress', float('nan')):.2f} ({100*ref.get('issue24_success', float('nan')):.0f} %) | {'예' if ref.get('fair') else '아니오 (업스트림 리셋 상태 파손)'} |")
    print("\n진행도 = 루브릭 기준 도달 비율(max-ever), 에피소드 450 스텝·15 Hz, IC 는 허브의 `initial_conditions.json` 순환. 성공률의 이항 구간은 Wilson 90 %; 진행도 구간은 부트스트랩으로 따로 계산할 것.")

if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--run", default="pi05"); main(p.parse_args().run)
