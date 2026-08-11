"""cosyvoice conda env를 서브프로세스로 호출하는 경량 클라이언트.

RAG/LLM(gemma-4, transformers>=5)과 TTS(CosyVoice, transformers==4.51.3)는
같은 Python 프로세스에 절대 같이 못 들어간다 (버전 요구사항이 정면충돌).
그래서 walkme-llm env(RAG/LLM/backend)에서는 tts.synthesize를 직접 import하지
않고, 이 모듈을 통해 cosyvoice env의 별도 python 프로세스로 합성을 위임한다.

torch/transformers를 import하지 않는 순수 stdlib 모듈이라 walkme-llm env에서
안전하게 import할 수 있다.
"""

from __future__ import annotations

import os
import shutil
import signal
import subprocess
import sys
import uuid
from pathlib import Path

TTS_DIR = Path(__file__).resolve().parent
DEFAULT_OUT_DIR = TTS_DIR / "outputs"
CONDA_ENV = "cosyvoice"


def _cosyvoice_python() -> str:
    """conda run 없이 cosyvoice env의 python 인터프리터를 직접 찾는다.

    Windows의 `conda run`은 인자에 개행이 섞이면 스크립트로 못 감싼다
    (`NotImplementedError: ... arguments contain newlines`) — 사투리
    답변은 항상 여러 줄이라 매번 이 버그에 걸린다. 그래서 `conda run`
    래퍼를 거치지 않고, 지금 실행 중인 인터프리터(walkme-llm env)의
    경로에서 conda root를 역산해 cosyvoice env의 python을 직접 찾아
    실행한다 (mac/리눅스에서도 동일하게 동작).
    """
    here = Path(sys.executable).resolve()
    for ancestor in here.parents:
        env_dir = ancestor / "envs" / CONDA_ENV
        if env_dir.exists():
            for rel in ("python.exe", "bin/python"):
                candidate = env_dir / rel
                if candidate.exists():
                    return str(candidate)
    raise RuntimeError(
        f"cosyvoice conda env의 python을 찾지 못했습니다 "
        f"(conda root를 {here} 기준으로 탐색했지만 실패)"
    )


def synthesize_via_cosyvoice_env(
    text: str,
    out_path: Path | str | None = None,
    *,
    prompt_text: str | None = None,
    prompt_wav: Path | str | None = None,
    timeout: float = 240.0,
) -> Path:
    """cosyvoice env의 synthesize.py를 서브프로세스로 실행해 wav를 생성한다.

    `conda run`은 내부적으로 셸 스크립트를 거쳐 실제 python을 exec하므로,
    직계 자식만 kill하면 손자 프로세스(진짜 작업)가 고아로 남는다. 그래서
    새 프로세스 그룹으로 띄우고 타임아웃 시 그룹 전체를 kill한다.

    호출마다 새 프로세스로 뜨기 때문에 매번 체크포인트(~3.4GB)를 디스크에서
    다시 읽고, `wetext` 프론트엔드가 로컬 캐시가 있어도 modelscope.cn에
    온라인 검증 요청을 두 번 보낸다(라이브러리 쪽에 local_files_only 우회
    옵션이 없음) — 정상 GPU 케이스에서도 이 오버헤드만으로 수십 초가 든다.
    그래서 타임아웃을 넉넉히 잡았다: 이 시간 안에도 안 끝나면 GPU를 못 잡고
    CPU로 넘어갔거나 뭔가 잘못됐다는 신호로 본다.
    """
    if out_path is None:
        DEFAULT_OUT_DIR.mkdir(parents=True, exist_ok=True)
        out_path = DEFAULT_OUT_DIR / f"guide_{uuid.uuid4().hex[:10]}.wav"
    else:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        _cosyvoice_python(), "synthesize.py", text, "--out", str(out_path),
    ]
    if prompt_text:
        cmd += ["--prompt-text", prompt_text]
    if prompt_wav:
        cmd += ["--prompt-wav", str(prompt_wav)]

    mem_limit = os.environ.get("WALKME_TTS_MEM_LIMIT", "10G")
    if shutil.which("systemd-run"):
        # cgroup 메모리 상한 + 스왑 금지: 한도 초과 시 느린 스왑 스래싱(시스템 전체
        # 먹통) 대신 이 프로세스만 즉시 OOM-kill되게 한다. --scope는 명령을
        # 호출 프로세스의 동기 자식으로 실행하므로 아래 killpg 로직과 그대로 맞물린다.
        cmd = [
            "systemd-run", "--user", "--scope", "--collect",
            "-p", f"MemoryMax={mem_limit}", "-p", "MemorySwapMax=0",
            "--",
        ] + cmd
    else:
        print(
            "[경고] systemd-run을 찾을 수 없어 TTS 서브프로세스에 메모리 한도를 "
            "걸지 못했습니다 (제한 없이 실행).",
            file=sys.stderr,
        )

    proc = subprocess.Popen(
        cmd,
        cwd=str(TTS_DIR),
        stdout=sys.stdout,
        stderr=sys.stderr,
        start_new_session=True,  # 별도 프로세스 그룹 -> 타임아웃 시 그룹 전체 kill 가능
    )
    try:
        returncode = proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except ProcessLookupError:
            pass
        proc.wait()
        raise RuntimeError(
            f"TTS 서브프로세스가 {timeout:.0f}초 안에 끝나지 않아 강제 종료했습니다. "
            "정상이면 GPU에서 수십 초 안에 끝나야 하니, GPU를 못 잡고 CPU로 넘어갔을 "
            "가능성이 큽니다 — LLM이 VRAM을 아직 안 놓았는지(_free_llm) 또는 다른 "
            "프로세스가 GPU를 점유 중인지 nvidia-smi로 확인하세요."
        ) from None

    if returncode != 0:
        hint = ""
        if returncode is not None and returncode < 0:
            hint = (
                f" 시그널로 강제 종료됐습니다 — 메모리 한도(WALKME_TTS_MEM_LIMIT="
                f"{mem_limit})를 초과해 OOM-kill됐을 가능성이 있습니다. 필요하면 "
                "환경변수를 올려서 재시도하세요."
            )
        raise RuntimeError(
            f"TTS 서브프로세스 실패 (exit={returncode}), cosyvoice env를 확인하세요."
            f"{hint}"
        )
    if not out_path.exists():
        raise RuntimeError(f"TTS 서브프로세스는 성공했지만 출력 파일이 없습니다: {out_path}")
    return out_path
