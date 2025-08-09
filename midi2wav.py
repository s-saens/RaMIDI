#!/usr/bin/env python3
import argparse
import os
import sys
from typing import Optional, Tuple

import numpy as np
import soundfile as sf
import pretty_midi as pm


BIT_DEPTH_TO_SUBTYPE = {
    16: "PCM_16",
    24: "PCM_24",
    32: "PCM_32",     # 정수 32-bit
    "float32": "FLOAT"  # 부동소수 32-bit
}


def resolve_subtype(bit_depth: str) -> str:
    if bit_depth.lower() == "float32":
        return BIT_DEPTH_TO_SUBTYPE["float32"]
    try:
        bd = int(bit_depth)
    except ValueError:
        raise ValueError("bit-depth 는 16, 24, 32 또는 float32 중 하나여야 합니다.")
    if bd not in (16, 24, 32):
        raise ValueError("bit-depth 는 16, 24, 32 또는 float32 중 하나여야 합니다.")
    return BIT_DEPTH_TO_SUBTYPE[bd]


def parse_instrument(spec: Optional[str]) -> Optional[int]:
    if spec is None or str(spec).strip() == "":
        return None
    s = str(spec).strip()
    # 숫자(0-127) 직접 지정
    if s.isdigit():
        val = int(s)
        if 0 <= val <= 127:
            return val
        raise ValueError("instrument 숫자는 0-127 범위여야 합니다.")
    # 이름으로 지정(예: "Acoustic Grand Piano", "Violin", "Trumpet", "Piano" 등)
    # pretty_midi 의 이름→프로그램 변환 사용 (+ 느슨한 부분일치 허용)
    names = [pm.program_to_instrument_name(i) for i in range(128)]
    lower_names = [n.lower() for n in names]
    s_lower = s.lower()

    # 1) 정확(대소문자 무시) 일치 우선
    for i, ln in enumerate(lower_names):
        if ln == s_lower:
            return i
    # 2) 부분 포함 일치(첫 번째 매치 채택)
    for i, ln in enumerate(lower_names):
        if s_lower in ln:
            return i
    # 3) pretty_midi 변환 시도(엄밀 일치만)
    try:
        return pm.instrument_name_to_program(s)
    except Exception:
        pass

    raise ValueError(
        f"instrument 인식 실패: '{spec}'. 숫자(0-127) 또는 GM 악기명으로 지정하세요."
    )


def enforce_instrument(pretty: pm.PrettyMIDI, program: int) -> None:
    # 드럼(채널 10; pretty_midi에서는 instrument.is_drum) 제외하고 모두 동일 program으로 강제
    for inst in pretty.instruments:
        if not inst.is_drum:
            inst.program = program


def ensure_stereo_shape(wav: np.ndarray) -> np.ndarray:
    if wav.ndim == 1:
        return np.column_stack([wav, wav])
    # 2D 인 경우, (n, 2)로 정규화
    if wav.ndim == 2:
        if wav.shape[1] == 2:
            return wav
        if wav.shape[0] == 2 and wav.shape[1] != 2:
            return wav.T
    # 그 외는 그대로 반환
    return wav


def to_mono(wav: np.ndarray) -> np.ndarray:
    if wav.ndim == 1:
        return wav
    wav2 = ensure_stereo_shape(wav)
    return wav2.mean(axis=1)


def midi_to_wav(
    midi_path: str,
    wav_out_path: str,
    soundfont_path: str,
    sample_rate: int = 44100,
    bit_depth: str = "16",
    channels: str = "stereo",
    instrument: Optional[str] = None
) -> Tuple[int, str]:
    if not os.path.isfile(midi_path):
        raise FileNotFoundError(f"MIDI 파일을 찾을 수 없습니다: {midi_path}")
    if not os.path.isfile(soundfont_path):
        raise FileNotFoundError(f"SoundFont(.sf2) 파일을 찾을 수 없습니다: {soundfont_path}")

    subtype = resolve_subtype(bit_depth)
    if channels not in ("mono", "stereo"):
        raise ValueError("channels 는 'mono' 또는 'stereo' 여야 합니다.")

    pretty = pm.PrettyMIDI(midi_path)

    prog = parse_instrument(instrument) if instrument is not None else None
    if prog is not None:
        enforce_instrument(pretty, prog)

    # 합성(FluidSynth 백엔드 필요; 시스템에 libfluidsynth/fluidsynth 가 설치되어야 함)
    audio = pretty.fluidsynth(soundfont_path, fs=sample_rate)

    # 채널 처리
    if channels == "mono":
        audio = to_mono(audio)
    else:
        audio = ensure_stereo_shape(audio)

    # 안전 클리핑(부동소수 입력 시 음압 과도 방지)
    if np.issubdtype(audio.dtype, np.floating):
        max_abs = np.max(np.abs(audio)) if audio.size else 0.0
        if max_abs > 1.0 and max_abs != 0.0:
            audio = audio / max_abs

    # 쓰기
    sf.write(wav_out_path, audio, samplerate=sample_rate, subtype=subtype, format="WAV")

    return sample_rate, subtype


def main():
    parser = argparse.ArgumentParser(
        description="MIDI → WAV 변환기 (샘플률/비트/모노·스테레오/악기 강제 지정 가능)"
    )
    parser.add_argument("midi_path", help="입력 MIDI 파일 경로")
    parser.add_argument("wav_out_path", help="출력 WAV 파일 경로")
    parser.add_argument(
        "--soundfont", "-s", required=True, help="SoundFont(.sf2) 파일 경로"
    )
    parser.add_argument(
        "--sample-rate", "-r", type=int, default=44100, help="샘플률(Hz), 기본 44100"
    )
    parser.add_argument(
        "--bit-depth", "-b", default="16",
        help="비트 깊이: 16 | 24 | 32 | float32 (기본 16)"
    )
    parser.add_argument(
        "--channels", "-c", choices=["mono", "stereo"], default="stereo",
        help="채널: mono | stereo (기본 stereo)"
    )
    parser.add_argument(
        "--instrument", "-i", default=None,
        help="강제 악기 지정(0-127 또는 GM 악기명, 예: 'Piano', 'Violin'). 지정 없으면 원본 유지"
    )

    args = parser.parse_args()

    try:
        sr, subtype = midi_to_wav(
            midi_path=args.midi_path,
            wav_out_path=args.wav_out_path,
            soundfont_path=args.soundfont,
            sample_rate=args.sample_rate,
            bit_depth=args.bit_depth,
            channels=args.channels,
            instrument=args.instrument
        )
        print(f"완료: {args.wav_out_path} (sr={sr}, subtype={subtype}, channels={args.channels})")
    except Exception as e:
        print(f"오류: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main() 