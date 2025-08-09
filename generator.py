#!/usr/bin/env python3

import argparse
import random
import os
from dataclasses import dataclass
from typing import List, Tuple

try:
    from mido import Message, MidiFile, MidiTrack, MetaMessage, bpm2tempo
except Exception as exc:  # pragma: no cover
    raise SystemExit(
        "mido 패키지가 필요합니다. 설치: pip install mido"
    ) from exc


@dataclass
class GeneratorConfig:
    output_path: str
    tempo_bpm: int  # 0이면 랜덤
    bars: int
    meter_numerator: int
    meter_denominator: int
    min_beat: int  # 1(온음표), 2, 4, 8, 16, ...
    note_count_target: int
    note_count_deviation: int
    seed: int
    program: int  # 0~127, GM program number
    channel: int  # 0~15
    pitch_low: int  # MIDI note number low bound
    pitch_high: int  # MIDI note number high bound (inclusive)
    max_chord_size: int  # 한 이벤트에서 동시에 울릴 최대 노트 수
    chord_probability: float  # 각 이벤트가 화음이 될 확률 (0~1)
    ticks_per_beat: int  # PPQ


def parse_meter(meter_str: str) -> Tuple[int, int]:
    parts = meter_str.strip().split('/')
    if len(parts) != 2:
        raise ValueError("METER는 '숫자/숫자' 형식이어야 합니다. 예: 4/4, 3/4, 6/8")
    n, d = int(parts[0]), int(parts[1])
    if n <= 0 or d not in (1, 2, 4, 8, 16, 32):
        raise ValueError("유효하지 않은 박자표입니다. 분모는 1,2,4,8,16,32 중 하나여야 합니다.")
    return n, d


def clamp(value: int, low: int, high: int) -> int:
    return max(low, min(high, value))


def partition_total_notes_into_chords(total_notes: int, max_groups: int, max_chord_size: int, chord_probability: float, rng: random.Random) -> List[int]:
    """총 노트 수를 화음(동시발음) 단위의 그룹 크기 리스트로 분할한다.

    - 그룹의 개수는 max_groups를 넘지 않도록 조정한다.
    - 각 그룹 크기는 1~max_chord_size 사이이며, 총합이 total_notes가 되도록 한다.
    - chord_probability를 통해 1보다 큰 그룹(화음) 비율을 대략적으로 유도한다.
    """
    total_notes = max(1, total_notes)
    sizes: List[int] = []
    remaining = total_notes

    # 1차 분배: 확률적으로 화음 생성
    while remaining > 0:
        can_make_chord = remaining >= 2 and max_chord_size >= 2
        if can_make_chord and (rng.random() < chord_probability):
            high_inclusive = min(max_chord_size, remaining)
            # 방어: high_inclusive가 2 미만이면 단음으로 처리
            if high_inclusive >= 2:
                chord_size = rng.randint(2, high_inclusive)
            else:
                chord_size = 1
        else:
            chord_size = 1
        sizes.append(chord_size)
        remaining -= chord_size

    # 그룹 수가 너무 많으면 병합
    while len(sizes) > max_groups:
        # 가장 작은 두 그룹을 병합하여 그룹 수를 줄인다
        sizes.sort()
        if len(sizes) >= 2:
            merged = sizes[0] + sizes[1]
            # 병합된 그룹이 max_chord_size를 초과하면 초과분을 다시 쪼갠다
            if merged <= max_chord_size:
                sizes = [merged] + sizes[2:]
            else:
                # 최대 크기를 유지하며 나머지를 새로운 그룹으로
                sizes = [max_chord_size, merged - max_chord_size] + sizes[2:]
        else:
            break

    # 혹시 0이나 음수가 들어갔을 안전망
    sizes = [s for s in sizes if s > 0]
    if sum(sizes) != total_notes:
        # 합이 틀어졌다면 마지막 그룹을 조정
        diff = total_notes - sum(sizes)
        if sizes:
            sizes[-1] = clamp(sizes[-1] + diff, 1, max_chord_size)
        else:
            sizes = [clamp(diff, 1, max_chord_size)]

    return sizes


def choose_unique_positions_in_bar(group_count: int, steps_per_bar: int, rng: random.Random) -> List[int]:
    """막대 내에서 group_count 개수만큼의 고유한 스텝 인덱스를 무작위로 선택한다."""
    group_count = min(group_count, steps_per_bar)
    positions = rng.sample(range(steps_per_bar), group_count)
    positions.sort()
    return positions


def generate_random_scale_pitches(low: int, high: int, chord_size: int, rng: random.Random) -> List[int]:
    """간단한 메이저 스케일 기반 피치 선택 (듣기 좋게)."""
    # 랜덤 루트 선택, 메이저 스케일 간격
    major_intervals = [0, 2, 4, 5, 7, 9, 11]
    # 48(C3)~72(C5) 사이에서 루트 선택
    root = rng.randint(max(24, low), min(72, high))
    # 루트를 동일 옥타브 내 C 기준으로 정렬
    root = root - (root % 12)

    candidates: List[int] = []
    # 여러 옥타브에 걸쳐 스케일 음 모으기
    for octave in range(-2, 3):
        base = root + 12 * octave
        for deg in major_intervals:
            p = base + deg
            if low <= p <= high:
                candidates.append(p)

    if not candidates:
        # 범위가 너무 좁으면 균등 랜덤으로 대체
        return rng.sample(range(low, high + 1), k=min(chord_size, max(1, high - low + 1)))

    # 화음 내에서는 중복 피치를 피하려고 하되, 후보가 부족하면 허용
    unique_count = min(chord_size, len(set(candidates)))
    selected = rng.sample(list(sorted(set(candidates))), k=unique_count)
    while len(selected) < chord_size:
        selected.append(rng.choice(candidates))
    return selected


def build_midi(config: GeneratorConfig) -> MidiFile:
    rng = random.Random(config.seed)

    # 템포 결정
    tempo_bpm = config.tempo_bpm if config.tempo_bpm != 0 else rng.randint(60, 160)

    # 준비
    midi = MidiFile(ticks_per_beat=config.ticks_per_beat)
    track = MidiTrack()
    midi.tracks.append(track)

    # 메타 메시지: 템포, 박자
    track.append(MetaMessage('set_tempo', tempo=bpm2tempo(tempo_bpm), time=0))
    track.append(MetaMessage('time_signature', numerator=config.meter_numerator, denominator=config.meter_denominator, time=0))

    # 프로그램 체인지 (악기)
    track.append(Message('program_change', program=clamp(config.program, 0, 127), channel=clamp(config.channel, 0, 15), time=0))

    # 계산용 상수
    quarter_ticks = config.ticks_per_beat  # 1/4음표 틱 수
    beats_per_bar_quarter_based = config.meter_numerator * (4 / config.meter_denominator)
    bar_length_ticks = int(round(beats_per_bar_quarter_based * quarter_ticks))

    # 최소 단위 스텝: MIN_BEAT 기준
    # 예) MIN_BEAT=16 -> 1/16음표 = 0.25 beat -> step_ticks = quarter_ticks * (4/16) = quarter_ticks/4
    step_ticks = int(round(quarter_ticks * (4 / config.min_beat)))
    if step_ticks <= 0:
        raise ValueError("MIN_BEAT이 너무 큽니다. step_ticks가 0 이하가 되었습니다.")

    steps_per_bar = max(1, bar_length_ticks // step_ticks)

    # 진행 시간 누적 (delta time 기반)
    current_time_ticks = 0

    for bar_index in range(config.bars):
        # 진행 상황 출력(루프 내 출력은 end='\r')
        print(f"생성 중: {bar_index + 1}/{config.bars} 마디", end='\r')

        # 바마다 목표 노트 수 결정
        deviation = rng.randint(-config.note_count_deviation, config.note_count_deviation)
        total_notes_this_bar = max(1, config.note_count_target + deviation)

        # 그룹(동시발음) 크기 분해
        chord_sizes = partition_total_notes_into_chords(
            total_notes=total_notes_this_bar,
            max_groups=steps_per_bar,
            max_chord_size=config.max_chord_size,
            chord_probability=config.chord_probability,
            rng=rng,
        )

        # 이벤트(그룹) 개수에 맞춰 바 내 고유 위치 선택
        positions = choose_unique_positions_in_bar(len(chord_sizes), steps_per_bar, rng)

        # 각 이벤트의 지속시간(스텝 단위) 무작위 지정, 바를 넘지 않도록 조정
        durations_steps: List[int] = []
        for pos in positions:
            max_steps_here = max(1, steps_per_bar - pos)
            durations_steps.append(rng.randint(1, max_steps_here))

        # 이벤트들을 바 내부 절대 틱 기준으로 정렬하여 삽입
        events: List[Tuple[int, int, int]] = []  # (start_tick_in_bar, duration_ticks, chord_index)
        for idx, pos in enumerate(positions):
            start_tick_in_bar = pos * step_ticks
            duration_ticks = durations_steps[idx] * step_ticks
            # 바 끝에서 컷
            duration_ticks = min(duration_ticks, bar_length_ticks - start_tick_in_bar)
            events.append((start_tick_in_bar, duration_ticks, idx))
        events.sort(key=lambda x: x[0])

        # 노트 온/오프 메시지 추가
        for start_tick_in_bar, duration_ticks, chord_idx in events:
            # delta time으로 전환하기 위해 전 이벤트로부터의 차이를 계산
            absolute_start = bar_index * bar_length_ticks + start_tick_in_bar
            delta_to_event = absolute_start - current_time_ticks
            if delta_to_event < 0:
                delta_to_event = 0  # 안전망

            # 화음 피치 선택
            chord_size = chord_sizes[chord_idx]
            pitches = generate_random_scale_pitches(
                config.pitch_low, config.pitch_high, chord_size, rng
            )

            # Note On 여러 개 동시 시작
            for i, pitch in enumerate(pitches):
                # 첫 Note On은 해당 delta, 나머지는 0으로 동시발음 구현
                delta_time = delta_to_event if i == 0 else 0
                track.append(
                    Message(
                        'note_on',
                        note=clamp(pitch, 0, 127),
                        velocity=rng.randint(60, 110),
                        time=delta_time,
                        channel=config.channel,
                    )
                )
            # 이제 현재 시간은 이벤트 시작 시점으로 이동
            current_time_ticks = absolute_start

            # Note Off들 추가 (동일 duration, 동시 종료)
            # 첫 Note Off의 delta는 duration_ticks, 이후는 0
            for i, pitch in enumerate(pitches):
                delta_time = duration_ticks if i == 0 else 0
                track.append(
                    Message(
                        'note_off',
                        note=clamp(pitch, 0, 127),
                        velocity=0,
                        time=delta_time,
                        channel=config.channel,
                    )
                )
            # 현재 시간은 이벤트 종료 시점으로 이동
            current_time_ticks += duration_ticks

        # 바 끝에 남은 시간(다음 바의 시작까지)만큼 쉬기
        bar_end_absolute = (bar_index + 1) * bar_length_ticks
        if current_time_ticks < bar_end_absolute:
            track.append(Message('note_off', note=0, velocity=0, time=bar_end_absolute - current_time_ticks, channel=config.channel))
            current_time_ticks = bar_end_absolute

    # 마지막 줄바꿈
    print()

    return midi


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="랜덤 MIDI 생성기")
    parser.add_argument('--output', required=True, help='출력 MIDI 파일 경로 (.mid)')
    parser.add_argument('--tempo', type=int, default=120, help='BPM, 0이면 랜덤 (기본 0)')
    parser.add_argument('--bars', type=int, default=16, help='마디 수 (기본 8)')
    parser.add_argument('--meter', type=str, default='4/4', help="박자표 예: '4/4', '3/4', '6/8' (기본 4/4)")
    parser.add_argument('--min-beat', type=int, default=16, help='최소 몇분음표 단위인지 (1=온음표, 2,4,8,16,...) (기본 16)')
    parser.add_argument('--note-cnt', type=int, default=8, help='한 마디 당 노트의 목표 개수 (기본 8)')
    parser.add_argument('--note-cnt-deviation', type=int, default=2, help='한 마디 노트 개수의 편차 (기본 2)')
    parser.add_argument('--seed', type=int, default=42, help='랜덤 시드 (기본 42)')
    parser.add_argument('--program', type=int, default=0, help='GM Program Number 0~127 (기본 0: Acoustic Grand Piano)')
    parser.add_argument('--channel', type=int, default=0, help='MIDI 채널 0~15 (기본 0)')
    parser.add_argument('--pitch-low', type=int, default=48, help='피치 하한 (기본 48=C3)')
    parser.add_argument('--pitch-high', type=int, default=72, help='피치 상한, 포함 (기본 72=C5)')
    parser.add_argument('--max-chord-size', type=int, default=3, help='한 이벤트 최대 동시 음 수 (기본 3)')
    parser.add_argument('--chord-probability', type=float, default=0.35, help='이벤트가 화음일 확률 (기본 0.35)')
    parser.add_argument('--ppq', type=int, default=480, help='ticks per beat (PPQ) (기본 480)')
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    meter_n, meter_d = parse_meter(args.meter)

    config = GeneratorConfig(
        output_path=args.output,
        tempo_bpm=args.tempo,
        bars=max(1, args.bars),
        meter_numerator=meter_n,
        meter_denominator=meter_d,
        min_beat=max(1, args.min_beat),
        note_count_target=max(1, args.note_cnt),
        note_count_deviation=max(0, args.note_cnt_deviation),
        seed=args.seed,
        program=clamp(args.program, 0, 127),
        channel=clamp(args.channel, 0, 15),
        pitch_low=clamp(args.pitch_low, 0, 127),
        pitch_high=clamp(args.pitch_high, 0, 127),
        max_chord_size=max(1, args.max_chord_size),
        chord_probability=min(max(args.chord_probability, 0.0), 1.0),
        ticks_per_beat=max(24, args.ppq),
    )

    if config.pitch_low > config.pitch_high:
        config.pitch_low, config.pitch_high = config.pitch_high, config.pitch_low

    if config.min_beat not in (1, 2, 4, 8, 16, 32):
        raise SystemExit("MIN_BEAT은 1,2,4,8,16,32 중 하나여야 합니다.")

    midi = build_midi(config)
    # 출력 폴더 자동 생성
    out_dir = os.path.dirname(config.output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    midi.save(config.output_path)
    print(f"저장 완료: {config.output_path}")


if __name__ == '__main__':
    main()
