"""Run matches against bundled baselines and external-bots/*/agent.py.

Example: python tools/play_baselines.py --deals 1000 --seeds 7 19
Each match gets fresh servers; the judge resets memory between halves.
"""
import argparse
from contextlib import ExitStack
from datetime import datetime
import json
import os
from pathlib import Path
import socket
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def bot_name(path):
    return path.parent.name if path.name == 'agent.py' else path.stem


def positive(text):
    value = int(text)
    if value <= 0:
        raise argparse.ArgumentTypeError('must be positive')
    return value


def stop(process):
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()


def run_match(bot, seed, args, directory):
    # Hold both sockets until launch so the selected ports are distinct.
    with socket.socket() as first, socket.socket() as second:
        first.bind(('127.0.0.1', 0))
        second.bind(('127.0.0.1', 0))
        ports = [first.getsockname()[1], second.getsockname()[1]]
    with ExitStack() as stack:
        env = os.environ.copy()
        # These are isolated local tests, independent of any shell auth settings.
        for key in ('AGENT_TOKEN', 'AGENT1_TOKEN', 'AGENT2_TOKEN'):
            env.pop(key, None)
        primary = Path(getattr(args, 'agent', ROOT / 'agent.py')).resolve()
        if primary.name == 'simple_agent.py':
            env['BOT_DIAGNOSTICS'] = '1'
        for path, port, name in zip((primary, bot), ports, ('agent', 'baseline')):
            log = stack.enter_context((directory / f'{name}.log').open('w', encoding='utf-8'))
            process = subprocess.Popen(
                [sys.executable, str(path)],
                cwd=path.parent if path.name == 'agent.py' else ROOT,
                env={**env, 'PORT': str(port)}, stdout=log, stderr=subprocess.STDOUT,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
            stack.callback(stop, process)
        command = [sys.executable, str(ROOT / 'judge_engine.py'),
                   '--agent1', f'http://127.0.0.1:{ports[0]}',
                   '--agent2', f'http://127.0.0.1:{ports[1]}',
                   '--deals', str(args.deals), '--seed', str(seed),
                   '--json', str(directory / 'result.json')]
        if args.logs:
            command.append('--logs')
        # Redirect continuously, preserving diagnostics even on timeout.
        with (directory / 'judge.log').open('w', encoding='utf-8') as log:
            judge = subprocess.Popen(command, cwd=ROOT, env=env,
                                     stdout=log, stderr=subprocess.STDOUT)
            stack.callback(stop, judge)
            code = judge.wait(timeout=args.timeout)
        if code:
            raise RuntimeError(f'judge exited with code {code}; see {directory / "judge.log"}')
    return json.loads((directory / 'result.json').read_text(encoding='utf-8'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--deals', type=positive, default=1000, help='deals per half (default: 1000)')
    parser.add_argument('--seeds', type=int, nargs='+', default=[7])
    parser.add_argument('--output', type=Path, default=ROOT / 'baseline-results')
    parser.add_argument('--timeout', type=positive, default=180, help='seconds allowed per match')
    parser.add_argument('--logs', action='store_true', help='include full hand histories in JSON')
    parser.add_argument('--agent', type=Path, default=ROOT / 'agent.py',
                        help='primary agent script (default: agent.py)')
    parser.add_argument('--bots', nargs='+', help='only these names, e.g. testpokerbot baseline_random')
    args = parser.parse_args()
    bots = sorted((ROOT / 'baseline-bots').glob('baseline_*.py'))
    bots += sorted((ROOT / 'external-bots').glob('*/agent.py'))
    if args.bots:
        unknown = set(args.bots) - {bot_name(bot) for bot in bots}
        if unknown:
            parser.error(f'unknown bots: {sorted(unknown)}')
        bots = [bot for bot in bots if bot_name(bot) in args.bots]
    if not bots:
        parser.error('no baseline bots found')
    output = args.output.resolve() / datetime.now().strftime('%Y%m%d-%H%M%S-%f')
    output.mkdir(parents=True)
    print(f'Results: {output}', flush=True)
    summary = []
    failed = False
    for bot in bots:
        for index, seed in enumerate(args.seeds):
            name = bot_name(bot)
            directory = output / f'{name}-seed{seed}-{index}'
            directory.mkdir()
            print(f'Playing {name}, seed {seed} ({args.deals * 2} hands)...', flush=True)
            row = dict(opponent=name, seed=seed)
            try:
                result = run_match(bot, seed, args, directory)
                row.update(verdict=result['verdict'], mean_bb_per_deal=result['mean_bb_per_deal'],
                           ci=result['ci'], agent_stats=result['agent_stats'])
                print(f"  {row['verdict']}: {row['mean_bb_per_deal']:+.3f} bb/deal; CI {row['ci']}", flush=True)
                stats = result['agent_stats']['agent1']
                row['healthy'] = not any(s['errors'] or s['forfeited'] or s['timed_out'] for s in stats)
                failed |= not row['healthy']
                print(f"  Our timing: {[s['time_seconds'] for s in stats]} seconds; "
                      f"errors: {sum(s['errors'] for s in stats)}; "
                      f"skipped decisions: {sum(s['skipped_calls'] for s in stats)}", flush=True)
            except (OSError, RuntimeError, ValueError, subprocess.TimeoutExpired) as exc:
                failed = True
                row['error'] = str(exc)
                print(f'  ERROR: {exc}', flush=True)
            summary.append(row)
            (output / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(f'Summary saved to {output / "summary.json"}', flush=True)
    return 1 if failed else 0


if __name__ == '__main__':
    raise SystemExit(main())
