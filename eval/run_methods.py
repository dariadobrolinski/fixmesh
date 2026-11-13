"""
Run one or more repair methods on a mesh and save each result separately,
so outputs can be opened side by side instead of overwriting one file.

Each run writes <output_dir>/<method>.stl and <method>.log (console output).

To run:
python eval/run_methods.py <input.stl> [--methods kenichi graft ...] [--output-dir examples/outputs]
    [--set name=value ...]   e.g. --set units=mm max_displacement=1.5
    [--name NAME]            files are saved as NAME.stl / NAME.log
                             instead of <method>.stl (one method only)
"""

import argparse
import ast
import contextlib
import os
import sys
import time

import pymesh

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import fixmesh  # noqa: E402

METHODS = {
    "kenichi": fixmesh.kenichi_cut_repair,
    "cut": fixmesh.cut_repair,
    "push_apart": fixmesh.push_apart_repair,
}


def _setting(text):
    name, _, value = text.partition("=")
    try:
        value = ast.literal_eval(value)
    except (ValueError, SyntaxError):
        pass  # plain words like mm are kept as strings
    return name, value


class _Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, text):
        for stream in self.streams:
            stream.write(text)
            stream.flush()

    def flush(self):
        for stream in self.streams:
            stream.flush()


def main():
    parser = argparse.ArgumentParser(description="Run repair methods side by side.")
    parser.add_argument("input")
    parser.add_argument("--methods", nargs="+", choices=list(METHODS),
                        default=list(METHODS))
    parser.add_argument("--output-dir", default="examples/outputs")
    parser.add_argument("--set", nargs="+", type=_setting, default=[],
                        metavar="NAME=VALUE",
                        help="Settings passed to every method run.")
    parser.add_argument("--name", help="File name for the result (one method only).")
    args = parser.parse_args()
    if args.name and len(args.methods) > 1:
        parser.error("--name works with one method at a time")
    settings = dict(args.set)

    os.makedirs(args.output_dir, exist_ok=True)
    for name in args.methods:
        stem = args.name or name
        out_path = os.path.join(args.output_dir, f"{stem}.stl")
        log_path = os.path.join(args.output_dir, f"{stem}.log")
        with open(log_path, "w") as log, \
                contextlib.redirect_stdout(_Tee(sys.stdout, log)):
            print(f"== {name} on {args.input} {settings or ''}")
            start = time.time()
            try:
                result = METHODS[name](pymesh.load_mesh(args.input), **settings)
            except Exception as error:  # keep going with the other methods
                print(f"FAILED: {error!r}")
                continue
            result.export(out_path)
            print(f"== {name} finished in {time.time() - start:.1f}s -> {out_path}")


if __name__ == "__main__":
    main()
