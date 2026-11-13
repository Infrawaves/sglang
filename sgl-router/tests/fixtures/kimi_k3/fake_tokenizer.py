#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
# SPDX-License-Identifier: Apache-2.0

"""JSONL process fixture; each symlink has its own scenario and capture files."""

import json
import os
import sys
import time
from pathlib import Path

# Preserve the symlink path so every test has independent state. The executable
# itself is never rewritten while another test could fork with an open writer.
directory = Path(sys.argv[0]).parent
scenario = json.loads((directory / "fixture.json").read_text())
process_count = directory / "process-count"
started_count = int(process_count.read_text()) if process_count.exists() else 0
process_count.write_text(str(started_count + 1))
with (directory / "processes.txt").open("a", encoding="utf-8") as started:
    started.write(str(os.getpid()) + "\n")
json.loads(sys.stdin.readline())
if started_count:
    time.sleep(scenario["restart_delay_millis"] / 1000)
print(json.dumps({"ready": True}), flush=True)
replies = scenario["replies"]
counter = directory / "reply-count"
for line in sys.stdin:
    request = json.loads(line)
    with (directory / "requests.jsonl").open("a", encoding="utf-8") as captured:
        captured.write(json.dumps(request) + "\n")
    count = int(counter.read_text()) if counter.exists() else 0
    counter.write_text(str(count + 1))
    reply = replies[min(count, len(replies) - 1)]
    if reply == "__stall__":
        time.sleep(30)
        reply = json.dumps({"input_ids": [999]})
    elif reply == "__slow__":
        time.sleep(1)
        reply = json.dumps({"input_ids": scenario["generated_ids"]})
    elif reply == "__wait_for_release__":
        while not (directory / "release").exists():
            time.sleep(0.005)
        reply = json.dumps({"input_ids": scenario["generated_ids"]})
    print(reply, flush=True)
