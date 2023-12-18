# Copyright (c) Alibaba, Inc. and its affiliates.

import sys
swift_dir = "/home/hanlv/workspace/code/research/infodemic/LLM/swift/examples/pytorch/llm"
if not swift_dir in sys.path:
    sys.path.append(swift_dir)

import custom

from swift.llm.run import app_ui_main


if __name__ == '__main__':
    app_ui_main()
