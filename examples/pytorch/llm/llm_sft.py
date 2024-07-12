# Copyright (c) Alibaba, Inc. and its affiliates.
import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
import custom

from swift.llm import sft_main

if __name__ == '__main__':
    output = sft_main()
