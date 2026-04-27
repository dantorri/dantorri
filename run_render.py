import os
import sys
import runpy

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch  # force torch to load first

print("torch ok")
print(torch.__version__)

# Forward all command line args to the module
sys.argv = ["ppo.render_ppo", *sys.argv[1:]]
runpy.run_module("ppo.render_ppo", run_name="__main__")