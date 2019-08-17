# Wrapper script for pytorch/main_ignite.py, passes through rest of arguments
# Annoyingly has to be written in python

import os
import sys

# Wrapper args
# Bizarrely, have to put "empty" arg at start that gets swallowed?
args = ["LSTM","LSTM","train","--workspace=workspace_full","--tr_snr=5","--te_snr=5"]

# Extra args, skip name of our own script
args += sys.argv[1:]

# Now call the real program. Use execvp so _path_ is used
os.execvp('pytorch/main_ignite.py', args=args)