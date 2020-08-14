# in case you don't want to run keras on your GPU

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'