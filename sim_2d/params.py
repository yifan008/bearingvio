VT_SIGMA = 1e-2
WT_SIGMA = 1e-4
BEARING_SIGMA = 0.01 # 0.5 # 0.01

BEARING_VAR = BEARING_SIGMA * BEARING_SIGMA

SIM_FLAG = 1 # 1-circle 2-towards landmark

ITER_NUM = 1000

STEP = 1
DURATION = 240

DATA_RECORDER = False

PRINT_TIME = True

DRAW_BOUNDS = False

Flag3 = True

algorithms = ['ideal', 'ekf', 'tekf']

# TIME_MARK = '2025-07-12 15:39:34'
# TIME_MARK = '2025-11-22 21:19:59' # towards landmark
TIME_MARK = '2025-11-22 21:49:35' # circle
