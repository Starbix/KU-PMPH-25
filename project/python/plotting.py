import matplotlib.pyplot as plt
import numpy as np


seq_lens = [1024, 2048, 4096, 8192]
speedups_flash_standard = [3.1430990236642637, 3.1125847506391295, 1.9598153504248523, 1.0582045736990746]



plt.plot(seq_lens, speedups_flash_standard)
plt.savefig('speedups.png')
plt.show()