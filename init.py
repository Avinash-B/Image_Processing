import numpy as np

wh1i = np.random.rand(100,784)
wh2h1 = np.random.rand(50,100)
woh2 = np.random.rand(10,50)

np.save('wh1i',wh1i)
np.save('wh2h1',wh2h1)
np.save('woh2',woh2)
