import numpy as np

wh1i = np.random.uniform(-100.0,100.0,(100,784))
wh2h1 = np.random.uniform(-100.0,100.0,(50,100))
woh2 = np.random.uniform(-100.0,100.0,(10,50))

np.save('wh1i',wh1i)
np.save('wh2h1',wh2h1)
np.save('woh2',woh2)
