if '__file__' in globals():
    import os
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from dezero import Variable
import dezero.functions as F

x1 = Variable(np.array([[1, 2, 3], [4, 5, 6]]))
y1 = F.reshape(x1, (6,))
y1.backward(retain_grad=True)
print(x1.grad)

x2 = Variable(np.array([[1, 2, 3], [4, 5, 6]]))
y2 = F.transpose(x2)
y2.backward(retain_grad=True)
print(x2.grad)
