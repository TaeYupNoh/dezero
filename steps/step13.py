import numpy as np

class Variable:
    def __init__(self,data):
        if data is not None:
            if not isinstance(data, np.ndarray):
                raise TypeError('{}은(는) 지원하지 않습니다'.format(type(data)))
        
        self.data = data
        self.grad = None
        self.creator = None
        
    def set_creator(self,func):
        self.creator = func
        
    def backward(self):
        if self.grad is None:
            self.grad = np.ones_like(self.data)

        funcs = [self.creator]
        while funcs:
            f = funcs.pop()
            gys = [output.grad for output in f.outputs] # outputs -> 여러 개의 출력변수
            gxs = f.backward(*gys) # 역전파 함수 호출, 여러 개 입력변수
            if not isinstance(gxs,tuple): # 튜플이 아니라면
                gxs = (gxs,) # 튜플로 변환
            
            for x,gx in zip(f.inputs,gxs):# 역전파로 전파되는 미분값을
                x.grad = gx # Variable의 인스턴스 변수 grad에 저장
							# i번째 원소에 대해 f.inputs[i]의 미분값 -> gxs[i]
                if x.creator is not None:
                    funcs.append(x.creator)
                
def as_array(x):
    if np.isscalar(x):
        return np.array(x)
    return x

class Function:
    def __call__(self, *inputs): # 가변 길이 인수
        xs = [x.data for x in inputs]
        ys = self.forward(*xs)
        if not isinstance(ys, tuple):
            ys = (ys,)
        outputs = [Variable(as_array(y)) for y in ys]
        
        for output in outputs:
            output.set_creator(self)
        self.inputs = inputs
        self.outputs = outputs

        return outputs if len(outputs) > 1 else outputs[0]
    
    def forward(self, x):
        raise NotImplementedError()
    
    def backward(self, x):
        raise NotImplementedError()
    
class Add(Function):
    def forward(self, x0, x1):
        y = x0 + x1
        return y
    
    def backward(self, gy):
        return gy, gy
    
def add(x0, x1):
    return Add()(x0, x1)

class Square(Function):
    def forward(self, x):
        y = x ** 2
        return y
    
    def backward(self, gy):
        x = self.inputs[0].data
        gx = 2 * x * gy
        return gx

def square(x):
    f = Square()
    return f(x)

x = Variable(np.array(2.0))
y = Variable(np.array(3.0))
z = add(square(x), square(y))
z.backward()
print(z.data)
print(x.grad)
print(y.grad)