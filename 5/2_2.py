from pyswip import Prolog, registerForeign

def hello(t): # 包含一个参数，返回值为布尔类型
    print("Hello,", t)

hello.arity = 1 # 这个属性是必须的
registerForeign(hello)
prolog = Prolog()
prolog.assertz("father(michael,john)") # 事实1：michael 是john 的父亲
prolog.assertz("father(michael,gina)") # 事实2：michael 是gina 的父亲

print(list(prolog.query("father(michael,X), hello(X)"))) # 查询
