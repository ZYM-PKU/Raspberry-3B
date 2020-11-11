from pyswip import Prolog # 导入模块
prolog = Prolog()
prolog.consult('bird.pl') # 导入Prolog 的代码（规则库）
prolog.assertz("color(blue)") # 添加事实
prolog.assertz('season(all_year)')
prolog.assertz('size(small)')

for result in prolog.query('bird(X)'): # query 代表查询
    print(result["X"])