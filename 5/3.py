from pyswip import Prolog, registerForeign

allfacts={}
def verify(t):
    if t in allfacts:
        if allfacts[t] == 'yes':
            return 1
        if allfacts[t] == 'no':
            return 0
    ans=input("是否{}(y/n)?  ".format(t))
    if (ans=='y') or (ans=='yes'):
        allfacts[t]='yes'
        return 1
    else:
        allfacts[t]='no'
        return 0
verify.arity = 1

registerForeign(verify)

prolog = Prolog()
prolog.consult("animal.pl")
for soln in prolog.query("hypothesize(X)"):
    print("这个动物是：", soln["X"])