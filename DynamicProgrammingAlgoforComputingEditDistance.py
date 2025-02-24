import numpy as np
def Levenshtein(s1, s2):
    if s1 == "":
        return len(s2)
    if s2 == "":
        return len(s1)
    if s1[-1] == s2[-1]:
        cost = 0
    else:
        cost = 1 
    res = min([Levenshtein(s1[:-1], s2)+1,
               Levenshtein(s1, s2[:-1])+1, 
               Levenshtein(s1[:-1], s2[:-1]) + cost])
    return res

print(Levenshtein("execution", "intention"))
