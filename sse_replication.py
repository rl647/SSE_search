#%%


import sys
import numpy as np
p1=sys.argv[1]
p2=sys.argv[2]
sse = {}
sse_range = {}
with open(f'{p1}')as f, \
    open(f'{p2}','w')as f1:
    for line in f:
        s=line.strip().split('\t')
        s2 = s[2].strip().split(' ')
        s2 = [np.array(e.strip().split('-'),dtype=int) for e in s2]
        s2r = [e+s2[-1][-1] for e in s2]
        s2+=s2r
        s2 = [f'{e[0]}-{e[1]}' for e in s2]
        f1.write(s[0]+'\t'+s[1]+s[1]+'\t'+' '.join(s2)+'\t')
        # print(s2,'\n',s2r)
        # break 
# %%
