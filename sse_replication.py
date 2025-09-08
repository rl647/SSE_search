#%%


import sys
import numpy as np
p1=sys.argv[1]
p2=sys.argv[2]
p3=sys.argv[3]
sse = {}
sse_range = {}
with open(f'{p1}')as f, \
    open(f'{p2}','w')as f1:
    for line in f:
        s=line.strip().split('\t')
        sse[s[0]] = s[1]
        
        s2 = s[2].strip().split(' ')
        s2 = [np.array(e.strip().split('-'),dtype=int) for e in s2]
        
        s2r = [e+s2[-1][-1] for e in s2]
        sse_range[s[0]] = [s2,s2r]
        s2+=s2r
        s2 = [f'{e[0]}-{e[1]}' for e in s2]
        f1.write(s[0]+'\t'+s[1]+s[1]+'\t'+' '.join(s2)+'\t')
        # print(s2,'\n',s2r)
        # break 
with open(f'{p3}/ss1.txt','w')as f1, open(f'{p3}/ssr1.txt','w') as f2, open(f'{p3}/ss2.txt','w')as f3, open(f'{p3}/ssr2.txt','w') as f4:
    for key,val in sse.items():
        f1.write(key+'\t'+val+'\n')
        f3.write(key+'\t'+val+val+'\n')
        v1 = [f'{e[0]}-{e[1]}' for e in val[0]]
        v2 = [f'{e[0]}-{e[1]}' for e in val[0]+val[1]]
        f2.write(key+'\t'+'\t'.join(v1))
        f4.write(key+'\t'+'\t'.join(v2))

# %%
