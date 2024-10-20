#作者 卢菁 微信:13426461033
max_t=0.99
min_t=0.90
T=200
delta=(max_t-min_t)/T
#单步噪声
aerfa_m=[1]
#累积的噪声
aerfa=[0 for _ in range(0,T+1)]
for i in range(0,T+1):
    aerfa[i]=max_t-delta*i#/T

for i in range(1,T+1):
    aerfa_m.append(aerfa[i]*aerfa_m[i-1])

