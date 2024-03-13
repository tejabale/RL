
import matplotlib.pyplot as plt
#graph1

p = [0,0.1,0.2,0.3,0.4,0.5]
values1 = [0.7, 0.28672, 0.18, 0.126, 0.108, 0.1]


q = [0.6,0.7,0.8,0.9,1]
values2 = [0.08, 0.126, 0.2, 0.3, 0.4]


# plt.plot(p, values1, linestyle='-')
# plt.xlabel('p')
# plt.ylabel('probability of winning')
# plt.title('probability of winning starting from position [05, 09, 08, 1] and q = 0.7')
# plt.savefig('plot1.png')

plt.plot(q, values2, linestyle='-')
plt.xlabel('q')
plt.ylabel('probability of winning')
plt.title('probability of winning starting from position [05, 09, 08, 1] and p = 0.3')
plt.savefig('plot2.png')
