import numpy as np
import matplotlib.pyplot as plt

collsize = [5,7.5,10,12.5,15,20,25,30,35,40,50,60]
student_data = np.load('pyth_lect4.npy')

plt.plot(collsize, student_data[0], 'b^', markersize=15, label='student 1')
plt.plot(collsize,student_data[1],'r--',linewidth=2, label='student 2')
plt.plot(collsize,student_data[2],'o',markersize=10,color=[0,1,1],
    markerfacecolor=[1,0,0], label='student 3')
plt.plot(collsize,student_data[3],'-',color=[0.5,0.5,0.5],
    linewidth=2,marker='s',markersize=17, label='student 4')
plt.legend()
plt.xlabel('Collimator Size[mm]')
plt.ylabel('Output Factor', fontsize=16)
plt.title('This is a LaTex example: $\\alpha$', fontsize=18)
plt.text(20,0.6,'This is how you enter text', fontname='fantasy', fontsize=15)
plt.grid()
plt.savefig('my_figure.png')
plt.show()

fig, axs = plt.subplots(2,2)
axs[0,0].plot(collsize, student_data[0])
axs[0,0].set_title('Student 1')
axs[0,1].plot(collsize, student_data[1])
axs[0,1].set_title('Student 2')
axs[1,0].plot(collsize, student_data[2])
axs[1,0].set_title('Student 3')
axs[1,1].plot(collsize, student_data[3])
axs[1,1].set_title('Student 4')
plt.show()

f, a = plt.subplots(4)
for i in range(4):
    a[i].plot(collsize, student_data[i])
    a[i].set_xlabel('coll size')
    a[i].set_ylabel('output ratio')
    a[i].set_title('Student {}'.format(i+1))
plt.show()

dir = 'C:/users/noahb/documents/pmpdata/'








