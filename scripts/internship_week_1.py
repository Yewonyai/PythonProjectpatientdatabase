import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from numpy import random
#
# sns.displot([0, 1, 2, 3, 4, 5])
#
# plt.show()
x = random.normal(size=(2, 3))

print(x)
def myadd (x,y):
    return x+y


myadd = np.frompyfunc(myadd, 2,1)
print(myadd, [2,3,4,5,1,6,7,8],[9,2,4,5,6,7,3,4])



# student_scores = np.array([
#     [55, 72, 90],
#     [43, 68, 55],
#     [90, 85, 78],
#     [30, 45, 60],
#     [70, 70, 70],
#     [88, 91, 95],
#     [20, 35, 40],
#     [65, 58, 72],
#     [50, 50, 50],
#     [78, 82, 77]
# ])
# print(np.mean(student_scores,axis=0))
# highest_student = np.argmax(np.mean(student_scores,axis = 1))
# lowest_student = np.argmin(np.mean(student_scores, axis = 1))
# print(f"highest score: student,{highest_student}")
# print(f"lowest score: student, {lowest_student}")
#
#
# student_scores_below_50 = student_scores < 50
# students_below=np.any(student_scores_below_50, axis=1)
# print(f"students who scored below 50: ₹{np.sum(students_below)}")
# #
#
