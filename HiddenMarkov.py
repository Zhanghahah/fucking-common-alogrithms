import numpy as np
class HiddenMarkov:
    def forward(self, Q, V, A, B, O, PI):
        N = len(Q)
        M = len(O)
        alphas = np.zeros((N,M))
        T = M
        for t in range(T):
            indexOf0 = V.index(O[t])
            for i in range(N):
                if t == 0:
                    alphas[i][t] = PI[t][i] * B[i][indexOf0]
                    print('alpha1(%d)=p%db%db(o1)=%f' % (i, i, i, alphas[i][t]))
                else:
                    alphas[i][t] = np.dot([alpha[t-1] for alpha in alphas], [a[i] for a in A] ) * B[i][indexOf0]
                    print('alpha%d(%d)=[sigma alpha%d(i)ai%d]b%d(o%d)=%f' % (t, i, t - 1, i, i, t, alphas[i][t]))

        P = np.sum([alpha[M-1] for alpha in alphas])

Q = [1, 2, 3]
V = ['红', '白']
A = [[0.5, 0.2, 0.3], [0.3, 0.5, 0.2], [0.2, 0.3, 0.5]]
B = [[0.5, 0.5], [0.4, 0.6], [0.7, 0.3]]
# O = ['红', '白', '红', '红', '白', '红', '白', '白']
O = ['红', '白', '红', '白']    #习题10.1的例子
PI = [[0.2, 0.4, 0.4]]

HMM = HiddenMarkov()
HMM.forward(Q, V, A, B, O, PI)


