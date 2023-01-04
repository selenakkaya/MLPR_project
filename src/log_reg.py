
import numpy as np
import scipy
import scipy.optimize


import arrangeData


mcol = arrangeData.mcol
mrow = arrangeData.mrow 


class LogisticRegression:

    def train_classifier(self, D, L, l, pi, type='linear'): #or type: quadratic
        self.LTR = L
        self.type = type

        if type == 'quadratic':
            DT = feature_exp(D) 
        else: 
            DT = D

        K = L.max() + 1
        M = DT.shape[0]
        obj = logreg_obj_wrapper(DT, L, l, pi)
        x, f, d = scipy.optimize.fmin_l_bfgs_b(
            obj,
            x0=np.zeros(M * K + K),
            approx_grad=True,
        )
        self.w, self.b = x[0:M], x[-1]
        return self


    def compute_scores(self, D):
        DE = feature_exp(D) if self.type == 'quadratic' else D
        postllr = np.dot(self.w, DE) + self.b
        return postllr - np.log(self.LTR[self.LTR == 1].shape[0] / self.LTR[self.LTR == 0].shape[0])


def logreg_obj_wrapper(D, L, l, pi):
    Z = (L * 2) - 1
    M = D.shape[0]

    def log_reg_obj(v):
        w, b = mcol(v[0:M]), v[-1]
        c1 = 0.5 * l * (np.linalg.norm(w) ** 2)
        c2 = ((pi) * (L[L == 1].shape[0] ** -1)) * np.logaddexp(0, -Z[Z == 1] * (np.dot(w.T, D[:, L == 1]) + b)).sum()
        c3 = ((1 - pi) * (L[L == 0].shape[0] ** -1)) * np.logaddexp(0, -Z[Z == -1] * (np.dot(w.T, D[:, L == 0]) + b)).sum()
        return c1 + c2 + c3
    return log_reg_obj


def feature_exp(D):
    expansion = []
    for i in range(D.shape[1]):
        e = np.reshape(np.dot(mcol(D[:, i]), mcol(D[:, i]).T), (-1, 1), order='F')
        expansion.append(e)
    return np.vstack((np.hstack(expansion), D))
