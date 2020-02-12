import numpy as np
from sklearn.utils.validation import check_array
from scipy import linalg

class krige_dace:

    def __init__(self, train_x, train_y):
        self.x = np.atleast_2d(train_x)
        self.y = np.atleast_2d(train_y)
        self.n_var = self.x.shape[1]

        self.mS = np.mean(self.x, axis=0)
        self.sS = np.std(self.x, axis=0, ddof=1)
        self.mY = np.mean(self.y, axis=0)
        self.sY = np.std(self.y, axis=0, ddof=1)

        self.x_norm = None
        self.y_norm = None

        self.paramters = {}

        # Check arguments needed
        # Check if the arguments are exactly right

    def poly0(self, x):
        # REGPOLY0 Zero order polynomial regression function
        # Call: f = regpoly0(S)
        # [f, df] = regpoly0(S)
        #
        # S: m * n matrix with design sites
        # f: ones(m, 1)
        # df: Jacobian at the first point(first row in S)
        m, n = x.shape
        f = np.atleast_2d(np.ones((m, 1)))
        df = np.atleast_2d(np.zeros((n, 1)))
        return f, df

    def train(self):
        self.x = check_array(self.x)
        self.y = check_array(self.y)

        m1, nx = self.x.shape
        m2, _ = self.y.shape

        if m1 != m2:
            raise ValueError(
                "X and Y must have the same number of rows"
            )

        min_x = np.amin(self.x, axis=0)
        max_x = np.amax(self.x, axis=0)

        lb = []
        for i in range(nx):
            lb.append(max(1e-6, min_x[i]))
        lb = np.atleast_2d(lb).T
        ub = np.atleast_2d(max_x).T
        theta0 = np.ones((1, nx))

        # just for this experiment
        lb = []
        ub = []
        for i in range(nx):
            lb.append(0.001)
            ub.append(1000)
        lb = np.atleast_2d(lb).T
        ub = np.atleast_2d(ub).T


        self.fit(self.poly0, self.corr_gau, theta0, lb, ub)

    def corr_gau(self, theta, d):
        m, n = d.shape
        theta = np.atleast_2d(theta).reshape(-1, 1)
        if len(theta) == 1:
            theta = (np.ones(n) * theta).reshape(-1, 1)
        elif len(theta) != n:
            raise ValueError(
                "Length of theta must be 1 or num of x variables"
            )

        td = d ** 2 * (-1 * theta.T)

        r = np.exp(td.sum(axis=1))
        r = np.atleast_2d(r).reshape(-1, 1)

        dr = -2 * theta.T * d * r

        return r, dr

    def fit(self, callback_poly, callback_corr, theta0, lob=None, upb=None):
        m, n = self.x.shape
        sy, _ = self.y.shape


        if sy != m:
            raise ValueError(
                "X and Y must have the same number of rows"
            )

        # determine whether optimization is needed
        lth = len(theta0.ravel())
        if lob is not None:
            if len(lob) != lth or len(upb) != lth:
                raise ValueError(
                    "theta0, lob and upb must have the same length"
                )
            if np.any(lob) < 0 or np.any(lob) > np.any(upb):
                raise ValueError(
                    "The bounds must satisfy  0 < lob <= upb"
                )
        else:
            if np.any(theta0) <= 0:
                raise ValueError(
                    "The bounds must satisfy  0 < lob <= upb"
                )

        # Check for 'missing dimension'
        self.sS[self.sS == 0.] = 1.
        self.sY[self.sY == 0.] = 1.

        self.x_norm = (self.x - self.mS) / self.sS
        self.y_norm = (self.y - self.mY) / self.sY

        # Calculate distances D between points
        mzmax = int(m * (m - 1) / 2)
        ij = np.zeros((mzmax, 2))
        D = np.zeros((mzmax, n))
        ll_1 = 0
        for k in range(m - 1):
            ll_0 = ll_1
            ll_1 = ll_0 + m - k - 1
            ij[ll_0:ll_1, 0] = k
            ij[ll_0:ll_1, 1] = np.arange(k + 1, m)
            D[ll_0:ll_1] = self.x_norm[k] - self.x_norm[(k + 1): m]

        ij = ij.astype(np.int)

        # check duplicated sites
        if np.min(np.sum(D, axis=1)) == 0.0:
            print("Multiple input features cannot have the same value.")
            print("Filter duplicated rows")
            # fixed in case
            unique_x, unique_index = np.unique(self.x_norm, axis=0, return_index=True)
            self.x_norm = self.x_norm[unique_x, :]
            self.y_norm = self.y_norm[unique_x, :]

        #go for regression matrix
        F, df = callback_poly(self.x_norm)
        mF, p = F.shape
        if mF != m:
            # means unique filter has been conducted
            m, n = self.x_norm.shape
            mzmax = int(m * (m - 1) / 2)
            ij = np.zeros((mzmax, 2))
            D = np.zeros((mzmax, n))
            ll_1 = 0
            for k in range(m - 1):
                ll_0 = ll_1
                ll_1 = ll_0 + m - k - 1
                ij[ll_0:ll_1, 0] = k
                ij[ll_0:ll_1, 1] = np.arange(k + 1, m)
                D[ll_0:ll_1] = self.x_norm[k] - self.x_norm[(k + 1): m]

            ij = ij.astype(np.int)

        if m != n:
            if p > mF:
                raise Exception("least squares problem is underdetermined")

        self.paramters['corr'] = callback_corr
        self.paramters['regr'] = callback_poly
        self.paramters['y'] = self.y_norm
        self.paramters['F'] = F
        self.paramters['D'] = D
        self.paramters['ij'] = ij
        self.paramters['scS'] = self.sS

        if lob is not None:
            theta, f, fit, perf = self.boxmin(theta0, lob, upb, self.paramters)
            if f == np.inf or f == -np.inf:
                print('Bad parameter region.  Try increasing  upb')
        else:
            raise ValueError(
                "make your train method set lob and upb! I am lazy to implement this line"
            )

        self.paramters['corr'] = callback_corr
        self.paramters['regr'] = callback_poly
        self.paramters['theta'] = theta.T
        self.paramters['beta'] = fit['beta']
        self.paramters['gamma'] = fit['gamma'] # gamma should be one row
        self.paramters['sigma2'] = self.sY ** 2 * fit['sigma2']
        self.paramters['S'] = self.x_norm
        self.paramters['Ssc'] = np.vstack((self.mS, self.sS))
        self.paramters['Ysc'] = np.vstack((self.mY, self.sY))
        self.paramters['C'] = fit['C']
        self.paramters['Ft'] = fit['Ft']
        self.paramters['G'] = fit['G']

    def objfunc(self, theta, par):

        # initialization
        obj = np.inf
        fit = {'sigma2': None, 'beta': None, 'gamma': None, 'C': None,  'Ft': None, 'G': None}
        m = par['F'].shape[0]
        corr_func = par['corr']
        r, _ = corr_func(theta, par['D'])
        if r.shape[1] != 1:
            raise ValueError(
                "output of corr_func should be one column"
            )
        index = np.where(r.ravel() > 0)
        # index is tuple type
        index = index[0]

        o = np.arange(0, m)

        MACHINE_EPSILON = np.finfo(np.double).eps;
        mu = (10 + m) * MACHINE_EPSILON

        R = np.eye(m) * (1.0 + mu)
        ij = par['ij']
        R[ij[:, 0], ij[:, 1]] = r[:, 0]
        R[ij[:, 1], ij[:, 0]] = r[:, 0]

        try:
            C = linalg.cholesky(R, lower=True)
        except (linalg.LinAlgError, ValueError) as e:
            print("exception : ", e)
            raise

        F = par['F']
        Ft = linalg.solve_triangular(C, F, lower=True)
        Q, G = linalg.qr(Ft, mode="economic")

        sv = linalg.svd(G, compute_uv=False)
        rcondG = sv[-1] / sv[0]
        if rcondG < 1e-10:
            # Check F
            sv = linalg.svd(self.F, compute_uv=False)
            condF = sv[0] / sv[-1]
            if condF > 1e15:
                raise Exception(
                    "F is too ill conditioned. Poor combination "
                    "of regression model and observations."
                )

            else:
                print("Ft is too ill conditioned, get out (try different theta)")
                return
        y = par['y']
        Yt = linalg.solve_triangular(C, y, lower=True)
        beta = linalg.solve_triangular(G, np.dot(Q.T, Yt))

        rho = Yt - np.dot(Ft, beta)
        sigma2 = np.sum(rho**2)/m

        gamma = linalg.solve_triangular(C.T, rho)
        gamma = gamma.T

        # The determinant of R is equal to the squared product of the diagonal
        # elements of its Cholesky decomposition C
        detR = (np.diag(C) ** (2.0 / m)).prod()
        obj = np.sum(sigma2) * detR

        fit = {'sigma2': sigma2, 'beta': beta, 'C': C, 'Ft': Ft, 'G': G, 'gamma': gamma}
        return obj, fit

    def boxmin(self, t0, lo, up, par):
        t, f, fit, itpar = self.start(t0, lo, up, par)

        if f != np.inf and f != -np.inf:
            # this line is to make sure len() works
            t = np.atleast_2d(t).reshape(-1, 1)
            p = len(t)
            if p <= 2:
                kmax = 2
            else:
                kmax = min(p, 4)

            for k in range(kmax):
                th = t.copy()
                t, f, fit, itpar = self.explore(t, f, fit, itpar, par)
                t, f, fit, itpar = self.move(th, t, f, fit, itpar, par)

        perf = {'nv': itpar['nv'], 'perf': itpar['perf'][:, 0: itpar['nv']+1]}

        return t, f, fit, perf


    def start(self, t0, lo, up, par):

        t = np.atleast_2d(t0).reshape(-1, 1)
        lo = np.atleast_2d(lo).reshape(-1, 1)
        up = np.atleast_2d(up).reshape(-1, 1)
        p = len(t)
        D = 2 ** ((np.atleast_2d(range(1, p+1)))/(p+2))
        D = np.atleast_2d(D).reshape(-1, 1)

        diff = np.atleast_2d(lo-up)
        index = np.where(diff == 0)
        if len(index[0]) > 0:
            # this process needs validation
            ins = len(index[0])
            print('in start, up and lo equals')
            D[index[0], index[1]] = np.atleast_2d(np.ones(ins))
            t[index[0], index[1]] = up[index[0], index[1]]

        # free starting point
        # lo_vio is of tuple type
        lo_vio = np.where((t - lo).ravel() < 0)
        up_vio = np.where((t - up).ravel() > 0)
        if len(lo_vio[0]) > 0:
            # t/lo/up are 2d arrays
            t[lo_vio[0], :] = (lo[lo_vio[0], :] * up[lo_vio[0], :]**7)**(1/8)
        if len(up_vio[0]) > 0:
            # t/lo/up are 2d arrays
            t[up_vio[0], :] = (lo[up_vio[0], :] * up[up_vio[0], :] ** 7) ** (1 / 8)

        # use ravel(), because D is only one column, although 2d
        ne = np.where(D.ravel() != 1)
        # cause where() returns tuple type
        ne = ne[0]

        # Check starting point and initialize performance info
        f, fit = self.objfunc(t, par)
        nv = 0

        itpar = {'D': D, 'ne': ne, 'lo': lo, 'up': up, 'perf': np.atleast_2d(np.zeros((p+2, 200*p))), 'nv': 1}
        itpar['perf'][:, 0] = np.vstack((t, f, 1)).ravel()

        if f == np.inf or f == -np.inf:
            print('Bad parameter region')
            return

        if len(lo_vio[0]) + len(up_vio[0]) > 1:
            #print('bad starting point, restart boxmin')
            vio = np.hstack((lo_vio[0], up_vio[0]))
            d0 = 16
            d1 = 2
            q = len(vio)
            th = t
            fh = f
            jdom = vio[0]
            for k in range(q):
                j = vio[k]
                fk = fh
                tk = th
                DD = np.ones((p, 1))
                DD[vio] = np.ones((q, 1)) * (1 / d1)

                DD[j] = 1 / d0

                alpha = min(np.log(lo[vio] / th[vio]) / np.log(DD[vio])) / 5
                v = DD ** alpha
                tk = th

                for rept in range(4):
                    tt = tk * v
                    ff,  fitt = self.objfunc(tt, par)
                    nv = nv + 1
                    itpar['perf'][:, nv] = np.vstack((tt, ff, 1)).ravel()
                    if ff <= fk:
                        tk = tt
                        fk = ff
                        if ff <= f:
                            t = tt
                            f = ff
                            fit = fitt
                            jdom = j
                    else:
                        itpar['perf'][-1, nv] = -1
                        break

            # Update Delta
            if jdom > 1:
                D[[0, jdom], :] = D[[jdom, 1], :]
                itpar['D'] = D
        itpar['nv'] = nv

        return t, f, fit, itpar


    def explore(self, t, f, fit, itpar, par):
        nv = itpar['nv']
        ne = itpar['ne']

        # be aware of use of len
        for k in range(len(ne)):
            j = ne[k]
            tt = t.copy()
            DD = itpar['D'][j]
            if t[j] == itpar['up'][j]:
                atbd = True
                tt[j] = t[j]/np.sqrt(DD)
            elif t[j] == itpar['lo'][j]:
                atbd = True
                tt[j] = t[j] * np.sqrt(DD)
            else:
                atbd = False
                tt[j] = min(itpar['up'][j], tt[j]*DD)

            # recalculate objective
            ff, fitt = self.objfunc(tt, par)
            nv = nv + 1
            itpar['perf'][:, nv] = np.vstack((tt, ff, 2)).ravel()

            if ff < f:
                t = tt
                f = ff
                fit = fitt
            else:
                itpar['perf'][-1, nv] = -2
                if not atbd:
                    tt[j, 0] = max(itpar['lo'][j], t[j]/DD)
                    ff, fitt = self.objfunc(tt, par)
                    nv = nv + 1
                    itpar['perf'][:, nv] = np.vstack((tt, ff, 2)).ravel()
                    if ff < f:
                        t = tt
                        f = ff
                        fit = fitt
                    else:
                        itpar['perf'][-1, nv] = -2

        itpar['nv'] = nv

        return t, f, fit, itpar

    def move(self, th, t, f, fit, itpar, par):
        nv = itpar['nv']
        ne = itpar['ne']
        [t_m, t_n] = t.shape
        if t_n != 1:
            raise ValueError(
                "t (name of theta vector) should be 1 column"
            )
        p = len(t)
        v = t / th
        if sum(v) == p:
            itpar['D'] = itpar['D']**0.2
            top = itpar['D'][0].copy()
            itpar['D'] = np.delete(itpar['D'], 0, axis=0)
            itpar['D'] = np.vstack((itpar['D'], top))
            return t, f, fit, itpar

        # proper move
        rept = True
        while rept:
            c1 = itpar['up']
            c2 = np.hstack((itpar['lo'], t * v))
            c2 = np.atleast_2d(np.max(c2, axis=1)).reshape(-1, 1)
            c3 = np.hstack((c1, c2))
            tt = np.atleast_2d(np.min(c3, axis=1)).reshape(-1, 1)

            ff, fitt = self.objfunc(tt, par)
            nv = nv + 1
            itpar['perf'][:, nv] = np.vstack((tt, ff, 3)).ravel()
            if ff < f:
                t = tt
                f = ff
                fit = fitt
                v = v ** 2
            else:
                itpar['perf'][-1, nv] = -3
                rept = False

            c1 = np.where(np.equal(tt.ravel(), itpar['up'].ravel()))
            c2 = np.where(np.equal(tt.ravel(), itpar['lo'].ravel()))
            if len(c1[0]) > 0 or len(c2[0]) > 0:
                rept = False
        itpar['nv'] = nv
        itpar['D'] = itpar['D'] ** 0.25
        top = itpar['D'][0].copy()
        itpar['D'] = np.delete(itpar['D'], 0, axis=0)
        itpar['D'] = np.vstack((itpar['D'], top))

        return t, f, fit, itpar

    def predictor(self, x):
        or1 = None
        or2 = None
        dmse = None

        if len(self.paramters) ==0:
            raise ValueError(
                "model has not been trained"
            )

        m, n = self.paramters['S'].shape
        sx = x.shape
        if min(sx) == 1 and n > 1:
            nx = max(sx)
            if nx == n:
                mx = 1
                x  = x
        else:
            mx = sx[0]
            nx = sx[1]

        if nx != n:
            raise ValueError(
                "Dimension of trial sites is not compatible with training data"
            )

        # Normalize trial sites
        x = (x - self.paramters['Ssc'][0, :])/self.paramters['Ssc'][1, :]
        # number of response functions
        q = self.paramters['Ysc'].shape[1]
        y = np.zeros((mx, q))

        # one site only
        if mx == 1:
            # distances to design sites
            dx = x - self.paramters['S']

            f, df = self.paramters['regr'](x)
            r, dr = self.paramters['corr'](self.paramters['theta'], dx)

            # Scaled Jacobian
            dy = (df.dot(self.paramters['beta'])).T + self.paramters['gamma'].dot(dr)

            # Unscaled Jacobian
            or1 = dy * self.paramters['Ysc'][1, :].T / self.paramters['Ssc'][1, :]

            if q == 1:
                or1 = or1.T

            # MSE wanted
            # C \ r
            rt = linalg.solve_triangular(self.paramters['C'], r, lower=True)
            # this u value starts differ from matlab value
            u = self.paramters['Ft'].T.dot(rt) - f.T
            v = u / self.paramters['G']
            or2 = self.paramters['sigma2'] * (1 + v**2 - np.sum(rt**2).T)


            # gradient/Jacobian of MSE not implemented

            # Scaled predictor
            sy = f.dot(self.paramters['beta']) + self.paramters['gamma'].dot(r)
            # Predictor
            y = self.paramters['Ysc'][0, :] + self.paramters['Ysc'][1, :] * sy
        else:
            # several trial sites
            # Get distances to design sites
            dx = np.zeros((mx * m, n))
            kk = np.arange(m)
            for k in range(mx):
                dx[kk, :] = x[k] - self.paramters['S']
                kk = kk + m

            f, _ = self.paramters['regr'](x)
            r, _ = self.paramters['corr'](self.paramters['theta'], dx)
            r = np.atleast_2d(r).reshape(m, mx, order='F')

            # Scaled predictor
            sy = f.dot(self.paramters['beta']) + self.paramters['gamma'].dot(r).T
            # Predictor
            y = self.paramters['Ysc'][0, :] + self.paramters['Ysc'][1, :] * sy

            rt = linalg.solve_triangular(self.paramters['C'], r, lower=True)
            # u's exact value is different from matlab, but on same scale
            u = linalg.solve_triangular(self.paramters['G'], (self.paramters['Ft'].T.dot(rt) - f.T), lower=True)
            or1 = self.paramters['sigma2'] * (1 + self.colsum(u**2) - self.colsum((rt**2))).T
            or2 = None
        # print("so far so good")
        return y, or1, or2

    def colsum(self, x):
        if x.shape[0] == 1:
            s = x
        else:
            s = np.sum(x, axis=0)
        return s

    def predict(self, X):

        X = check_array(X)
        samples, _ = X.shape

        if samples > 1:
            f, ssqr, _ = self.predictor(X)
        else:
            f, _, ssqr = self.predictor(X)

        if f.shape[0] == X.shape[0]:
            y = f
        else:
            y = f.T

        mse = np.sqrt(np.abs(ssqr))
        y = np.atleast_2d(y)
        mse = np.atleast_2d(mse)
        return y, mse





