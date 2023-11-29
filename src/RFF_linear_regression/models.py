import torch

class LinearRegression:
    """Standard linear regression in fully determined case, selects min-norm solution
    in the underdetermined case."""
    def _init_(self):
        pass
    def fit(self, X, y):
        # fully determined case (standard least squares)
        if X.shape[0] >= X.shape[1]:
            self.ytobeta = torch.linalg.solve(X.t() @ X, X.t())
            self.beta = self.ytobeta @ y
        # underdetermined case with min norm solution (see: https://see.stanford.edu/materials/lsoeldsee263/08-min-norm.pdf)
        elif X.shape[0] < X.shape[1]:
            # self.ytobeta = X.t() @ torch.inverse(X @ X.t())
            # self.ytobeta = torch.linalg.solve(X @ X.t(), X).t()
            self.ytobeta = torch.linalg.pinv(X)  # this is the most numerically stable method
            self.beta = self.ytobeta @ y
        self.X = X; self.y = y  # store data

    def eff_p_l2(self, X_eval):
        S = X_eval @ self.ytobeta
        return torch.mean(torch.linalg.vector_norm(S, dim=1))
    
    def eff_p_l2_squared(self, X_eval):
        S = X_eval @ self.ytobeta
        return torch.mean(torch.linalg.vector_norm(S, dim=1)**2)

    def predict(self, X):
        return X @ self.beta

    def weight_norm(self):
        return torch.linalg.vector_norm(self.beta, ord=2)


class PCA:
    def __init__(self):
        self.f_means = None
        self.f_stds = None

    def fit(self, X, n_components):
        X_stand = self._standardize(X)
        # E, V = torch.linalg.eigh(X_stand.t() @ X_stand)
        # key = torch.argsort(E).flip(0)[:n_components]
        # T = X_stand @ V[:, :n_components]
        # self.V = V[:, key]

        # also the most numerically stable method (PCA can have numerical issues on 
        # near square matrices)
        _, _, self.V = torch.pca_lowrank(X_stand, q=n_components, center=False, niter=10)

    def transform(self, X):
        X_stand = self._standardize(X)
        return X_stand @ self.V

    def fit_transform(self, X, n_components):
        self.fit(X, n_components)
        return self.transform(X)

    def _standardize(self, X):
        if self.f_means is None:
            self.f_means = torch.mean(X, 0)
        if self.f_stds is None:
            self.f_stds = torch.std(X, 0)
        X = X - self.f_means
        X = X / self.f_stds
        return X
    
def apply_PCA(X_train, X_test, n_components):
    """Wrapper function that applies the PCA class"""
    pca = PCA()
    X_train = pca.fit_transform(X_train, n_components)
    X_test = pca.transform(X_test)
    return X_train, X_test