from operator import itemgetter
from scipy.linalg import cho_solve, cholesky, solve_triangular
import numpy as np
from scipy import stats
import scipy as scipy

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Kernel
from sklearn.gaussian_process.kernels import ConstantKernel as C
from sklearn.base import clone
from sklearn.preprocessing._data import _handle_zeros_in_scale
from sklearn.utils import check_random_state

GPR_CHOLESKY_LOWER = True


class GaussianProcessRegressor2dNoise(GaussianProcessRegressor):

    # training function ---> add hyperparameter constraints here
    def fit(self, X, y, priors=True):
        """Fit Gaussian process regression model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features) or list of object
            Feature vectors or other representations of training data.

        y : array-like of shape (n_samples,) or (n_samples, n_targets)
            Target values.
        
        priors: bool
            The choice of using priors on the hyperparameters.
            Default is True.
            
        Returns
        -------
        self : object
            GaussianProcessRegressor class instance.
        """
        
        if self.kernel is None:  # Use an RBF kernel as default
            self.kernel_ = C(1.0, constant_value_bounds="fixed") * RBF(
                1.0, length_scale_bounds="fixed"
            )
        else:
            self.kernel_ = clone(self.kernel)

        self._rng = check_random_state(self.random_state)

        if self.kernel_.requires_vector_input:
            dtype, ensure_2d = "numeric", True
        else:
            dtype, ensure_2d = None, False
        X, y = self._validate_data(
            X,
            y,
            multi_output=True,
            y_numeric=True,
            ensure_2d=ensure_2d,
            dtype=dtype,
        )
        
        # added because it is not passing this for some reason from sklearn
        n_targets = None
        self.n_targets = n_targets

        n_targets_seen = y.shape[1] if y.ndim > 1 else 1
        if self.n_targets is not None and n_targets_seen != self.n_targets:
            raise ValueError(
                "The number of targets seen in `y` is different from the parameter "
                f"`n_targets`. Got {n_targets_seen} != {self.n_targets}."
            )

        # Normalize target value (no using this; doesn't preserve covariances of data set)
        if self.normalize_y:
            self._y_train_mean = np.mean(y, axis=0)
            self._y_train_std = _handle_zeros_in_scale(np.std(y, axis=0), copy=False)

            # Remove mean and make unit variance
            y = (y - self._y_train_mean) / self._y_train_std
            
        else:
            shape_y_stats = (y.shape[1],) if y.ndim == 2 else 1
            self._y_train_mean = np.zeros(shape=shape_y_stats)
            self._y_train_std = np.ones(shape=shape_y_stats)

        if np.iterable(self.alpha) and self.alpha.shape[0] != y.shape[0]:
            if self.alpha.shape[0] == 1:
                self.alpha = self.alpha[0]
            else:
                raise ValueError(
                    "alpha must be a scalar or an array with same number of "
                    f"entries as y. ({self.alpha.shape[0]} != {y.shape[0]})"
                )

        self.X_train_ = np.copy(X) if self.copy_X_train else X
        self.y_train_ = np.copy(y) if self.copy_X_train else y

        # the part I really care about
        if self.optimizer is not None and self.kernel_.n_dims > 0:
            # Choose hyperparameters based on maximizing the log-marginal
            # likelihood (potentially starting from several initial values)
            def obj_func(theta, eval_gradient=True, priors=priors):
                
                if eval_gradient:
                    lml, grad = self.log_marginal_likelihood(
                        theta, eval_gradient=True, clone_kernel=False, prior=priors
                    ) 
                    return -lml, -grad
                
                else:
                    return -self.log_marginal_likelihood(theta, clone_kernel=False)
                
            # First optimize starting from theta specified in kernel
            optima = [
                (
                    self._constrained_optimization(
                        obj_func, self.kernel_.theta, self.kernel_.bounds
                    )
                )
            ]

            # Additional runs are performed from log-uniform chosen initial
            # theta
            if self.n_restarts_optimizer > 0:
                if not np.isfinite(self.kernel_.bounds).all():
                    raise ValueError(
                        "Multiple optimizer restarts (n_restarts_optimizer>0) "
                        "requires that all bounds are finite."
                    )
                bounds = self.kernel_.bounds
                for iteration in range(self.n_restarts_optimizer):
                    theta_initial = self._rng.uniform(bounds[:, 0], bounds[:, 1])
                    optima.append(
                        self._constrained_optimization(obj_func, theta_initial, bounds)
                    )
            # Select result from run with minimal (negative) log-marginal
            # likelihood
            lml_values = list(map(itemgetter(1), optima))
            self.kernel_.theta = optima[np.argmin(lml_values)][0]
            self.kernel_._check_bounds_params()

            self.log_marginal_likelihood_value_ = -np.min(lml_values)
        else:
            self.log_marginal_likelihood_value_ = self.log_marginal_likelihood(
                self.kernel_.theta, clone_kernel=False
            )

        # Precompute quantities required for predictions which are independent
        # of actual query points
        # Alg. 2.1, page 19, line 2 -> L = cholesky(K + sigma^2 I)
        K = self.kernel_(self.X_train_)
        # Handle 2d noise:
        if np.iterable(self.alpha) and self.alpha.ndim == 2:
            K += self.alpha
        else:
            K[np.diag_indices_from(K)] += self.alpha

        try:
            self.L_ = cholesky(K, lower=GPR_CHOLESKY_LOWER, check_finite=False)
        except np.linalg.LinAlgError as exc:
            exc.args = (
                           (
                               f"The kernel, {self.kernel_}, is not returning a positive "
                               "definite matrix. Try gradually increasing the 'alpha' "
                               "parameter of your GaussianProcessRegressor estimator."
                           ),
                       ) + exc.args
            raise
        # Alg 2.1, page 19, line 3 -> alpha = L^T \ (L \ y)
        self.alpha_ = cho_solve(
            (self.L_, GPR_CHOLESKY_LOWER),
            self.y_train_,
            check_finite=False,
        )
        return self

    def log_marginal_likelihood(
            self, theta=None, eval_gradient=False, clone_kernel=True, prior=False
    ):
        """Return log-marginal likelihood of theta for training data.

        Parameters
        ----------
        theta : array-like of shape (n_kernel_params,) default=None
            Kernel hyperparameters for which the log-marginal likelihood is
            evaluated. If None, the precomputed log_marginal_likelihood
            of ``self.kernel_.theta`` is returned.

        eval_gradient : bool, default=False
            If True, the gradient of the log-marginal likelihood with respect
            to the kernel hyperparameters at position theta is returned
            additionally. If True, theta must not be None.

        clone_kernel : bool, default=True
            If True, the kernel attribute is copied. If False, the kernel
            attribute is modified, but may result in a performance improvement.

        Returns
        -------
        log_likelihood : float
            Log-marginal likelihood of theta for training data.

        log_likelihood_gradient : ndarray of shape (n_kernel_params,), optional
            Gradient of the log-marginal likelihood with respect to the kernel
            hyperparameters at position theta.
            Only returned when eval_gradient is True.
        """
        if theta is None:
            if eval_gradient:
                raise ValueError("Gradient can only be evaluated for theta!=None")
            return self.log_marginal_likelihood_value_

        if clone_kernel:
            kernel = self.kernel_.clone_with_theta(theta)
        else:
            kernel = self.kernel_
            kernel.theta = theta

        if eval_gradient:
            K, K_gradient = kernel(self.X_train_, eval_gradient=True)
        else:
            K = kernel(self.X_train_)

        # Alg. 2.1, page 19, line 2 -> L = cholesky(K + sigma^2 I)
        ### Jordan's code below ###
        # Handle 2d noise:
        if np.iterable(self.alpha) and self.alpha.ndim == 2:
            K += self.alpha
        else:
            K[np.diag_indices_from(K)] += self.alpha
        try:
            L = cholesky(K, lower=GPR_CHOLESKY_LOWER, check_finite=False)
        except np.linalg.LinAlgError:
            return (-np.inf, np.zeros_like(theta)) if eval_gradient else -np.inf

        # Support multi-dimensional output of self.y_train_
        y_train = self.y_train_
        if y_train.ndim == 1:
            y_train = y_train[:, np.newaxis]
            
        ### Where gsum implements stuff ### ----> do we need to alter the likelihood? I don't think so...
        
        # Alg 2.1, page 19, line 3 -> alpha = L^T \ (L \ y)
        alpha = cho_solve((L, GPR_CHOLESKY_LOWER), y_train, check_finite=False)

        # Alg 2.1, page 19, line 7
        # -0.5 . y^T . alpha - sum(log(diag(L))) - n_samples / 2 log(2*pi)
        # y is originally thought to be a (1, n_samples) row vector. However,
        # in multioutputs, y is of shape (n_samples, 2) and we need to compute
        # y^T . alpha for each output, independently using einsum. Thus, it
        # is equivalent to:
        # for output_idx in range(n_outputs):
        #     log_likelihood_dims[output_idx] = (
        #         y_train[:, [output_idx]] @ alpha[:, [output_idx]]
        #     )
        log_likelihood_dims = -0.5 * np.einsum("ik,ik->k", y_train, alpha)
        log_likelihood_dims -= np.log(np.diag(L)).sum()
        log_likelihood_dims -= K.shape[0] / 2 * np.log(2 * np.pi)
        # the log likehood is sum-up across the outputs
        log_likelihood = log_likelihood_dims.sum(axis=-1)

        if eval_gradient:
            # Eq. 5.9, p. 114, and footnote 5 in p. 114
            # 0.5 * trace((alpha . alpha^T - K^-1) . K_gradient)
            # alpha is supposed to be a vector of (n_samples,) elements. With
            # multioutputs, alpha is a matrix of size (n_samples, n_outputs).
            # Therefore, we want to construct a matrix of
            # (n_samples, n_samples, n_outputs) equivalent to
            # for output_idx in range(n_outputs):
            #     output_alpha = alpha[:, [output_idx]]
            #     inner_term[..., output_idx] = output_alpha @ output_alpha.T
            inner_term = np.einsum("ik,jk->ijk", alpha, alpha)
            # compute K^-1 of shape (n_samples, n_samples)
            K_inv = cho_solve(
                (L, GPR_CHOLESKY_LOWER), np.eye(K.shape[0]), check_finite=False
            )
            # create a new axis to use broadcasting between inner_term and
            # K_inv
            inner_term -= K_inv[..., np.newaxis]
            # Since we are interested about the trace of
            # inner_term @ K_gradient, we don't explicitly compute the
            # matrix-by-matrix operation and instead use an einsum. Therefore
            # it is equivalent to:
            # for param_idx in range(n_kernel_params):
            #     for output_idx in range(n_output):
            #         log_likehood_gradient_dims[param_idx, output_idx] = (
            #             inner_term[..., output_idx] @
            #             K_gradient[..., param_idx]
            #         )
            log_likelihood_gradient_dims = 0.5 * np.einsum(
                "ijl,jik->kl", inner_term, K_gradient
            )
            # the log likelihood gradient is the sum-up across the outputs
            log_likelihood_gradient = log_likelihood_gradient_dims.sum(axis=-1)
            
        ### ---- adding the log prior in a proxy lml expression ---- ###
        if prior is True:
            log_prior_value_ls = self.log_prior_ls(theta)
            log_prior_value_sig = self.log_prior_sig(theta)
            log_total = log_likelihood + log_prior_value_ls + log_prior_value_sig
            log_total_gradient = log_likelihood_gradient + \
            self.log_prior_ls_gradient(theta) + self.log_prior_sig_gradient(theta)
        else:
            log_total = log_likelihood
            log_total_gradient = log_likelihood_gradient

        if eval_gradient:
            return log_total, log_total_gradient
        else:
            return log_likelihood
        
    # define the prior for the lengthscale (truncated normal)
    def log_prior_ls(self, theta, *args):
        
        # take in lengthscale only for this prior
        ls = np.exp(theta[1])
        a = np.exp(self.kernel_.bounds[1,0])
        b = np.exp(self.kernel_.bounds[1,1])
                
        # log uniform prior, bounded
        def luniform_ls(ls, a, b):
            if ls > a and ls < b:
                return 0.0
            else:
                return -np.inf
        
       # return luniform_ls(ls, a, b) + stats.norm.logpdf(ls, 0.8, 0.1) # 20n0
            
        return luniform_ls(ls, a, b) + stats.norm.logpdf(ls, 1.05, 0.1)  # 40n0
    
    
    def log_prior_ls_gradient(self, theta, *args):
        
        # take in lengthscale only for this prior
        ls = np.exp(theta[1])
        
        return - 2.0 / (ls - 1.05)
        
       # return - 2.0 / (ls - 0.8)   #40n0 => 1.05)
    
    
    # define the prior for the lengthscale (truncated normal)
    def log_prior_sig(self, theta, *args):
        
        # take in lengthscale only for this prior
        sig = np.exp(theta[0])
        a = np.exp(self.kernel_.bounds[0,0])
        b = np.exp(self.kernel_.bounds[0,1])
                
        # log uniform prior, bounded
        def luniform_sig(sig, a, b):
            if sig > a and sig < b:
                return 0.0
            else:
                return -np.inf
            
        return luniform_sig(sig, a, b) + stats.norm.logpdf(sig, 1.0, 0.25) 
    
    
    def log_prior_sig_gradient(self, theta, *args):
        
        # take in lengthscale only for this prior
        sig = np.exp(theta[0])
        
        return -2.0 / (sig - 1.0)
    
    
    def _constrained_optimization(self, obj_func, initial_theta, bounds):
        if self.optimizer == "fmin_l_bfgs_b":
            opt_res = scipy.optimize.minimize(
                obj_func,
                initial_theta,
                method="L-BFGS-B",
                jac=True,
                bounds=bounds,
                tol=1e-12,
            )
            self._check_optimize_result("lbfgs", opt_res)
            theta_opt, func_min = opt_res.x, opt_res.fun
        elif callable(self.optimizer):
            theta_opt, func_min = self.optimizer(obj_func, initial_theta, bounds=bounds)
        else:
            raise ValueError(f"Unknown optimizer {self.optimizer}.")

        return theta_opt, func_min
    
    
    def _check_optimize_result(self, solver, result, max_iter=None, extra_warning_msg=None):
        """Check the OptimizeResult for successful convergence

        Parameters
        ----------
        solver : str
           Solver name. Currently only `lbfgs` is supported.

        result : OptimizeResult
           Result of the scipy.optimize.minimize function.

        max_iter : int, default=None
           Expected maximum number of iterations.

        extra_warning_msg : str, default=None
            Extra warning message.

        Returns
        -------
        n_iter : int
           Number of iterations.
        """
        # handle both scipy and scikit-learn solver names
        if solver == "lbfgs":
            if result.status != 0:
                try:
                    # The message is already decoded in scipy>=1.6.0
                    result_message = result.message.decode("latin1")
                except AttributeError:
                    result_message = result.message
                warning_msg = (
                    "{} failed to converge (status={}):\n{}.\n\n"
                    "Increase the number of iterations (max_iter) "
                    "or scale the data as shown in:\n"
                    "    https://scikit-learn.org/stable/modules/"
                    "preprocessing.html"
                ).format(solver, result.status, result_message)
                if extra_warning_msg is not None:
                    warning_msg += "\n" + extra_warning_msg
                warnings.warn(warning_msg, ConvergenceWarning, stacklevel=2)
            if max_iter is not None:
                # In scipy <= 1.0.0, nit may exceed maxiter for lbfgs.
                # See https://github.com/scipy/scipy/issues/7854
                n_iter_i = min(result.nit, max_iter)
            else:
                n_iter_i = result.nit
        else:
            raise NotImplementedError

        return n_iter_i
