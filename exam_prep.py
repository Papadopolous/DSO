#%%
from math import floor, ceil
from IPython.display import display, Math
from numpy.lib.function_base import vectorize
import scipy.optimize as opt
import numpy as np
import numpy.linalg as linalg
import pandas as pd
from matplotlib import pyplot as plt
import sympy as sym
from sympy.core.sympify import sympify
from sympy.plotting.plot import plot
from sympy.utilities.lambdify import lambdify
# from sympy.core.symbol import Symbol

#%%
#Helper Functions
def plot_contour(function, range_x, range_y):
    X, Y = np.meshgrid(range_x, range_y)
    Z = function(X, Y)
    plt.contourf(X, Y, Z, levels=20)
    plt.colorbar()

#########################################################################    
#%%
class GoldenSection2():
    def __init__(self, f, a, b, iters):
        self.function =  sym.lambdify("x", sym.sympify(f))
        
        self.iter_count = 0
        self._data_array = np.zeros((iters, 6))
        
        self.invphi = (np.sqrt(5) - 1) / 2  # 1 / phi
        self.invphi2 = (3 - np.sqrt(5)) / 2  # 1 / phi^2
        
        final_bracket = self.gssrec(self.function, a, b, iters=iters)
        
        self.data = pd.DataFrame(self._data_array, columns=['a', 'b', 'c', 'd', 'f(c)', 'f(d)'])
        display(self.data)
        
    def gssrec(self, f, a, b, tol=1e-10, h=None, c=None, d=None, fc=None, fd=None, iters=5):
        """ Golden-section search, recursive.

        Given a function f with a single local minimum in
        the interval [a,b], gss returns a subset interval
        [c,d] that contains the minimum with d-c <= tol.

        Example:
        >>> f = lambda x: (x-2)**2
        >>> a = 1
        >>> b = 5
        >>> tol = 1e-5
        >>> (c,d) = gssrec(f, a, b, tol)
        >>> print (c, d)
        1.9999959837979107 2.0000050911830893
        """
        (a, b) = (min(a, b), max(a, b))
        if h is None: h = b - a
        if h <= tol:
            print("Tolerance met... Exiting")
            return (a, b)
        if c is None: c = a + self.invphi2 * h
        if d is None: d = a + self.invphi * h
        if fc is None: fc = f(c)
        if fd is None: fd = f(d)
        
        self._data_array[self.iter_count, :] = a, b, c, d, fc, fd
        self.iter_count +=1
        
        if self.iter_count >= iters: return(a, b)
        
        if fc < fd:
            # print("fc < fd")
            return self.gssrec(f, a, d, tol, h * self.invphi, c=None, fc=None, d=c, fd=fc, iters=iters)
        else:
            # print("fc > fd")
            return self.gssrec(f, c, b, tol, h * self.invphi, c=d, fc=fd, d=None, fd=None, iters=iters)
#%%
class GoldenSection():
    def __init__(self, function, x1, x2, iters):
        self.function = sym.sympify(function)
        self.lambda_func = sym.lambdify("x", self.function)
        
        self._data_np = np.zeros((iters+2, 2))
        
        self._data_np[0, :] = x1, self.lambda_func(x1)
        self._data_np[1, :] = x2, self.lambda_func(x2)
        
        self._method(iters)
        
        x_min, x_max = min(self._data_np[:, 0]), max(self._data_np[:, 0])
        plotting_range = np.linspace(0.9*x_min, 1.1*x_max, 100)
        plt.plot(plotting_range, self.lambda_func(plotting_range), label='f(x)')
        plt.plot(self._data_np[:, 0], self._data_np[:, 1], 'x-', label='Golden Section')
        plt.legend()
        plt.show()
        
        display(self.data)
        
    
    def _step(self, x1, x2):
        golden_ratio = 2.0/(1.0+np.sqrt(5.0))
        
        xa = x1 + (1-golden_ratio) * (x2 - x1)
        fxa = self.lambda_func(xa)
        
        xb = x2 + (1-golden_ratio) * (x2 - x1)
        fxb = self.lambda_func(xb)
        
        if fxa < fxb:
            return xa, fxa
        else:
            return xb, fxb
        
        
        return next_x, next_fx
    
    def _method(self, iters):
        for i in range(iters):
            self._data_np[i+2, :] = self._step(self._data_np[i, 0], self._data_np[i+1, 0])

        self.data = pd.DataFrame(self._data_np, columns=['x', 'f(x)'], index = range(1, iters+3))

#%%
class LinearNewton():
    """Uses Newton Algorithm to find the turning points of a 1D function

    Args:
        function (string or sympy equation): Expression for the objective value in terms of x, x_1 or x_2
        start_x (float): initial guess of location of turning point
        iters (int): number of iterations to carry out
    """
    def __init__(self, function, start_x, iters):
        if type(function) is not str:
            self.function = str(function)
        else:
            self.function = function
        self.function = sym.sympify(self.function.replace("x_1","x").replace("x_2", "x"))
        self.dydx = sym.diff(self.function, "x")
        self.dydx2 = sym.diff(self.dydx, "x")
        
        self.x = start_x
        self.iters = iters

        self.__data_np = np.empty((5, iters))
        
        self.iter_num = 0

        self.run()
    
    def __step(self):
        fx = self.function.evalf(subs={"x" : self.x})
        dydx_eval = self.dydx.evalf(subs = {"x" : self.x})
        dydx2_eval = self.dydx2.evalf(subs = {"x" : self.x})
        x_next = self.x - dydx_eval/dydx2_eval
        
        self.__data_np[:, self.iter_num] = [self.x, fx, dydx_eval, dydx2_eval, x_next]
        
        self.x = x_next
        
    def run(self):
        for i in range(self.iters):
            self.__step()
            self.iter_num += 1
        
        x_min, x_max = min(self.__data_np[0, :]), max(self.__data_np[0, :])
        plotting_range = np.linspace(0.9*x_min, 1.1*x_max, 100)
        plotting_func = sym.lambdify('x', self.function)
        plt.plot(plotting_range, plotting_func(plotting_range), label='function')
        plt.plot(self.__data_np[0, :], self.__data_np[1, :], label='Newton')
        plt.legend()
        plt.show()
        self.data = pd.DataFrame(self.__data_np.T, columns=['x', 'f(x)', 'dy/dx', 'dy/dx^2', 'x_{i+1}'])
        
        return self.data

# %%
class SteepestDescent():
    """Utilises the steepest graident algorithm to find the minima of a function.

    Args:
        function (str): A string of the objective function. Must be written in terms of x_1 and x_2
        start_x (dict): A dictionary of the starting values of "x_1" and "x_2"
        iters (int): The number of iterations to be carried out
    
    Returns:
        data (DataFrame): DataFrame containing iteration information
    
    Example:
        Input
            SteepestGradient(function =   "(2*x_1) - (x_2) + (x_1*x_2) + (x_1**2) + (x_2**2)",
                            start_x  =  {"x_1" : -0.39,
                                        "x_2" : -0.84},
                            iters    =    5)
        Returns:
            DataFrame
                x_1	x_2	f(x_1, x_2)	df/dx1	df/dx2	lambda	x_1-lambda*df/dx1	x_2-lambda*df/dx2	f(x-lambda.Nabla(f(x)))
            1	-0.390000	-0.840000	1.245300	0.380000	-3.070000	0.569418	-0.606379	0.908114	-1.479167
            2	-0.606379	0.908114	-1.479167	1.695356	0.209849	0.445668	-1.361945	0.814591	-2.129456
            3	-1.361945	0.814591	-2.129456	0.090700	-0.732763	0.569418	-1.413592	1.231840	-2.284671
            4	-1.413592	1.231840	-2.284671	0.404656	0.050088	0.445668	-1.593934	1.209517	-2.321718
            5	-1.593934	1.209517	-2.321718	0.021649	-0.174900	0.569418	-1.606261	1.309108	-2.330561
    """
    def __init__(self, function, start_x, iters):
        # ns={}
        # ns.update(dict(x_1=sym.Symbol("x_1")))
        # ns.update(dict(x_2=sym.Symbol("x_2")))
        self.function = sym.sympify(function)
        self.x = start_x
        self.iters = iters
        self.fx = self.function.evalf(subs=self.x)
        self.partial_x1 = sym.diff(self.function, "x_1")
        self.partial_x2 = sym.diff(self.function, "x_2")

        
        self.run()
        
    def _step(self):
        dfdx1 = self.partial_x1.evalf(subs=self.x)
        dfdx2 = self.partial_x2.evalf(subs=self.x)
        
        lambda_function = sym.lambdify(["x_1, x_2"], self.function)
        def opt_function(learning_rate):
            x_1 = self.x["x_1"] - (learning_rate * dfdx1)
            x_2 = self.x["x_2"] - (learning_rate * dfdx2)
            value = float(self.function.evalf(subs={"x_1" : x_1,
                                             "x_2" : x_2}))
            return value

        learning_rate_opt = opt.minimize_scalar(opt_function)
        learning_rate = learning_rate_opt.x
        new_x = {}
        new_x["x_1"] = self.x["x_1"] - (learning_rate * dfdx1)
        new_x["x_2"] = self.x["x_2"] - (learning_rate * dfdx2)
        new_fx = learning_rate_opt.fun
        
        self._data_np[:, self.iter_num] =   [self.x["x_1"], self.x["x_2"], self.fx, dfdx1, dfdx2,
                                            learning_rate, new_x["x_1"], new_x["x_2"],
                                            self.function.evalf(subs=new_x)]
        self.x = new_x
        self.fx = new_fx
    
    def run(self):
        for i in [self.function, self.partial_x1, self.partial_x2]:
            display(i)

        self.iter_num = 0
        self._data_np = np.empty((9, self.iters))
        
        for i in range(self.iters):
            self._step()
            self.iter_num += 1
        self.data = pd.DataFrame(self._data_np.T,
                              index = range(1, self.iters+1),
                              columns = ['x_1', 'x_2',
                                        'f(x_1, x_2)',
                                        'df/dx1', 'df/dx2',
                                        'lambda', 'x_1-lambda*df/dx1', 'x_2-lambda*df/dx2', 'f(x-lambda.Nabla(f(x)))'])
        
        max_x_1 = max(self._data_np[0, :])
        min_x_1 = min(self._data_np[0, :])
        x_1_range = max_x_1 - min_x_1
        max_x_2 = max(self._data_np[1, :])
        min_x_2 = min(self._data_np[1, :])
        x_2_range = max_x_2 - min_x_2
        plot_range_x1 = np.linspace(min_x_1 - 0.1*x_1_range, max_x_1 + 0.1*x_1_range, 50)
        plot_range_x2 = np.linspace(min_x_2 - 0.1*x_2_range, max_x_2 + 0.1*x_2_range, 50)
        plot_contour(sym.lambdify(["x_1, x_2"], self.function),
                     plot_range_x1, plot_range_x2)
        plt.plot(self._data_np[0, :], self._data_np[1, :], '-x')
        plt.show()
        display(self.data)
        
        return self.data
    
#########################################################################
# %%
class ConjugateGradient():
    def __init__(self, function, start_x, iters):
        """Utilises the Conjugate Gradient algorithm to find the minima of a function.

        Args:
            function (str): A string of the objective function. Must be written in terms of x_1 and x_2
            start_x (dict): A dictionary of the starting values of "x_1" and "x_2"
            iters (int): The number of iterations to be carried out
        
        Returns:
            data (DataFrame): DataFrame containing iteration information
        
        Example:
            Input
                ConjugateGradient(function =   "x_1 - x_2 + 1.8*x_1*x_2 + 2*x_1**2 + x_2**2",
                                  start_x  =  {"x_1" : 0,
                                               "x_2" : 0},
                                iters      =    5)
            Returns:
                DataFrame
                	x_1	x_2	f(x_1, x_2)	df/dx1	df/dx2	|df/dx|^2	S2_1	S2_2	lambda	x_1-lambda*df/dx1	x_2-lambda*df/dx2	f(x-lambda.Nabla(f(x)))
                1	0.000000	0.000000	0.000000	1.000000e+00	-1.000000e+00	2.000000e+00	1.000000e+00	-1.000000e+00	0.833333	-0.833333	0.833333	-0.833333
                2	-0.833333	0.833333	-0.833333	-8.333333e-01	-8.333333e-01	1.388889e+00	1.388889e-01	1.527778e+00	-0.252101	-0.798319	1.218487	-1.008403
                3	-0.798319	1.218487	-1.008403	-2.473278e-08	-2.473278e-08	1.223421e-15	2.473278e-08	2.473278e-08	-0.190983	-0.798319	1.218487	-1.008403
                4	-0.798319	1.218487	-1.008403	2.663755e-09	-6.783325e-09	5.310909e-17	-1.590097e-09	7.856983e-09	0.618034	-0.798319	1.218487	-1.008403
                5	-0.798319	1.218487	-1.008403	-2.145897e-09	-1.472617e-08	2.214649e-16	-8.961964e-09	4.301264e-08	-0.190983	-0.798319	1.218487	-1.008403
        """
        # ns={}
        # ns.update(dict(x_1=sym.Symbol("x_1")))
        # ns.update(dict(x_2=sym.Symbol("x_2")))
        self.function = sym.sympify(function)
        self.x = start_x
        self.iters = iters
        
        self.fx = self.function.evalf(subs=self.x)
        self.partial_x1 = sym.diff(self.function, "x_1")
        self.partial_x2 = sym.diff(self.function, "x_2")
        
        for i in [self.function, self.partial_x1, self.partial_x2]:
            display(i)
        
        self.iter_num = 0
        self._data_np = np.empty((12, iters))
        self.run()
    def _step(self):
        dfdx1 = self.partial_x1.evalf(subs=self.x)
        dfdx2 = self.partial_x2.evalf(subs=self.x)
        
        dfdx_squared = dfdx1**2 + dfdx2**2
        
        if self.iter_num != 0:
            old_dfdx1, old_dfdx2, old_dfdx_squared = self._data_np[3:6, self.iter_num-1]
            S2_1 = -dfdx1 + (dfdx_squared/old_dfdx_squared)*(-old_dfdx1)
            S2_2 = -dfdx2 + (dfdx_squared/old_dfdx_squared)*(-old_dfdx2)
        else:
            S2_1 = dfdx1
            S2_2 = dfdx2
        
        def opt_function(learning_rate):
            x_1 = self.x["x_1"] - (learning_rate * S2_1)
            x_2 = self.x["x_2"] - (learning_rate * S2_2)
            value = float(self.function.evalf(subs={"x_1" : x_1,
                                            "x_2" : x_2}))
            return value

        learning_rate_opt = opt.minimize_scalar(opt_function)
        learning_rate = learning_rate_opt.x
        new_x = {}
        new_x["x_1"] = self.x["x_1"] - (learning_rate * S2_1)
        new_x["x_2"] = self.x["x_2"] - (learning_rate * S2_2)
        new_fx = learning_rate_opt.fun
        
        self._data_np[:, self.iter_num] =   [self.x["x_1"], self.x["x_2"], self.fx, dfdx1, dfdx2,
                                            dfdx_squared, S2_1, S2_2,
                                            learning_rate, new_x["x_1"], new_x["x_2"],
                                            self.function.evalf(subs=new_x)]
        self.x = new_x
        self.fx = new_fx

    def run(self):
        for i in range(self.iters):
            self._step()
            self.iter_num += 1
        self.data = pd.DataFrame(self._data_np.T,
                              index = range(1, self.iters+1),
                              columns = ['x_1', 'x_2', 'f(x_1, x_2)','df/dx1', 'df/dx2',
                                         '|df/dx|^2', 'S2_1', 'S2_2',
                                        'lambda', 'x_1-lambda*df/dx1', 'x_2-lambda*df/dx2', 'f(x-lambda.Nabla(f(x)))'])
        
        max_x_1 = max(self._data_np[0, :])
        min_x_1 = min(self._data_np[0, :])
        x_1_range = max_x_1 - min_x_1
        max_x_2 = max(self._data_np[1, :])
        min_x_2 = min(self._data_np[1, :])
        x_2_range = max_x_2 - min_x_2
        plot_range_x1 = np.linspace(min_x_1 - 0.1*x_1_range, max_x_1 + 0.1*x_1_range, 50)
        plot_range_x2 = np.linspace(min_x_2 - 0.1*x_2_range, max_x_2 + 0.1*x_2_range, 50)
        plot_contour(sym.lambdify(["x_1, x_2"], self.function),
                     plot_range_x1, plot_range_x2)
        plt.plot(self._data_np[0, :], self._data_np[1, :], '-x')
        plt.show()
        display(self.data)
        
        return self.data
#########################################################################
# %%
class Newton():
    """Utilises the Newton algorithm to find the minima of a function.

    Args:
        function (str): A string of the objective function. Must be written in terms of x_1 and x_2
        start_x (dict): A dictionary of the starting values of "x_1" and "x_2"
        iters (int): The number of iterations to be carried out
    
    Returns:
        data (DataFrame): DataFrame containing iteration information
    
    Example:
        Input
            Newton(function =   "x_1 - x_2 + 1.8*x_1*x_2 + 2*x_1**2 + x_2**2",
                    start_x  =  {"x_1" : 0,
                                "x_2" : 0},
                    iters      =    1)
        Returns:
            DataFrame
                x_1	x_2	f(x_1, x_2)	df/dx1	df/dx2	df/dx1x1	df/dx1x2	df/dx2x2	determinant	x_1-lambda*df/dx1	x_2-lambda*df/dx2	f(x-lambda.Nabla(f(x)))
            1	0.0	0.0	0.0	1.0	-1.0	4.0	2.0	2.0	4.0	-1.0	1.5	-1.25
    """
    def __init__(self, function, start_x, iters):
        # ns={}
        # ns.update(dict(x_1=sym.Symbol("x_1")))
        # ns.update(dict(x_2=sym.Symbol("x_2")))
        self.function = sym.sympify(function)
        self.x = start_x
        self.iters = iters
        
        self.fx = self.function.evalf(subs=self.x)
        self.partial_x1 = sym.diff(self.function, "x_1")
        self.partial_x2 = sym.diff(self.function, "x_2")
        self.partial_x1x1 = sym.diff(self.partial_x1, "x_1")
        self.partial_x1x2 = sym.diff(self.partial_x1, "x_2")
        self.partial_x2x2 = sym.diff(self.partial_x2, "x_2")
        
        for i in [self.function, self.partial_x1, self.partial_x2, self.partial_x1x1, self.partial_x1x2, self.partial_x2x2]:
            display(i)
        
        self.iter_num = 0
        self._data_np = np.empty((12, iters))
        self.run()
    
    def _step(self):
        dfdx1 = self.partial_x1.evalf(subs=self.x)
        dfdx2 = self.partial_x2.evalf(subs=self.x)
        
        dfdx1x1 = self.partial_x1x1.evalf(subs=self.x)
        dfdx1x2 = self.partial_x1x2.evalf(subs=self.x)
        dfdx2x2 = self.partial_x2x2.evalf(subs=self.x)
        
        determinant = dfdx1x1*dfdx2x2 - (dfdx1x2**2)
        
        new_x = {}
        new_x["x_1"] = self.x["x_1"] - ( (1/determinant) * ( (dfdx2x2 * dfdx1) + (-dfdx1x2 * dfdx2) ) )
        new_x["x_2"] = self.x["x_2"] - ( (1/determinant) * ( (dfdx1x1 * dfdx2) + (-dfdx1x2 * dfdx1) ) )
        new_fx = self.function.evalf(subs=new_x)
        
        self._data_np[:, self.iter_num] = [self.x["x_1"], self.x["x_2"], self.fx, dfdx1, dfdx2,
                                            dfdx1x1, dfdx1x2, dfdx2x2, determinant,
                                            new_x["x_1"], new_x["x_2"],
                                            new_fx]
        self.x = new_x
        
    def run(self):
        for i in range(self.iters):
            self._step()
            self.iter_num += 1
        self.data = pd.DataFrame(self._data_np.T,
                              index = range(1, self.iters+1),
                              columns = ['x_1', 'x_2', 'f(x_1, x_2)','df/dx1', 'df/dx2',
                                         'df/dx1x1', 'df/dx1x2', 'df/dx2x2', 'determinant',
                                         'x_1-lambda*df/dx1', 'x_2-lambda*df/dx2', 'f(x-lambda.Nabla(f(x)))'])
        
        max_x_1 = max(self._data_np[0, :])
        min_x_1 = min(self._data_np[0, :])
        x_1_range = max_x_1 - min_x_1
        max_x_2 = max(self._data_np[1, :])
        min_x_2 = min(self._data_np[1, :])
        x_2_range = max_x_2 - min_x_2
        plot_range_x1 = np.linspace(min_x_1 - 0.1*x_1_range, max_x_1 + 0.1*x_1_range, 50)
        plot_range_x2 = np.linspace(min_x_2 - 0.1*x_2_range, max_x_2 + 0.1*x_2_range, 50)
        plot_contour(sym.lambdify(["x_1, x_2"], self.function),
                     plot_range_x1, plot_range_x2)
        plt.plot(self._data_np[0, :], self._data_np[1, :], '-x')
        plt.show()
        display(self.data)
        
        return self.data
#########################################################################
# %%
class HookeAndJeeves():
    """Utilises the Hooke and Jeeves algorithm to find the minima of a function.

    Args:
        function (str): A string of the objective function. Must be written in terms of x_1 and x_2
        start_x (dict): A dictionary of the starting values of "x_1" and "x_2"
        iters (int): The number of iterations to be carried out
        initial_step (float): [default = 0.1] the initial step length tried during exploration
    
    Returns:
        data (DataFrame): DataFrame containing iteration information
    
    Example:
        Input
            HookeAndJeeves(("x_1 - x_2 + 2*x_1*(x_2**2) + x_2**3",
                {"x_1" : 0.0,
                "x_2" : 1.0},
                iters = 5)
        Returns:
            DataFrame
            x2	x3	delta x2	delta x3	f(x)	delta f(x)	step length	pattern delta x2	pattern delta x3
            0	0	1			0.e-125		0.1		
            Explore right 1	0.1	1	0.1	0	0.300000000000000	0.300000000000000	0.1		
            Explore left 1	-0.1	1	-0.1	0	-0.300000000000000	-0.300000000000000	0.1		
            Explore up 1	-0.1	1.1	-0.1	0.1	-0.111000000000000	0.189000000000000	0.1		
            Explore down 1	-0.1	0.9	-0.1	-0.1	-0.433000000000000	-0.133000000000000	0.1		
            Basepoint 1	-0.1	0.9			-0.433000000000000				
            Pattern Move 1								-0.1	-0.1
            Basepoint after Pattern Move 1	-0.2	0.8			-0.744000000000000	
    """
    def __init__(self, function, start_x, iters, initial_step = 0.1):
        # ns={}
        # ns.update(dict(x_1=sym.Symbol("x_1")))
        # ns.update(dict(x_2=sym.Symbol("x_2")))
        
        self.function = sym.sympify(function)
        self.basepoint = start_x
        self.iters = iters
        self.initial_step = initial_step
    
        self.fx = self.function.evalf(subs=self.basepoint)
        self.pattern_vector = [0, 0]
        self.iter_num = 0
        
        self.exploration_points = []
        self.pattern_moves = np.empty((2, self.iters))
        
        display(self.function)
        
        self.data = pd.DataFrame(np.array([[self.basepoint["x_1"], self.basepoint["x_2"], '', '', self.fx, '', self.initial_step, '', '']]),
                                 columns = ['x_1', 'x_2', 'delta x_1', 'delta x_2', 'f(x)', 'delta f(x)', 'step length', 'pattern delta x_1', 'pattern delta x_2'])
        
        self.run()
        
    def __explore(self):
        stepsize = self.initial_step
        improvement = False
        while not improvement:
            east_x = {"x_1" : self.basepoint["x_1"] + stepsize,
                      "x_2" : self.basepoint["x_2"]}
            west_x = {"x_1" : self.basepoint["x_1"] - stepsize,
                      "x_2" : self.basepoint["x_2"]}
            east_delta = self.fx - self.function.evalf(subs=east_x)
            west_delta = self.fx - self.function.evalf(subs=west_x)
            if east_delta > west_delta and east_delta > 0:
                EW_movement = stepsize
                best_fx = self.function.evalf(subs=east_x)
            elif east_delta < west_delta and west_delta > 0:
                EW_movement = -stepsize
                best_fx = self.function.evalf(subs=west_x)
            else:
                EW_movement = 0
                best_fx = self.fx
            
            north_x = {"x_1" : self.basepoint["x_1"] + EW_movement,
                      "x_2" : self.basepoint["x_2"] + stepsize}
            south_x = {"x_1" : self.basepoint["x_1"] + EW_movement,
                      "x_2" : self.basepoint["x_2"] - stepsize}
            north_delta = best_fx - self.function.evalf(subs=north_x)
            south_delta = best_fx - self.function.evalf(subs=south_x)
            
            if north_delta > south_delta and north_delta > 0:
                NS_movement = stepsize
            elif north_delta < south_delta and south_delta > 0:
                NS_movement = -stepsize
            else:
                NS_movement = 0
            
            
            if EW_movement == 0 and NS_movement == 0:
                stepsize *= 0.5
                if stepsize < 10**-6:
                    break
                continue
            else:
                explore_choice = [EW_movement, NS_movement]

                self.exploration_points.append(([self.basepoint["x_1"], self.basepoint["x_2"]],
                                                [east_x["x_1"], east_x["x_2"]],
                                                [west_x["x_1"], west_x["x_2"]],
                                                [north_x["x_1"], north_x["x_2"]],
                                                [south_x["x_1"], south_x["x_2"]]))
                exploration_data = pd.DataFrame(
                                   np.array([[east_x["x_1"],    east_x["x_2"],      stepsize,       0,          self.function.evalf(subs=east_x), -east_delta,     stepsize, '', ''],
                                             [west_x["x_1"],    west_x["x_2"],      -stepsize,      0,          self.function.evalf(subs=west_x), -west_delta,     stepsize, '', ''],
                                             [north_x["x_1"],   north_x["x_2"],     EW_movement,    stepsize,   self.function.evalf(subs=north_x), -north_delta,    stepsize, '', ''],
                                             [south_x["x_1"],   south_x["x_2"],     EW_movement,    -stepsize,  self.function.evalf(subs=south_x), -south_delta,    stepsize, '', '']]),
                                   index = ['Explore right {}'.format(self.iter_num+1), 'Explore left {}'.format(self.iter_num+1), 'Explore up {}'.format(self.iter_num+1), 'Explore down {}'.format(self.iter_num+1)],
                                   columns = ['x_1', 'x_2', 'delta x_1', 'delta x_2', 'f(x)', 'delta f(x)', 'step length', 'pattern delta x_1', 'pattern delta x_2']
                )
                self.data = pd.concat([self.data, exploration_data])
                return explore_choice

    
    def __pattern_move(self):
        keep_pattern = True
        while keep_pattern:
            while self.iter_num < self.iters:
                explore_choice = self.__explore()
                self.basepoint = {"x_1" : self.basepoint["x_1"] + explore_choice[0],
                                "x_2" : self.basepoint["x_2"] + explore_choice[1]}
                self.fx = self.function.evalf(subs = self.basepoint)
                
                
                self.pattern_vector = [explore_choice[i] + self.pattern_vector[i] for i in range(2)]

                basepoint_data = pd.DataFrame(
                    np.array([[self.basepoint["x_1"],    self.basepoint["x_2"], '', '', self.fx, '', '', '', ''],
                              ['', '', '', '', '', '', '', self.pattern_vector[0], self.pattern_vector[1]]]),
                    index = ['Basepoint {}'.format(self.iter_num+1), 'Pattern Move {}'.format(self.iter_num+1)],
                    columns = ['x_1', 'x_2', 'delta x_1', 'delta x_2', 'f(x)', 'delta f(x)', 'step length', 'pattern delta x_1', 'pattern delta x_2']
                )
                
                self.data = pd.concat([self.data, basepoint_data])
                
                pattern_x = {}
                pattern_x["x_1"] = self.basepoint["x_1"] + self.pattern_vector[0]
                pattern_x["x_2"] = self.basepoint["x_2"] + self.pattern_vector[1]
                pattern_fx = self.function.evalf(subs = pattern_x)

                if pattern_fx < self.fx:
                    keep_pattern = True
                    plt.plot([self.basepoint["x_1"], pattern_x["x_1"]], [self.basepoint["x_2"], pattern_x["x_2"]])
                    self.basepoint = pattern_x
                    self.fx = pattern_fx

                else:
                    keep_pattern = False
                    self.pattern_vector = [0, 0]
                
                pattern_move_data = pd.DataFrame(
                    np.array([[self.basepoint["x_1"],    self.basepoint["x_2"], '', '', self.fx, '', '', '', '']]),
                    index = ['Basepoint after Pattern Move {}'.format(self.iter_num+1)],
                    columns = ['x_1', 'x_2', 'delta x_1', 'delta x_2', 'f(x)', 'delta f(x)', 'step length', 'pattern delta x_1', 'pattern delta x_2']
                )
                
                self.data = pd.concat([self.data, pattern_move_data])
                
                self.iter_num += 1
            break

    
    def run(self):
        plot_contour(sym.lambdify(["x_1, x_2"], self.function),
                     np.linspace(-2, 2, 50), np.linspace(-2, 2, 50))
        
        for i in range(self.iters):
            self.__pattern_move()
            self.iter_num += 1
        
        for i in range(len(self.exploration_points)):
            base, east, west, north, south = self.exploration_points[i]
            xs = base[0], east[0], west[0], north[0], south[0]
            ys = base[1], east[1], west[1], north[1], south[1]
            plt.plot(xs, ys, '+', label=i+1)
            plt.legend()
        plt.show()
        display(self.data)
        return self.data    

# %%
class GeneticAlgorithm():
    """Carries out a GA child-creation step, with one generation and a population of 2

    Args:
        parents (list): List of two parent values
        bounds (list): List of lower and upper bounds on the values of the parents
        random_numbers (list): List of 3 random numbers
        count_direction (str, optional): Whether to cound the cross over point from left to right 'LR' or right to left 'RL'. Defaults to 'LR'.
    """
    def __init__(self, parents, bounds, random_numbers, count_direction = 'LR'):
        self.parents = parents
        self.bounds = bounds
        self.random_numbers = random_numbers
        self.count_direction = count_direction
        
        self.method()
        display(self.data)
        
    def __int_to_binary(self, num):
        return  '{0:06b}'.format(num)
    
    def __binary_to_int(self, bin_string):
        return int(bin_string, 2)
    
    def __flip_bit(self, string, bit_number):
        string = list(string)
        if string[bit_number] == '0':
            string[bit_number] = '1'
        else:
            string[bit_number] = '0'
        return ''.join(string)
    
    def method(self):
        parent1, parent2 = self.parents
        lower_bound, upper_bound = self.bounds
        
        scaled_parent1 = floor(63* (parent1 - lower_bound)/(upper_bound - lower_bound))
        scaled_parent2 = floor(63* (parent2 - lower_bound)/(upper_bound - lower_bound))
        
        self.binary_parent1 = self.__int_to_binary(scaled_parent1)
        self.binary_parent2 = self.__int_to_binary(scaled_parent2)
        
        if type(self.random_numbers) is list:
            self.cross_over_point = floor(self.random_numbers[0] / (1/5) ) + 1
            self.mutation_1 = floor(self.random_numbers[1] / (1/6) )
            self.mutation_2 = floor(self.random_numbers[2] / (1/6) )
        else:
            self.cross_over_point = floor(self.random_numbers / (1/5) ) + 1
            self.mutation_1 = ''
            self.mutation_2 = ''
        
        if self.count_direction == 'RL':
            self.cross_over_point == 6 - self.cross_over_point
            if type(self.random_numbers) is list:
                self.mutation_1 = 6 - self.mutation_1
                self.mutation_2 = 6 - self.mutation_2
            
        self.child1 = self.binary_parent2[:self.cross_over_point] + self.binary_parent1[self.cross_over_point:]
        self.child2 = self.binary_parent1[:self.cross_over_point] + self.binary_parent2[self.cross_over_point:]
           
        if type(self.random_numbers) is list:
            self.child1_mutated = self.__flip_bit(self.child1, self.mutation_1)
            self.child2_mutated = self.__flip_bit(self.child2, self.mutation_2)
            
            self.scaled_result1 = self.__binary_to_int(self.child1_mutated)
            self.scaled_result2 = self.__binary_to_int(self.child2_mutated)
        
        else:
            self.child1_mutated = ''
            self.child2_mutated = ''
            
            self.scaled_result1 = self.__binary_to_int(self.child1)
            self.scaled_result2 = self.__binary_to_int(self.child2)
        
        self.results1 = lower_bound + (self.scaled_result1/63) * (upper_bound - lower_bound)
        self.results2 = lower_bound + (self.scaled_result2/63) * (upper_bound - lower_bound)
        
        self.results = [self.results1, self.results2]

        self.data = pd.DataFrame(np.array([[parent1, parent2],
                                           [scaled_parent1, scaled_parent2],
                                           [self.binary_parent1, self.binary_parent2],
                                           [self.cross_over_point, ''],
                                           [self.child1, self.child2],
                                           [self.mutation_1, self.mutation_2],
                                           [self.child1_mutated, self.child2_mutated],
                                           [self.scaled_result1, self.scaled_result2],
                                           [self.results1, self.results2]]),
                                 index=['Parents', 'Scaled Parents', 'Binary Parents', 'Cross-over Point',
                                        'Children', 'Mutation Points', 'Mutated Children', 'Scaled Results', 'Results'],
                                 columns=['1', '2'])
        return self.data
        
        
# %%
# TODO FIX THIS SO THAT IT ACTUALLY WORKS - IT'S WRONG RN
class ConstraintElimination():
    """Takes an expression and an equation and substitutes the equation into the expression

    Args:
        f2 (str): Expression for the objective function
        f3 (str): Equation for one of the variables in the objective function
    """
    def __init__(self, f2, f3):
        self.f2 = sym.sympify(f2)
        self.f3 = f3
        
        self.method()
        
    def method(self):
        variable_to_sub, f2_expr = self.f3.replace(' ', '').split('=')
        f2_expr = sym.sympify(f2_expr)
        
        self.f = self.f2.subs(variable_to_sub, f2_expr)
        
        display(self.f2)
        display(sym.Eq(sym.sympify(variable_to_sub), f2_expr))
        display(self.f)
        
        return self.f
    
    def LinearNewton(self, start_x, iters=5):
        linear_optimisation = LinearNewton(self.f, start_x, iters)
        self.data = linear_optimisation.data
        
        display(self.data)
        
    def Newton(self, start_x, iters=5):
        newton = Newton(self.f, start_x-start_x, iters=iters)
# %%
class ParetoFront():
    """Aids in finding the Pareto front for a dataset or two 1-D function

    Args:
        data (list, optional): [list(xvalues), list(yvalues)]. Defaults to None.
        f1 (str, optional): Expression for f2. Defaults to None.
        f2 (str, optional): Expression for f3. Defaults to None.
    """
    def __init__ (self, data=None, f1=None, f2=None):
        self.func1_inflection = None
        self.func2_inflection = None
        self.func1_minima = None
        self.func2_minima = None
        
        if f1 is not None and f2 is not None:
            self.func1 = sym.sympify(f1)
            self.func2 = sym.sympify(f2)
            self.functions()
            
        elif data is not None:
            self.data = data
            self.database()
        else:
            print("Unexpected Input")
    
    def functions(self):
        x_range = np.linspace(-2, 2, 50)
        lambda_func1 = sym.lambdify("x", self.func1)
        lambda_func2 = sym.lambdify("x", self.func2)
        
        func1_evals = lambda_func1(x_range)
        func2_evals = lambda_func2(x_range)
        
        plt.plot(x_range, func1_evals, label=r'$f_1$')
        plt.plot(x_range, func2_evals, label=r'$f_2$')
        plt.xlabel(r'$x$')
        plt.ylabel('Value')
        plt.legend()
        plt.show()
        
        plt.plot(func1_evals, func2_evals)
        plt.xlabel(r'$f_1$')
        plt.ylabel(r'$f_2$')
        plt.show()
        
        df1dx = sym.diff(self.func1, "x")
        df2dx = sym.diff(self.func2, "x")
        display(Math(r"\frac{df_1}{dx} = "), df1dx)
        display(Math(r"\frac{df_2}{dx} =  "), df2dx)
        
        self.func1_inflection = sym.solve(df1dx, "x", domain=sym.S.Reals)
        self.func2_inflection = sym.solve(df2dx, "x", domain=sym.S.Reals)        
        
        self.func1_data = [self.func1, df1dx]
        self.func2_data = [self.func2, df2dx]
        
        for point in self.func1_inflection:
            value = self.func1.evalf(subs={"x" : point})
            self.func1_data.append("({}, {})".format(point, value))
        
        for point in self.func2_inflection:
            value = self.func2.evalf(subs={"x" : point})
            self.func2_data.append("({}, {})".format(point, value))
        
        if len(self.func1_data) < len(self.func2_data):
            for i in range(len(self.func2_data) - len(self.func1_data)):
                self.func1_data.append('')
        elif len(self.func1_data) > len(self.func2_data):
            for i in range(len(self.func1_data) - len(self.func2_data)):
                self.func2_data.append('')
        
        function_data = np.array([self.func1_data, self.func2_data])
        
        display(pd.DataFrame(function_data)) #.style.set_properties(**{'white-space' : 'pre-wrap',})
        
    def database(self):
        xs, ys = self.data
        fig, ax = plt.subplots()
        ax.scatter(xs, ys)
        for i in range(len(xs)):
            ax.annotate(i+1, (xs[i]*1.02, ys[i]*1.02))
        fig.show()

    
#%%
class ClCdOptimisation():
    def __init__(self, Cl_func, Cd_func):
        self.Cl_func = sym.sympify(Cl_func)
        self.Cd_func = sym.sympify(Cd_func)
        
        self.ClCd = self.Cl_func/self.Cd_func
        self.ClCd = self.ClCd.subs("C_L", self.Cl_func)
        
        display(self.Cl_func)
        display(self.Cd_func)
        display(self.ClCd)
        
        self.data = LinearNewton(self.ClCd.subs('a', 'x'), 0, 10).data
        display(self.data)
# %%
class EigenVectorWeighting():
    """Generates normalised relationship weightings based on the relationship between three values

    Args:
        Pab (float): How much more important b is than a
        Pbc (float): How much more important c is than b
        Pac (float, optional): How much more important c is than a. If None, will be calculated as Pab * Pbc. Defaults to None.
    """
    def __init__(self, Pab, Pbc, Pac=None):
        if Pac == None:
            Pac = Pab * Pbc
        self.relationship_matrix = np.array([[1,     Pab,    Pac],
                                             [1/Pab, 1,      Pbc],
                                             [1/Pac, 1/Pbc,  1]])
        self.eigenValues, self.eigenVectors = linalg.eig(self.relationship_matrix)
        idx = self.eigenValues.argsort()[::-1]   
        self.eigenValues = self.eigenValues[idx]
        self.eigenVectors = self.eigenVectors[:,idx]
        
        # print(self.relationship_matrix)
        # print(self.eigenValues)
        # print(self.eigenVectors)
        
        self.normalised_weightings = np.empty((3, 3))
        for i, vectors in enumerate(zip(self.eigenVectors[0], self.eigenVectors[1], self.eigenVectors[2])):
            vector_sum = np.sum(vectors)
            self.normalised_weightings[:, i] = vectors/vector_sum
        
        # print(self.normalised_weightings)
        
        self.data = pd.DataFrame(np.vstack([self.relationship_matrix, self.eigenValues, self.eigenVectors, self.normalised_weightings]),
                                 index=['Relationship Weighting a', 'b', 'c', 'Eigen Values',
                                        'Eigen Vectors', '', '', 'Normalised Weightings a', 'b', 'c'])
        display(self.data) 
        
            
        
    
#%%
class InverseParabolic():
    def __init__(self, function, x1, x2, x3, iters):
        self.function = sym.sympify(function)

        display(self.function)
        
        self.x1 = x1
        self.x2 = x2
        self.x3 = x3
 
        self.iter_num = 0
        self.iters = iters
 
        self._data_array = np.empty((self.iters, 8))
        
        self.run()
        
        self.data = pd.DataFrame(self._data_array, columns=['x_1', 'x_2', 'x_3', 'f(x_1)', 'f(x_2)', 'f(x_3)', 'x*', 'f(x*)'])
        display(self.data)
 
    def _step(self):
 
        x1 = self.x1
        x2 = self.x2
        x3 = self.x3
 
        f1 = self.function.subs("x", x1).evalf()
        f2 = self.function.subs("x", x2).evalf()
        f3 = self.function.subs("x", x3).evalf()
 
 
        upper = (f3-f2)*(x2**2 - x1**2) + (f1-f2)*(x3**2 - x2**2) 
        lower = 2*((f3-f2)*(x2 - x1) + (f1-f2)*(x3 - x2))
        x_star = upper / lower
        f_xstar = self.function.subs("x", x_star).evalf()
        self._data_array[self.iter_num, :] = [x1, x2, x3, f1, f2, f3, x_star, f_xstar]
 
        self.x1 = floor(x_star)
        self.x2 = x_star
        self.x3 = ceil(x_star)
 
        f1 = self.function.subs("x", self.x1).evalf()
        f2 = self.function.subs("x", self.x2).evalf()
        f3 = self.function.subs("x", self.x3).evalf()

 
 
    def run(self):
        for i in range(self.iters):
            self._step()
            self.iter_num += 1
# %%
class FuzzyLogic():
    def __init__(self, obj1, obj2, membership1, membership2):
        self.obj1 = sym.sympify(obj1)
        self.obj2 = sym.sympify(obj2)
        display(self.obj1)
        display(self.obj2)
        self.obj1 = sym.lambdify("x", self.obj1)
        self.obj2 = sym.lambdify("x", self.obj2)
        
        self.membership1 = self._calc_membership_func(membership1["unacceptable"], membership1["acceptable"])
        self.membership2 = self._calc_membership_func(membership2["unacceptable"], membership2["acceptable"])
        
        self._data_array = np.empty((100, 6))
        
        self.run()
        
        
        self.data = pd.DataFrame(self._data_array, columns=['x', 'f_1(x)', 'f_2(x)', 'Membership 1', 'Membership 2', 'Combined'])
        
        fig, ax1 = plt.subplots()

        ax1.set_xlabel('x')
        ax1.set_ylabel(r'$f_1(x), f_2(x)$')
        ax1.plot(self.data['x'], self.data['f_1(x)'], label=r'$f_1(x)$')
        ax1.plot(self.data['x'], self.data['f_2(x)'], label=r'$f_2(x)$')
        ax1.tick_params(axis='y')
        ax1.legend()
        ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

        ax2.set_ylabel('Memberships')  # we already handled the x-label with ax1
        ax2.plot(self.data['x'], self.data['Membership 1'], 'r', label='Membership 1')
        ax2.plot(self.data['x'], self.data['Membership 2'], 'b', label='Membership 2')
        ax2.plot(self.data['x'], self.data['Combined'], 'g', label='Combined')
        ax2.tick_params(axis='y')

        ax2.legend()
        fig.tight_layout()  # otherwise the right y-label is slightly clipped
        plt.show()
        
        # display(self.data)
        display(self.data.iloc[self.data['Combined'].idxmax()])
        
    def _calc_membership_func(self, unacceptable, acceptable, unacceptable_value = 0, acceptable_value = 1):
        m = (unacceptable_value - acceptable_value) / (unacceptable - acceptable)
        c = acceptable_value - acceptable * m
        def membership_function(x):
            if x > unacceptable:
                return unacceptable_value
            elif x < acceptable:
                return acceptable_value 
            else:
                return m * x + c
        return membership_function
    
    def run(self, range=np.linspace(0.1, 3, 100)):
        for i, x in enumerate(range):
            f1 = self.obj1(x)
            f2 = self.obj2(x)
            mem1 = self.membership1(f1)
            mem2 = self.membership2(f2)
            combined = mem1 + mem2
            self._data_array[i, :] = x, f1, f2, mem1, mem2, combined
            
        
# %%
