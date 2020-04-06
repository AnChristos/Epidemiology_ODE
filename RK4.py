#  Author : Christos Anastopoulos
#  Purpose :
#  Class to integrate  systems of ordinary differential equations
#  of the form :
#  dY(i)/dX = F(T,Y(0),Y(1), ...  ,Y(DIM-1)) (1)
#  using the 4th order explicit Runge-Kutta method
#  Usage :
#  Input:
#  - Function(T,X) to evaluate the right hand side of  equations (1);
#  - initialPoint: Value of X where the initialValues for Y[i] are specified
#  - initialValues: Initial values for  Y(i) at X=InitialPoint
#  - stepSize: Size h for each step in X for the Runge-Kutta method
#  Advance Solution:
#  - integrateStep()
#  Advances the solution by a step h in X.
#  It updates the X and Y(I) internally.
#  Current state of the solution:
#  -currentPoint()
#  Returns the current X
#  -currentValues()
#  Returns the current Y(I)


import numpy as np


class RK4():
    """Class to integrate  systems of ordinary differential equations
    of the form : y_i'=f_i(x,y_i(x))
    using the 4th order Runge-Kutta method
    """

    def __init__(
            self,
            function=None,
            stepSize=None,
            initialPoint=None,
            initialValues=None):
        """Create an instance of the RK4 class
        Arguments
        - function:
        Accept 2 arguments : current value of x and
        a numpy array with the current values of the y_i(x).
        Return a numpy array containing the values of the f_i
        ==>f_i (x_i,y(x_i)), for this x.
        - stepSize: Size for each step in the Runge-Kutta method
        - initialPoint: Value of x where the initialValues for y_i are specified
        - initialValues: Initial values for y_i at x=InitialPoint
        """
        if(not hasattr(function, '__call__')):
            raise TypeError(('Error provided function ',
                             function, 'seems not to be a function'))

        if(len(initialValues) != len(function(initialPoint, initialValues))):
            raise ValueError(
                ('The size of the initialValues array is not the same as the one returned by function'))

        # user input
        self._h = stepSize
        self._fcn = function
        self._x = initialPoint
        self._y = initialValues

    def setStepSize(self, stepSize):
        """set the stepSize"""
        self._h = stepSize
        return

    def setFunction(self, function):
        """set the function"""
        if(not hasattr(function, '__call__')):
            raise TypeError(('Error provided function ', function,
                             'seems not to be a function'))
        self._fcn = function
        return

    def setInitialPoint(self, initialPoint):
        """set the initial point """
        self._x = initialPoint
        return

    def setInitialValues(self, initialValues):
        """set the initial values in the solution"""
        self._y = initialValues
        return

    def stepSize(self):
        """Return the result of the function"""
        return self._h

    def fcn(self, x, values):
        """Return the result of the function"""
        return self._fcn(x, values)

    def currentPoint(self):
        """Return the current point in the solution"""
        return self._x

    def currentValues(self):
        """Return the current values in the solution"""
        return self._y

    def integrateStep(self):
        """The main method where the RK4 is implemented
           needs to be called as many times as steps needed
        """
        self._k1 = self._h * self.fcn(self._x, self._y)
        self._k2 = self._h * \
            self.fcn(self._x + 0.5 * self._h, self._y + 0.5 * self._k1)
        self._k3 = self._h * \
            self.fcn(self._x + 0.5 * self._h, self._y + 0.5 * self._k2)
        self._k4 = self._h * self.fcn(self._x + self._h, self._y + self._k3)
        self._y = self._y + (self._k1 + 2 * self._k2 +
                             2 * self._k3 + self._k4) / 6.0
        self._x = self._x + self._h

