import numpy as np
import matplotlib.pyplot as plt

def gradient_descent(f, guess, learningRate, maxIter, threshold=0.00000001):
    # f([x]) => y
    # f([x, z]) => y
    # 1 input. 1 output  = [1]              y = x^2
    # 2 inputs, 1 output = [1.5,6.7]        y = z^23 * x/45
    
    dimensions = len(guess)
    evalSize = learningRate
    iterations = maxIter

    for i in range(maxIter):
        gradient = np.zeros_like(guess)

        # Compute partial derivatives
        for j in range(dimensions):
            
            gradPos = np.copy(guess)
            gradNeg = np.copy(guess)

            gradPos[j] = guess[j] + evalSize
            gradNeg[j] = guess[j] - evalSize
            
            gradient[j] = ( f( gradPos) - f(gradNeg) ) / evalSize / 2 * learningRate

        # Update guess (-gradient)
        lastGuess = np.copy(guess)
        guess = np.subtract(guess, gradient)

        # Check convergence, iterations=i
        if np.abs(f(lastGuess) - f(guess)) < threshold:
            iterations = i
            break

    return guess, f(guess), iterations

def func1D(v):
    return np.sin(v[0])**2

def func2D(v):
    #-4x/(x^2+y^2+1)
    return -4*v[0] / (v[0]**2 + v[1]**2 + 1)

def graph_helper(f, point):
  x = np.linspace(-5, 5, 100)
  y = []

  for i in x:
      y.append(f([i]))

  fig = plt.figure()
  ax = fig.add_subplot(1, 1, 1)
  ax.spines['left'].set_position('center')
  ax.spines['bottom'].set_position('zero')
  ax.spines['right'].set_color('none')
  ax.spines['top'].set_color('none')
  ax.xaxis.set_ticks_position('bottom')
  ax.yaxis.set_ticks_position('left')

  plt.plot(point[0], point[1], 'go')
  plt.plot(x, y, 'r')
  plt.show()

# point_x, point_y, iterations = gradient_descent(func1D, [-1.0], 0.5, 100)
point_xy, point_z, iterations = gradient_descent(func2D, [0.5, 0.6], 0.005, 1000)
print(point_xy, point_z, iterations)
# graph_helper(func1D, [point_x, point_y])