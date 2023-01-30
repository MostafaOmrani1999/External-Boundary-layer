import sys
import numpy as np
import matplotlib.pyplot as plt

"""
Blasius solution for boundary layer flow(for flow problem) 

assumptions:
        ∂²u/∂x²<< ∂²u/∂y²
        Neglectiing momentum in Y direction
        No pressure gradient within the boundary layer (∂p/dx=0)
        
Countinuty :
         ∂u/∂x + ∂v/∂y = 0       
    
Momentum in X direction:
        u(∂u/∂x) + v(∂u/∂y) = 	μ/ρ(∂²u/∂y²)  (Nonlinear partial differntial equation)
        
Energy Equation:
    u(∂T/∂x) + v(∂T/∂y) = 	α(∂²T/∂y²)
        
Boundary conditions:
        @ y=0 u = v = 0
        as y⟶∞ u⟶U
            
Similarity vairiable:
        η = y/𝛿(x)
        𝛿(x) = √(vx/U) = x/√Re
        
New equation:
        d³f/dη³ + ½ (m+1) f df²/dη² + m * (1 - (df/dη)²) = 0  (Odrinary differntial equation)
        
New boundary conditions based on η:
        @ η=0 f = of suction/blowing coefficient and  df/dη = 0
        as η⟶∞ df/dη⟶1
        
m and of suction/blowing coefficient are inputs        

"""

m = float(input("Enter the value of m: "))
coeff = float(input("Enter the value of suction/blowing coefficient: "))

def RungeKutta(f, η0, ηmax, y, N):
    h = (ηmax - η0) / N 
    yn = [] 
    xn = [] 
    x = η0  
    
    for i in range(1, N + 1):
        k1 = f(x, y)
        k2 = f(x + h / 2, y + h * k1 / 2)
        k3 = f(x + h / 2, y + h * k2 / 2)
        k4 = f(x + h, y + h * k3)
        y = y + h * (k1 / 6 + k2 / 3 + k3 / 3 + k4 / 6)
        x = x + h
        yn.append(y) 
        xn.append(x) 
        
    return np.array(xn), np.array(yn)

def f(t, y):
    f = np.zeros(3)
    f[0] = y[1]
    f[1] = y[2]
    f[2] = -0.5 * (m + 1) * y[0] * y[2] - m * (1 - y[1] ** 2)
    return np.array(f)

def y0(c):
    return np.array([-2 * coeff / (m + 1), 0, c])

def F(c, ηmax):
    y = y0(c)
    xn, yn = RungeKutta(f, η0, ηmax, y, N)
    F = (yn[len(yn)-2])[1] - 1
    return F

def varr(F, c0, c1, err_max, ηmax):       
    global c
    F0 = F(c0, ηmax)
    F1 = F(c1, ηmax)
    iteration_counter = 0
    
    while abs(F1) > err_max :
        try:
            c = c1 - F1*(c1 - c0) / (F1 - F0)
        except ZeroDivisionError:
            print('Error! - F1 - F0 zero for c =', c1)
            sys.exit(1)
        c0 = c1
        c1 = c
        F0 = F1
        F1 = F(c1, ηmax)
        iteration_counter = iteration_counter + 1
    if abs(F(c,ηmax)) > err_max:
        iteration_counter = -1
    return c, iteration_counter

print ("   ")
η0 = 0
ηmax = 15
N = 8000
c0 = 0.01
c1 = 0.2
err_max = 1.0e-8

c, no_iterations = varr(F, c0, c1, err_max, ηmax)

y=y0(c)
xn, yn = RungeKutta(f, η0, ηmax, y, N)
f1 = yn[:,0]
f2 = yn[:,1]
f3 = yn[:,2]
err_max = 1.0e-8


"""
energy solution of boundary layer flow
 
Similarity vairiable:
    θ=(T-T_s)/(T-T_∞)
    
 New equation:
    dθ²/dη² + ½(m+1)Prξ(η)dθ/dη = 0
     
 Boundary conditions based on η:
    θ(0)=0
    as η⟶∞ θ⟶1

first we use falkner skan solution to find ξ"(0)
the by coupling the equations we find θ and dθ/dη
prandtl number is the input

"""

Pr = float(input("Enter the value of Prantdl number: "))


def RungeKutta(f, a, ηmax, y, N):
    h = (ηmax - η0) / N 
    yn = [] 
    xn = [] 
    x = η0  
    for i in range(1, N + 1):
        k1 = f(x, y)
        k2 = f(x + h / 2, y + h * k1 / 2)
        k3 = f(x + h / 2, y + h * k2 / 2)
        k4 = f(x + h, y + h * k3)
        y = y + h * (k1 / 6 + k2 / 3 + k3 / 3 + k4 / 6)
        x = x + h
        yn.append(y) 
        xn.append(x)     
    return np.array(xn), np.array(yn)

def heatflow(t, y):
    e = y[0:3]
    θ = y[3:5]
    return np.array([e[1], e[2], -0.5 * (m + 1) * e[0] * e[2] - m * (1 - e[1] ** 2) , θ[1], - 0.5 *  Pr * (m + 1) * e[0] * θ[1] ])


def y0(s):
    return np.array([-2 * coeff / (m + 1), 0, f3[0], 0, s])


def F(s, ηmax):
    y = y0(s)
    xn, yn = RungeKutta(heatflow, η0, ηmax, y, N)
    F = (yn[len(yn)-2])[3] - 1
    return F


def varr(F, s0, s1, err_max, ηmax):
    global c
    F0 = F(s0, ηmax)
    F1 = F(s1, ηmax)
    iteration_counter = 0

    while abs(F1) > err_max:
        try:
            s = s1 - F1*(s1 - s0) / (F1 - F0)
        except ZeroDivisionError:
            print('Error! - F1 - F0 zero for s =', s1)
            sys.exit(1)
        s0 = s1
        s1 = s
        F0 = F1
        F1 = F(s1, ηmax)
        iteration_counter = iteration_counter + 1
    if abs(F(s, ηmax)) > err_max:
        iteration_counter = -1
    return s


s0 = 1
s1 = 4.5

s = varr(F, s0, s1, err_max, ηmax)

y = y0(s)
zn, sn = RungeKutta(heatflow, η0, ηmax, y, N)

o1 = sn[:, 0]
o2 = sn[:, 1]
o3 = sn[:, 2]
θ1 = sn[:, 3]
θ2 = sn[:, 4]

print ("   ")
print('the values of ζ" is:')
print("    ")
print(o3[0])
print("   ")
print('the value of dθ/dη(0) is:')
print("   ")
print(θ2[0])

plt.figure(" Veclocity profiles")
plt.plot(o1,xn,label=" ξ ")
plt.plot(o2,xn,label=" ξ′ ")
plt.plot(o3,xn,label=" ξ″ ")
plt.title("Blasius equation solution")
plt.ylabel("η")
plt.grid()
plt.legend()

plt.figure(" Temperature profiles")
plt.plot(θ1, xn, label = "θ")
plt.plot(θ2, xn, label = "dθ/dη")
plt.title("Pohlhausen quation solution")
plt.ylabel("η")
plt.grid()
plt.legend()
