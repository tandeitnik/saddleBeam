#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from numba import njit

wl = 1550e-9
NA = 0.6
w0 = wl/(np.pi*NA)
k = 2*np.pi/wl
zr = (k*w0**2)/2
A = 1
B = 0.75
P = 0.5
c = 3e8
e0 = 8.85e-12
radius = 150e-9
n_particle = 1.4204 #https://refractiveindex.info/?shelf=main&book=SiO2&page=Nyakuchena
n_medium = 1
kB = 1.380649e-23
T0 = 300
rho_SiO2 = (2220+2170)/2 # https://www.azom.com/properties.aspx?ArticleID=1387
eta_air = 18.27e-6 # Pa # (J.T.R.Watson (1995)).
d_gas = 0.372e-9 #m #(Sone (2007)), ρSiO2
pressure = 1e-5 #mbar

e_r = (n_particle/n_medium)**2
V = 4*np.pi*radius**3/3
alpha_cm = 3*V*e0*(e_r-1)/(e_r+2)
alpha_rad = alpha_cm/(1-((e_r-1)/(e_r+2))*((k*radius)**2 + 2j/3*(k*radius)**3))
mass_particle = (4/3)*np.pi*radius**3*rho_SiO2
sigma_ext = (k/e0)*np.imag(alpha_rad)

kx = (2*P*np.real(alpha_rad)/(np.pi*c*e0*w0**2*(1+2*B**2))) * (4*np.sqrt(2)*B -2)/w0**2
ky = (2*P*np.real(alpha_rad)/(np.pi*c*e0*w0**2*(1+2*B**2))) * (-4*np.sqrt(2)*B -2)/w0**2
kz = (2*P*np.real(alpha_rad)/(np.pi*c*e0*w0**2*(1+2*B**2))) * (-1/zr**2)

w_x = np.sqrt(np.abs(kx)/mass_particle)
w_y = np.sqrt(np.abs(ky)/mass_particle)
w_z = np.sqrt(np.abs(kz)/mass_particle)

def Gamma_env(Pressure_mbar):
    
    def mfp(P_gas):
        mfp_val = kB*T0/(2**0.5*np.pi*d_gas**2*P_gas)
        return mfp_val
    
    Pressure_pascals = 100*Pressure_mbar
    s = mfp(Pressure_pascals)
    K_n = s/radius
    c_K = 0.31*K_n/(0.785 + 1.152*K_n + K_n**2)
    gamma = 6*np.pi*eta_air*radius/mass_particle * 0.619/(0.619 + K_n) * (1+c_K)
    return gamma

gamma = Gamma_env(pressure)

stabilityCriteria = np.sqrt(0.5*(-gamma**2+w_y**2-w_x**2) + 0.5*np.sqrt((gamma**2+w_x**2-w_y**2)**2+4*w_x**2*w_y**2))

omega_x_gauss = np.sqrt( (4*P*np.real(alpha_rad)) / (c*e0*np.pi*w0**4*mass_particle) )
omega_y_gauss = np.sqrt( (4*P*np.real(alpha_rad)) / (c*e0*np.pi*w0**4*mass_particle) )
omega_z_gauss = np.sqrt( (2*P*np.real(alpha_rad)) / (c*e0*np.pi*w0**2*zr**2*mass_particle) )

z_eq = 3*wl**4*e0*np.real(alpha_rad)/(16*np.pi**3*np.abs(alpha_rad)**2) -0.5*np.sqrt((3*wl**4*e0*np.real(alpha_rad)/(8*np.pi**3*np.abs(alpha_rad)**2))**2 -4*zr**2)

x_gauss = np.sqrt(kB*T0/(mass_particle*omega_x_gauss**2))
y_gauss = np.sqrt(kB*T0/(mass_particle*omega_y_gauss**2))
z_gauss = z_eq + np.sqrt(kB*T0/(mass_particle*omega_z_gauss**2))

v_gauss = np.sqrt(kB*T0/mass_particle)

#simulation parameters
pos0 = np.array((x_gauss,y_gauss,z_gauss))
vel0 = np.array((v_gauss,v_gauss,v_gauss))

#Define the time-step of the numerical integration and the max. time of integration
N = 200_000 #how many steps are simulated
dt_simulation = 1e-9 #simulation time step

Omega = stabilityCriteria*2

@njit(fastmath=True)
def intensity(x,y,z,t):
    
    r = np.sqrt(x**2+y**2)
    phi = np.arctan2(y,x) +-1*(np.sign(np.arctan2(y,x))-1)*np.pi

    w = w0*np.sqrt(1 + z**2/zr**2)
            
    chi = np.arctan2(z,zr)
    
    pre = (2*P/(c*e0))*(1/(1+2*B**2))*(2/np.pi)*(1/w**2)*np.exp(-2*r**2/w**2)
    coef_0 = 1
    coef_2 = 4*np.sqrt(2)*B*np.cos(2*phi+2*Omega*t)*np.cos(2*chi)/w**2
    coef_4 = 8*B**2*np.cos(2*phi+2*Omega*t)**2/w**4
    
    intens = c*e0*pre*(coef_0 + coef_2*r**2 + coef_4*r**4)/2
    
    return intens
@njit(fastmath=True)
def taylor_x(x,y,z,t):
    
    ans = y* (-((32 *np.sqrt(2)*B* P* np.sin(2*Omega*t))/((1 + 2*B**2) *c* np.pi* w0**4*e0)))+2*x*((8*P*(-1 + 2*np.sqrt(2)*B*np.cos(2*Omega*t)))/((1 + 2*B**2) *c* np.pi* w0**4*e0))
    
    return (np.real(alpha_rad)/4)*ans
@njit(fastmath=True)
def taylor_y(x,y,z,t):
    
    ans = 2 *y* (-((8* P* (1 + 2* np.sqrt(2)* B* np.cos(2*Omega*t)))/((1 + 2* B**2) *c* np.pi* w0**4 *e0))) + x* (-((32 *np.sqrt(2)*B* P* np.sin(2*Omega*t))/((1 + 2* B**2) *c* np.pi* w0**4 *e0)))
                                                      
    return (np.real(alpha_rad)/4)*ans
@njit(fastmath=True)
def taylor_z(x,y,z,t):
    
    ans = -((8 *P* z)/((1 + 2* B**2) *c* np.pi *w0**2* zr**2* e0))
    
    return (np.real(alpha_rad)/4)*ans

@njit(fastmath=True)
def ode_x(x,y,z,v_x,v_y,t):
    
    ode_1 = v_x
    ode_2 = taylor_x(x,y,z,t)/mass_particle -gamma*v_x

    return np.array((ode_1,ode_2))
@njit(fastmath=True)
def ode_y(x,y,z,v_x,v_y,t):
    
    ode_1 = v_y
    ode_2 = taylor_y(x,y,z,t)/mass_particle -gamma*v_y

    return np.array((ode_1,ode_2))
@njit(fastmath=True)
def ode_z(x,y,z,v_z,t):
    
    ode_1 = v_z
    ode_2 = taylor_z(x,y,z,t)/mass_particle -gamma*v_z  + intensity(x,y,z,t)*sigma_ext/(c*mass_particle)

    return np.array((ode_1,ode_2))


#The next function applies an integration step perfoming Runge-Kutta 4th order
@njit(fastmath=True)
def rungeKuttaStep(pos,vel,dt,t):
    
    x = pos[0]
    y = pos[1]
    z = pos[2]
    
    v_x = vel[0]
    v_y = vel[1]
    v_z = vel[2]
    
    k1_x = dt*ode_x(x,y,z,v_x,v_y,t)
    k1_y = dt*ode_y(x,y,z,v_x,v_y,t)
    k1_z = dt*ode_z(x,y,z,v_z,t)
    
    k2_x = dt*ode_x(x+k1_x[0]/2,y+k1_y[0]/2,z+k1_z[0]/2,v_x+k1_x[1]/2,v_y+k1_y[1]/2,t)
    k2_y = dt*ode_y(x+k1_x[0]/2,y+k1_y[0]/2,z+k1_z[0]/2,v_x+k1_x[1]/2,v_y+k1_y[1]/2,t)
    k2_z = dt*ode_z(x+k1_x[0]/2,y+k1_y[0]/2,z+k1_z[0]/2,v_z+k1_z[1]/2,t)
    
    k3_x = dt*ode_x(x+k2_x[0]/2,y+k2_y[0]/2,z+k2_z[0]/2,v_x+k2_x[1]/2,v_y+k2_y[1]/2,t)
    k3_y = dt*ode_y(x+k2_x[0]/2,y+k2_y[0]/2,z+k2_z[0]/2,v_x+k2_x[1]/2,v_y+k2_y[1]/2,t)
    k3_z = dt*ode_z(x+k2_x[0]/2,y+k2_y[0]/2,z+k2_z[0]/2,v_z+k2_z[1]/2,t)
    
    k4_x = dt*ode_x(x+k3_x[0],y+k3_y[0],z+k3_z[0],v_x+k3_x[1],v_y+k3_y[1],t)
    k4_y = dt*ode_y(x+k3_x[0],y+k3_y[0],z+k3_z[0],v_x+k3_x[1],v_y+k3_y[1],t)
    k4_z = dt*ode_z(x+k3_x[0],y+k3_y[0],z+k3_z[0],v_z+k3_z[1],t)
    
    newPos = np.array((
        (x + (k1_x[0] + 2*k2_x[0] + 2*k3_x[0] + k4_x[0])/6),
        (y + (k1_y[0] + 2*k2_y[0] + 2*k3_y[0] + k4_y[0])/6),
        (z + (k1_z[0] + 2*k2_z[0] + 2*k3_z[0] + k4_z[0])/6)))
    
    
    newVel = np.array((
        (v_x + (k1_x[1] + 2*k2_x[1] + 2*k3_x[1] + k4_x[1])/6 + np.sqrt(2*kB *T0*gamma*mass_particle)*np.random.normal()*np.sqrt(dt)/mass_particle),
        (v_y + (k1_y[1] + 2*k2_y[1] + 2*k3_y[1] + k4_y[1])/6 + np.sqrt(2*kB *T0*gamma*mass_particle)*np.random.normal()*np.sqrt(dt)/mass_particle),
        (v_z + (k1_z[1] + 2*k2_z[1] + 2*k3_z[1] + k4_z[1])/6 + np.sqrt(2*kB *T0*gamma*mass_particle)*np.random.normal()*np.sqrt(dt)/mass_particle)))
    
    
    return newPos, newVel

#The next funcion applies the Runge-Kutta method the desired time interval
@njit(fastmath=True)
def rungeKutta(pos0,vel0,dt_sim,tMax,t):
    
    posSim = np.zeros((3,int(tMax/dt_sim)+1))
    velSim = np.zeros((3,int(tMax/dt_sim)+1))
                      
    posSim[:,0] = pos0
    velSim[:,0] = vel0
    
    for i in range(1,int(tMax/dt_sim)):
        
        newPos, newVel = rungeKuttaStep(posSim[:,i-1],velSim[:,i-1],dt_sim,t[i])
        posSim[0,i] = newPos[0]
        posSim[1,i] = newPos[1]
        posSim[2,i] = newPos[2]
        velSim[0,i] = newVel[0]
        velSim[1,i] = newVel[1]
        velSim[2,i] = newVel[2]
        

    return posSim, velSim


#Time parameters
tMax = N*dt_simulation
t = np.linspace(0,tMax,N)

#simulate the system!
positions, velocities = rungeKutta(pos0,vel0,dt_simulation,tMax,t)

pos_x = positions[0,:-1]
pos_y = positions[1,:-1]
pos_z = positions[2,:-1]

rho = np.sqrt(pos_x**2+pos_y**2)


# np.save('rhoVarianceMean_taylor.npy',rhoVarianceMean)

# plt.rcParams.update({
#     "text.usetex": True,
#     "font.family": "Times New Roman",
#     'font.size': 10
# })

# plt.rcParams["axes.linewidth"] = 1

# plt.scatter(Omega_list/stabilityCriteria,np.sqrt(rhoVarianceMean), marker = 'd', label = 'lin. dynamics')
# plt.xlabel(r'$\Omega/\Omega_c$')
# plt.ylabel(r'$\sqrt{\langle r^2\rangle}/\lambda_0$')
# plt.yscale('log')
# plt.grid(alpha = 0.4)
# plt.legend(loc = 'upper right', fancybox = True, shadow = True)

# f, PSD = signal.welch(pos_x, fs = 1/dt_sampling, nfft = 2**int(np.log2(len(pos_x))+1),nperseg = int(len(pos_x)/10),scaling='density',return_onesided=True)
# plt.plot(f,PSD)
# plt.yscale('log')