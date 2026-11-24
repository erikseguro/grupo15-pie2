import numpy as  np
import matplotlib.pyplot as plt

# DATOS
 # Semilla para reproducibilidad
np . random . seed (42)

 # Tiempo de la mision [ minutos ]
t = np . linspace (0 , 120 , 1000)

 # Altitud registrada por telemetria [km] (con ligeras oscilaciones )
h = 250 + 1.2 * t + 5 * np . sin (0.1 * t )

 # Potencia de señal recibida en la estacion terrestre ( Gateway ) [dBm]
P_rx = -40 - 0.15 * t + 1.5 * np . random . randn (1000)

 # Datos atmosfericos tabulados de la U.S. Standard Atmosphere
h_tab = np . array ([200 , 300 , 400 , 500 , 600]) # Altitud [km]
rho_tab = np . array ([2.789e-10 , 1.95e-11 , 2.82e-12 , 5.21e-13 , 1.13e-13])

 # Matriz de acoplamiento de la antena phased - array y retardos medidos
A_cal = np . array ([[1.20 , 0.30 , 0.10] ,
[0.20 , 1.05 , 0.20] ,
[0.10 , 0.15 , 1.10]])
b_cal = np . array ([15.2 , 12.5 , 14.8])



# FUNCIONES my
from numpy import array, ndarray, shape

def my_dot(A: ndarray, B: ndarray) -> float:
    return sum (A[i] * B[i] for i in range(len(A)))

def my_matmul(A: ndarray, B: ndarray) -> ndarray:
    N,M = shape(A)
    return array([[my_dot(A[i,:], B[:,j]) for j in range(M)] for i in range(N)])

def my_inv(A: ndarray) -> ndarray:
    N = shape(A)[0]
    I = np.eye(N)
    A_inv = np.zeros_like(A)
    for i in range(N):
        A_inv[i] = np.linalg.solve(A, I[i])
    return A_inv

def my_matvec(A: ndarray, b: ndarray) -> ndarray:
    N, M = shape(A)
    return array([my_dot(A[i], b) for i in range(N)])

def my_transpose(A: ndarray) -> ndarray:
    N, M = shape(A)
    return array([[A[j][i] for j in range(N)] for i in range(M)])


#*********************
#  PASO 1 - CALIBRACIÓN
#*********************
def calibracion(A:np.ndarray, b:np.ndarray) -> np.ndarray:
    A_inv = my_inv(A)
    X = my_matvec(A_inv, b)
    return X

print("X->Coeficientes de calibración:", calibracion(A_cal, b_cal))


#**********************
#PASO 2 - ATENUACIÓN DE SEÑAL 
#*********************
# Ajuste lineal por mínimos cuadrados
m, c = np.polyfit(t, P_rx, 1)

# Calcular la potencia estimada
P_ajustada = m * t + c

# Mostrar resultados
print("->Resultados de la regresión lineal:")
print("Pendiente:", m, ":Tasa de degradación en dBm/min")
print("Ordenada en el origen:", c, ":Potencia inicial en dBm")
print("Ecuación de la potencia(Modelo: P_rx(t)) =", m, "* t +", c)



plt.scatter(t, P_rx, s=10, label="Datos medidos")
plt.plot(t, P_ajustada, label="Regresión lineal",color="red", linewidth=2)

plt.xlabel("Tiempo (min)")
plt.ylabel("Potencia recibida (dBm)")
plt.title("Atenuación de la señal")
plt.legend()
plt.grid(True)
plt.show()


#**********************
#PASO 3 - DENSIDAD ATMOSFÉRICA
#*********************
# Datos atmosfericos tabulados de la U.S. Standard Atmosphere
h_tab = np . array ([200 , 300 , 400 , 500 , 600]) # Altitud [km]
rho_tab = np . array ([2.7869e-10 , 1.95e-11 , 2.82e-12 , 5.21e-13 , 1.13e-13])

def interpolar_densidad (h_tab, rho_tab, h):
    h=np.asarray(h, dtype=float)
    log_rho_tab = np.log10(rho_tab)
    log_rho_interp = np.interp(h, h_tab, log_rho_tab)
    rho_interp = 10 ** log_rho_interp
    return rho_interp
print ("Densidad atmosférica interpolada:", interpolar_densidad(h_tab, rho_tab, h))
#**********************
#PASO 4 - ALTITUD CRITICA
#*********************



#**********************
#PASO 5 - CINEMÁTICA DEL SATÉLITE
#*********************



#**********************
#PASO 6 - TELEMETRÍA DE DATOS TRANSMITIDOS
#*********************



#**********************
#Gráfica
#*********************