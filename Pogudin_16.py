import math
import numpy as np
from numpy.fft import fft, fftshift
import matplotlib.pyplot as plt

# Функция гауссового импульса
def gauss(q, m, d_g, w_g, d_t):
     return np.exp(-((((q - m) - (d_g / d_t))
                      / (w_g / dt)) ** 2))
 
# Параметры моделирования

W0 = 120.0 * np.pi # Волновое сопротивление свободного пространства

Sc = 1.0 # Число Куранта

c = 299792458.0 # Скорость света

maxSize_m = 0.5 # Присваиваем значение размера области моделирования

dx = 1.0e-3 # присваиваем значение дискрета по пространству

maxSize = math.floor(maxSize_m / dx + 0.5) # Рассчитываем размер области моделирования

maxTime = 498 # Время расчёта

dt = Sc * dx / c # Рассчитываем дискрет по времени

tlist = np.arange(0, maxTime * dt, dt) # Оформляем сетку

df = 1.0 / (maxTime * dt) # Рассчёт шага частоты

freq = np.arange(-maxTime / 2 * df, maxTime / 2 * df, df) # Рассчёт частотной сетки

# Параметры гауссова сигнала

A_0 = 100 # Задаём уровень ослабления в начальный момент времени

A_max = 100 # Задаём уровень ослабления на частоте F_max

F_max = 3e9 # Задаём ширину спектра по уровню 0.01

wg = np.sqrt(np.log(A_max)) / (np.pi * F_max) # Рассчитываем параметр области импульса

dg = wg * np.sqrt(np.log(A_0)) # Рассчитываем параметр остановки импульса

sourcePos_m = 0.05 # Источник

sourcePos = math.floor(sourcePos_m / dx + 0.5) # Положение источника

probePos_m = 0.25 # Задаём значение датчика

probePos = math.floor(probePos_m / dx + 0.5) # Рассчитываем положение датчика

# Определяем параметры датчика
probeEz = np.zeros(maxTime)
probeHy = np.zeros(maxTime)

# Определяем параметры полей
Ez = np.zeros(maxSize)
Hy = np.zeros(maxSize)

xlist = np.arange(0, maxSize_m, dx) # Пространственная сетка

plt.ion() # Выход из режима построения графика

fig, ax = plt.subplots() # Окно для графика

# Отображемые интервалы осей
ax.set_xlim(0, maxSize_m)
ax.set_ylim(-1.1, 1.1)

# Описываем оси
ax.set_xlabel('x, м')
ax.set_ylabel('Ez, В/м')

# Сетка на графике
ax.grid()

# Источник и датчик
ax.plot(sourcePos_m, 0, 'ok')
ax.plot(probePos_m, 0, 'xr')

# График
ax.legend(['Источник ({:.2f} м)'.format(sourcePos_m),
           'Датчик ({:.2f} м)'.format(probePos_m)],
          loc='lower right')

line, = ax.plot(xlist, Ez) # Отображение поля на начальном этапе

# Цикл для расчёта полей
for t in range(1, maxTime):
# Граничные условия для поля H
    Hy[-1] = Hy[-2]
# Расчет компоненты поля H
    Hy[:-1] = Hy[:-1] + (Ez[1:] - Ez[:-1]) * Sc / W0
    
# Метод Total Field / Scattered Field
    Hy[sourcePos - 1] -= (Sc / W0) * gauss(t, sourcePos, dg, wg, dt)
    
# Граничные условия для поля E
    Ez[0] = Ez[1]
    
# Расчет компоненты поля E
    Ez[1:] = Ez[1:] + (Hy[1:] - Hy[:-1]) * Sc * W0
    
# Метод Total Field / Scattered Field
    Ez[sourcePos] += Sc * gauss(t + 1, sourcePos, dg, wg, dt)
    
# Регистрация поля в датчиках
    probeHy[t] = Hy[probePos]
    probeEz[t] = Ez[probePos]
    
# Выход из интерактивного режима
plt.ioff()

EzSpec = fftshift(np.abs(fft(probeEz))) # Расчёт спектра сигнала

# Вывод сигнала и спектра
fig, (ax1, ax2) = plt.subplots(2, 1)
ax1.set_xlim(0, maxTime * dt)
ax1.set_ylim(0, 1.1)
ax1.set_xlabel('t, с')
ax1.set_ylabel('Ez, В/м')
ax1.plot(tlist, probeEz)
ax1.minorticks_on()
ax1.grid()
ax2.set_xlim(0, 10e9)
ax2.set_ylim(0, 1.1)
ax2.set_xlabel('f, Гц')
ax2.set_ylabel('|S| / |Smax|, б/р')
ax2.plot(freq, EzSpec / np.max(EzSpec))
ax2.minorticks_on()
ax2.grid()
plt.show()