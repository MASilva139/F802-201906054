# Documentación del Proyecto: Simulación del Oscilador Armónico Cuántico (QHO)

## 📋 Tabla de Contenidos

1. [Introducción](#introducción)
2. [Descripción del Proyecto](#descripción-del-proyecto)
3. [Marco Teórico](#marco-teórico)
4. [Pestañas de Simulación](#pestañas-de-simulación)
5. [Observables del Sistema](#observables-del-sistema)
6. [Guía de Uso](#guía-de-uso)
7. [Estructura del Código](#estructura-del-código)
8. [Exportación de Datos](#exportación-de-datos)
9. [Referencias](#referencias)

---

## 1. Introducción

Este proyecto implementa una **simulación interactiva en tiempo real** del Oscilador Armónico Cuántico (QHO) con múltiples representaciones de estados cuánticos y clásicos. La visualización se realiza en el **espacio de fase** (X, P) utilizando Pygame, permitiendo observar la evolución temporal de diferentes estados cuánticos y compararlos con el oscilador clásico.

### Características principales:
- ✅ **6 pestañas independientes** con diferentes tipos de simulaciones
- ✅ **Visualización en tiempo real a 60 FPS**
- ✅ **Exportación automática de datos** en formato CSV
- ✅ **Más de 40 observables** físicos calculados por frame
- ✅ **Interacción mediante drives y fuerzas externas**
- ✅ **Comparación cuántico vs. clásico**

---

## 2. Descripción del Proyecto

### 2.1 Objetivos

El proyecto tiene como objetivos principales:

1. **Visualizar** la evolución de diferentes estados cuánticos en el espacio de fase
2. **Comparar** el comportamiento cuántico con el clásico
3. **Estudiar** el efecto de fuerzas externas (drives) en sistemas cuánticos
4. **Analizar** las propiedades estadísticas y de información cuántica
5. **Generar datos** para análisis posterior mediante técnicas de procesamiento de señales

### 2.2 Tecnologías Utilizadas

- **Python 3.8+**: Lenguaje de programación principal
- **NumPy**: Cálculos numéricos y álgebra lineal
- **Pygame**: Visualización gráfica en tiempo real
- **SciPy**: Funciones matemáticas avanzadas (expm para propagadores)
- **CSV**: Almacenamiento de datos para análisis posterior

---

## 3. Marco Teórico

### 3.1 El Oscilador Armónico Cuántico

El Hamiltoniano del oscilador armónico cuántico (sin fuerza externa) está dado por:

```
Ĥ₀ = ℏω(â†â + 1/2)
```

Donde:
- `ω`: Frecuencia angular del oscilador
- `â` y `â†`: Operadores de aniquilación y creación
- `ℏ`: Constante de Planck reducida (ℏ = 1 en unidades naturales)

### 3.2 Operadores de Cuadratura

Los operadores de posición y momento adimensionales se definen como:

```
X̂ = (â + â†)/√2
P̂ = i(â† - â)/√2
```

Estos operadores satisfacen la relación de conmutación canónica `[X̂, P̂] = i`.

### 3.3 Estados Cuánticos Implementados

#### 3.3.1 Estado Coherente |α⟩
- **Definición**: Autoestado del operador de aniquilación `â|α⟩ = α|α⟩`
- **Propiedad**: Estado de mínima incerteza con `ΔX = ΔP = 1/√2`
- **Interpretación**: Estado "más clásico" del oscilador cuántico

#### 3.3.2 Estado Comprimido (Squeezed State)
- **Definición**: Estado con incerteza reducida en una cuadratura a expensas de la otra
- **Parámetros**: `r` (squeezing parameter) y `θ` (ángulo de squeezing)
- **Propiedad**: `ΔX·ΔP ≥ 1/2`, pero `ΔX ≠ ΔP`

#### 3.3.3 Superposición de Estados de Fock
- **Definición**: Combinación lineal `|ψ⟩ = Σ cₙ|n⟩`
- **Característica**: Incertezas variables en el tiempo
- **Aplicación**: Estudio de efectos de interferencia cuántica

#### 3.3.4 Oscilador Clásico
- **Ecuación de movimiento**: `ẍ + ω²x = 0`
- **Solución**: `x(t) = A cos(ωt + φ)`
- **Trayectoria**: Círculo en el espacio de fase

### 3.4 Función de Green

La respuesta del sistema a una fuerza externa `F(t)` se calcula mediante:

```
α(t) = ∫₀ᵗ G(t-t') F(t') dt'
```

Donde `G(t)` es la función de Green del oscilador armónico.

**Dos implementaciones:**

1. **Método Analítico**: Solución cerrada para fuerza armónica
2. **Split-Operator**: Propagación numérica `U = e^(-iĤ₀dt/2) e^(-iFX̂dt) e^(-iĤ₀dt/2)`

---

## 4. Pestañas de Simulación

### Pestaña 1: Estado Coherente Cuántico 🔴
- **Color de trayectoria**: Rojo
- **Estado inicial**: |α₀⟩ con α₀ = 1.5·e^(iπ/6)
- **Drive**: F(t) = F₀cos(νt), activable con tecla `D`
- **Visualización**: Elipse verde de incerteza que rota

### Pestaña 2: Estado Comprimido 🔴
- **Color de trayectoria**: Rojo
- **Estado inicial**: Estado comprimido con r = 0.7, θ = π/4
- **Característica**: Elipse elongada que rota
- **Drive**: Igual que coherente

### Pestaña 3: Superposición de Estados de Fock 🟣
- **Color de trayectoria**: Púrpura
- **Estado inicial**: Superposición de |0⟩ a |5⟩
- **Característica**: Incertezas variables (elipse se deforma)
- **Drive**: Desplazamiento del centro de masa

### Pestaña 4: Oscilador Clásico 🔵
- **Color de trayectoria**: Azul
- **Ecuación**: Harmónico simple ẍ + ω²x = 0
- **Visualización**: Círculo gris de referencia (amplitud A)

### Pestaña 5: Green Split-Operator 🟦
- **Color de trayectoria**: Cian
- **Método**: Propagación numérica con 30 estados de Fock
- **Estado inicial**: |0⟩ (ground state)
- **Fuerza**: F(t) = F₀cos(νt), activable con tecla `F`
- **Característica**: Solución exacta numérica

### Pestaña 6: Green Analítico 🟠
- **Color de trayectoria**: Naranja
- **Método**: Solución analítica cerrada
- **Estado inicial**: α = 0
- **Fuerza**: Igual que Split-Operator
- **Característica**: Rápida, ideal para parámetros resonantes

---

## 5. Observables del Sistema

### 5.1 Observables Básicos de Cuadratura

| Observable | Símbolo | Descripción | Unidades |
|-----------|---------|-------------|----------|
| `tiempo` | t | Tiempo de simulación | s |
| `frame_number` | - | Número de frame desde t=0 | - |
| `X_avg` | ⟨X̂⟩ | Valor esperado de la posición adimensional | - |
| `P_avg` | ⟨P̂⟩ | Valor esperado del momento adimensional | - |
| `delta_X` | ΔX | Desviación estándar de la posición | - |
| `delta_P` | ΔP | Desviación estándar del momento | - |
| `producto_incerteza` | ΔX·ΔP | Producto de incertezas (≥ 1/2 por Heisenberg) | - |

**Interpretación física:**
- **⟨X̂⟩ y ⟨P̂⟩**: Centro del paquete de ondas en el espacio de fase
- **ΔX y ΔP**: Ancho del paquete de ondas en cada dirección
- **ΔX·ΔP**: Medida de "no-clasicalidad"; mínimo = 1/2 para estados coherentes

---

### 5.2 Parámetro de Desplazamiento Complejo α

| Observable | Símbolo | Descripción | Unidades |
|-----------|---------|-------------|----------|
| `alpha_real` | Re(α) | Parte real del parámetro de desplazamiento | - |
| `alpha_imag` | Im(α) | Parte imaginaria del parámetro de desplazamiento | - |
| `alpha_magnitud` | \|α\| | Magnitud del desplazamiento | - |
| `alpha_fase` | arg(α) | Fase del parámetro complejo | rad |

**Definición:**
```
α = (⟨X̂⟩ + i⟨P̂⟩)/√2
```

**Interpretación:**
- **|α|²**: Número promedio de fotones (excitación del oscilador)
- **arg(α)**: Fase del estado coherente
- **α en plano complejo**: Representación compacta del estado

---

### 5.3 Observables de Energía

| Observable | Símbolo | Descripción | Fórmula | Unidades |
|-----------|---------|-------------|---------|----------|
| `energia_cinetica` | T | Energía cinética | ⟨P̂²⟩/2 | ℏω |
| `energia_potencial` | V | Energía potencial | ⟨X̂²⟩/2 | ℏω |
| `energia_total` | E | Energía total del sistema | T + V | ℏω |

**Nota:** En unidades naturales (ℏ=1, m=1, ω=1), la energía se expresa en unidades de ℏω.

**Conservación de energía:**
- Sin fuerza externa: E = constante
- Con fuerza externa: dE/dt = F(t)·⟨P̂⟩ (potencia inyectada)

---

### 5.4 Trabajo y Potencia (Solo Green)

| Observable | Símbolo | Descripción | Fórmula | Unidades |
|-----------|---------|-------------|---------|----------|
| `trabajo_instantaneo` | W | Trabajo instantáneo de la fuerza | F(t)·⟨X̂⟩ | ℏω |
| `potencia` | P | Potencia instantánea | F(t)·⟨P̂⟩ | ℏω/s |

**Interpretación termodinámica:**
- **Trabajo**: Energía transferida por desplazamiento
- **Potencia**: Tasa de cambio de energía del sistema
- **∫P dt = ΔE**: La integral de potencia da el cambio de energía

---

### 5.5 Matriz de Covarianza (Σ)

La matriz de covarianza describe las correlaciones cuánticas:

```
Σ = [ ⟨X̂²⟩ - ⟨X̂⟩²    ⟨X̂P̂⟩ - ⟨X̂⟩⟨P̂⟩ ]
    [ ⟨X̂P̂⟩ - ⟨X̂⟩⟨P̂⟩    ⟨P̂²⟩ - ⟨P̂⟩²   ]
```

| Observable | Descripción | Significado Físico |
|-----------|-------------|-------------------|
| `Sigma_XX` | Σ₁₁ | Varianza de la posición |
| `Sigma_XP` | Σ₁₂ = Σ₂₁ | Covarianza posición-momento |
| `Sigma_PP` | Σ₂₂ | Varianza del momento |
| `det_Sigma` | det(Σ) | Determinante (área mínima = 1/4) |
| `traza_Sigma` | Tr(Σ) | Traza (suma de varianzas) |

**Propiedades:**
- **det(Σ) ≥ 1/4**: Principio de incertidumbre en forma de matriz
- **Σ simétrica**: La covarianza es simétrica por definición
- **Autovalores > 0**: Matriz definida positiva

---

### 5.6 Autovalores y Geometría de la Elipse

| Observable | Descripción | Interpretación |
|-----------|-------------|----------------|
| `lambda_1` | λ₁ | Mayor autovalor de Σ (semieje mayor²) |
| `lambda_2` | λ₂ | Menor autovalor de Σ (semieje menor²) |
| `theta_ellipse` | θ | Ángulo de orientación de la elipse | rad |
| `excentricidad` | e | Excentricidad de la elipse | - |
| `area_elipse` | A | Área de la elipse de incerteza | π·√det(Σ) |

**Fórmulas:**
```
Semiejes: a = √λ₁,  b = √λ₂
Excentricidad: e = √(1 - λ₂/λ₁)
Área: A = πab = π√det(Σ)
```

**Interpretación:**
- **Elipse en espacio de fase**: Región de incerteza cuántica
- **Rotación con ω**: La elipse rota con frecuencia del oscilador
- **Área constante**: Teorema de Liouville para estados puros

---

### 5.7 Información Cuántica

| Observable | Símbolo | Descripción | Rango | Interpretación |
|-----------|---------|-------------|-------|----------------|
| `pureza` | γ | Pureza del estado | [0, 1] | γ=1: estado puro, γ<1: mixto |
| `entropia` | S | Entropía de von Neumann | [0, ∞) | Medida de mixtura del estado |

**Fórmulas para estados gaussianos:**
```
Pureza: γ = 1/(2√det(Σ))
Entropía: S(ν) donde ν = √det(Σ)
```

**Significado físico:**
- **Pureza alta**: Estado bien definido (onda coherente)
- **Entropía alta**: Estado mezclado (pérdida de coherencia)
- **Relación**: Estados puros tienen S=0 y γ=1

---

### 5.8 Estadística de Fotones (Estados Coherentes/Comprimidos)

| Observable | Descripción | Fórmula | Interpretación |
|-----------|-------------|---------|----------------|
| `n_promedio` | ⟨n̂⟩ | Número promedio de fotones | \|α\|² | Excitación del oscilador |
| `n_varianza` | (Δn)² | Varianza del número de fotones | - | Fluctuaciones cuánticas |
| `mandel_Q` | Q | Parámetro de Mandel | (Δn² - ⟨n̂⟩)/⟨n̂⟩ | Tipo de estadística |
| `fano_F` | F | Factor de Fano | Δn²/⟨n̂⟩ | Razón ruido/señal |

**Clasificación por Mandel Q:**
- **Q = 0**: Estadística de Poisson (luz coherente)
- **Q < 0**: Sub-Poisson (luz comprimida, no-clásica)
- **Q > 0**: Super-Poisson (luz térmica, agrupamiento)

---

### 5.9 Squeezing (Solo Estado Comprimido)

| Observable | Descripción | Unidades | Interpretación |
|-----------|-------------|----------|----------------|
| `squeezing_r` | r | - | Parámetro de squeezing |
| `squeezing_theta_inicial` | θ₀ | rad | Ángulo inicial de squeezing |
| `squeezing_theta_actual` | θ(t) | rad | Ángulo actual = θ₀ + ωt |
| `squeezing_dB` | Sq | dB | Squeezing en decibelios |

**Fórmula de squeezing en dB:**
```
Sq = -10·log₁₀(min(ΔX², ΔP²))
```

**Interpretación:**
- **r > 0**: Grado de compresión de la incerteza
- **Sq > 0 dB**: Reducción cuántica por debajo del shot noise
- **Aplicación**: Mejora de sensibilidad en interferometría

---

### 5.10 Superposición de Fock - Probabilidades

| Observable | Descripción | Rango |
|-----------|-------------|-------|
| `prob_n0` a `prob_n5` | P(n) | [0, 1] |
| `c{n}_real` | Re(cₙ) | Parte real del coeficiente |
| `c{n}_imag` | Im(cₙ) | Parte imaginaria del coeficiente |
| `c{n}_abs` | \|cₙ\| | Magnitud del coeficiente |
| `c{n}_arg` | arg(cₙ) | Fase del coeficiente |

**Estado:**
```
|ψ⟩ = Σ cₙ|n⟩
donde Σ|cₙ|² = 1
```

**Probabilidad de medir n fotones:**
```
P(n) = |⟨n|ψ⟩|² = |cₙ|²
```

---

### 5.11 Coherencias Cuánticas (Superposición)

| Observable | Descripción | Interpretación |
|-----------|-------------|----------------|
| `rho_{nm}_real` | Re(ρₙₘ) | Parte real de elemento de matriz densidad |
| `rho_{nm}_imag` | Im(ρₙₘ) | Parte imaginaria |
| `rho_{nm}_abs` | \|ρₙₘ\| | Magnitud de la coherencia |

**Matriz densidad:**
```
ρ = |ψ⟩⟨ψ| = Σₙₘ cₙc*ₘ |n⟩⟨m|
Elementos: ρₙₘ = cₙc*ₘ
```

**Interpretación:**
- **Diagonal (n=m)**: Poblaciones P(n)
- **Fuera diagonal (n≠m)**: Coherencias cuánticas (interferencia)
- **|ρₙₘ|**: Grado de superposición entre estados |n⟩ y |m⟩

---

### 5.12 Momentos de Orden Superior (Superposición)

| Observable | Descripción | Uso |
|-----------|-------------|-----|
| `X3_momento` | ⟨X̂³⟩ | Asimetría de la distribución |
| `X4_momento` | ⟨X̂⁴⟩ | Kurtosis (grosor de colas) |
| `P3_momento` | ⟨P̂³⟩ | Asimetría en momento |
| `P4_momento` | ⟨P̂⁴⟩ | Kurtosis en momento |
| `skewness_X` | γ₁ₓ | Asimetría normalizada de X |
| `skewness_P` | γ₁ₚ | Asimetría normalizada de P |
| `kurtosis_X` | γ₂ₓ | Exceso de kurtosis de X |
| `kurtosis_P` | γ₂ₚ | Exceso de kurtosis de P |

**Fórmulas:**
```
Skewness: γ₁ = ⟨X̂³⟩/(ΔX)³
Kurtosis: γ₂ = ⟨X̂⁴⟩/(ΔX)⁴ - 3
```

**Interpretación:**
- **Skewness = 0**: Distribución simétrica (Gaussiana)
- **Skewness ≠ 0**: Asimetría (más peso a un lado)
- **Kurtosis = 0**: Distribución Gaussiana
- **Kurtosis > 0**: Colas más pesadas (más eventos extremos)

---

### 5.13 Oscilador Clásico - Observables Específicos

| Observable | Descripción | Unidades |
|-----------|-------------|----------|
| `posicion` | x(t) | - |
| `velocidad` | v(t) = ẋ(t) | - |
| `aceleracion` | a(t) = ẍ(t) | - |
| `momentum` | p = mv | - |
| `amplitud` | A | Amplitud de oscilación | - |
| `fase_inicial` | φ₀ | Fase en t=0 | rad |
| `fase_instantanea` | φ(t) | Fase actual | rad |
| `periodo` | T = 2π/ω | Periodo de oscilación | s |
| `frecuencia` | f = ω/(2π) | Frecuencia | Hz |
| `distancia_origen` | r | Radio en espacio de fase | - |

**Ecuaciones:**
```
x(t) = A·cos(ωt + φ₀)
v(t) = -Aω·sin(ωt + φ₀)
E = (v² + ω²x²)/2
```

---

### 5.14 Fuerza Externa y Driving (Green)

| Observable | Descripción | Unidades |
|-----------|-------------|----------|
| `fuerza_externa` | F(t) | Fuerza aplicada instantánea | - |
| `F0_amplitud` | F₀ | Amplitud de la fuerza | - |
| `nu_frecuencia` | ν | Frecuencia de la fuerza | rad/s |
| `omega_oscilador` | ω | Frecuencia natural | rad/s |
| `force_activa` | 0 o 1 | Estado de la fuerza (ON/OFF) | - |

**Fuerza armónica:**
```
F(t) = F₀·cos(νt)
```

**Resonancia:**
- **ν ≈ ω**: Resonancia (amplitud crece linealmente con t)
- **ν ≠ ω**: Batimiento entre dos frecuencias
- **|ω - ν|**: Medida de detuning

---

### 5.15 Periodos y Frames

| Observable | Descripción |
|-----------|-------------|
| `periodo_oscilador` | Número de periodos completados (ωt/2π) |
| `periodo_fuerza` | Número de ciclos de fuerza (νt/2π) |
| `drive_activo` | Estado del drive (0=OFF, 1=ON) |
| `metodo` | Método de cálculo ('analitico', 'split_operator') |
| `n_basis` | Número de estados en base de Fock (solo Split-Op) |

---

## 6. Guía de Uso

### 6.1 Controles del Teclado

#### Controles Globales:
- **1, 2, 3, 4, 5, 6**: Cambiar entre pestañas
- **G**: Activar/desactivar guardado automático de datos
- **E**: Exportar CSV de la pestaña actual
- **ESC**: Salir de la simulación

#### Controles Específicos:
- **D**: Toggle drive (pestañas 1, 2, 3)
- **F**: Toggle fuerza (pestañas 5, 6)
- **+/-**: Ajustar frecuencia ν de la fuerza (pestañas 5, 6)
- **[/]**: Ajustar amplitud F₀ de la fuerza (pestañas 5, 6)
- **R**: Reset completo (pestañas 5, 6)

### 6.2 Interfaz Visual

Cada pestaña muestra:
1. **Área de simulación** (600×600 px): Espacio de fase con ejes X y P
2. **Trayectoria coloreada**: Histórico del movimiento (últimos 2000 puntos)
3. **Elipse verde**: Región de incerteza cuántica (1σ)
4. **Punto negro**: Posición actual ⟨X̂⟩, ⟨P̂⟩
5. **Panel de observables**: Valores numéricos actuales
6. **Panel de parámetros**: Configuración del sistema
7. **Gráfico de historia** (Green): Evolución temporal de F(t)

### 6.3 Interpretación Visual

- **Trayectoria circular**: Comportamiento cuasi-clásico
- **Elipse rotando**: Estado coherente/comprimido
- **Elipse deformándose**: Superposición de Fock (no-clásico)
- **Vector rojo** (Green): Fuerza externa aplicada

---

## 7. Estructura del Código

### 7.1 Arquitectura

```
qho_6_tabs.py
│
├── SimulacionBase (clase base)
│   ├── world_to_screen()
│   ├── draw_axes()
│   ├── exportar_csv()
│   └── update()
│
├── EstadoCoherente
│   ├── get_alpha()
│   ├── rotate_covariance()
│   └── forcing()
│
├── EstadoComprimido (hereda de EstadoCoherente)
│
├── SuperposicionEstados
│   ├── _precalcular_matrices()
│   ├── get_coeffs_t()
│   ├── get_observables()
│   └── _calcular_momentos_superiores_eficiente()
│
├── OsciladorClasico
│   └── get_position()
│
├── GreenSplitOperator
│   ├── inicializar_estado()
│   ├── hamiltoniano_libre()
│   ├── operador_posicion_fock()
│   ├── propagador_split_operator()
│   └── calcular_observables()
│
└── GreenAnalitico
    ├── alpha_respuesta_fuerza()
    ├── get_position_momentum()
    └── get_covariance_matrix()
```

### 7.2 Flujo de Ejecución

1. **Inicialización**: Crear instancias de las 6 simulaciones
2. **Loop principal** (60 FPS):
   - Procesar eventos del teclado/mouse
   - Actualizar solo la simulación activa
   - Renderizar la escena
   - Guardar datos (si GUARDAR_AUTOMATICO = True)
3. **Al cambiar pestaña**: Exportar CSV automáticamente
4. **Al cerrar**: Exportar todos los CSVs pendientes

---

## 8. Exportación de Datos

### 8.1 Formato CSV

Cada simulación genera un archivo CSV con:
- **Nombre**: `{nombre_simulacion}_{timestamp}.csv`
- **Formato**: Valores separados por comas
- **Cabeceras**: Nombres de observables
- **Frecuencia**: 1 fila por frame (60 filas/segundo)

### 8.2 Análisis Posterior

Los datos exportados pueden ser analizados con:
- **Python**: pandas, matplotlib, scipy
- **MATLAB**: readtable, fft, fitlm
- **R**: read.csv, ggplot2, signal
- **Excel**: Para visualización rápida

### 8.3 Ejemplo de Análisis

```python
import pandas as pd
import matplotlib.pyplot as plt

# Cargar datos
df = pd.read_csv('estado_coherente_20250111_120000.csv')

# Graficar trayectoria
plt.figure(figsize=(8, 8))
plt.plot(df['X_avg'], df['P_avg'])
plt.xlabel('⟨X̂⟩')
plt.ylabel('⟨P̂⟩')
plt.title('Trayectoria en Espacio de Fase')
plt.axis('equal')
plt.grid(True)
plt.show()

# FFT para análisis de frecuencias
from scipy.fft import fft, fftfreq

N = len(df)
dt = df['tiempo'].iloc[1] - df['tiempo'].iloc[0]
freq = fftfreq(N, dt)
fft_X = fft(df['X_avg'])

plt.figure()
plt.plot(freq[:N//2], np.abs(fft_X[:N//2]))
plt.xlabel('Frecuencia')
plt.ylabel('Amplitud FFT')
plt.title('Espectro de Frecuencias')
plt.show()
```

---

## 9. Referencias

### 9.1 Bibliografía Recomendada

1. **Gerry, C., & Knight, P. (2005)**. *Introductory Quantum Optics*. Cambridge University Press.
   - Capítulos sobre estados coherentes y comprimidos

2. **Walls, D. F., & Milburn, G. J. (2008)**. *Quantum Optics* (2nd ed.). Springer.
   - Teoría completa del oscilador cuántico

3. **Sakurai, J. J., & Napolitano, J. (2017)**. *Modern Quantum Mechanics* (2nd ed.). Cambridge University Press.
   - Fundamentos de mecánica cuántica

4. **Schleich, W. P. (2001)**. *Quantum Optics in Phase Space*. Wiley-VCH.
   - Representación en espacio de fase

### 9.2 Recursos Online

- **Qiskit Textbook**: https://qiskit.org/textbook/
- **QuTiP Documentation**: https://qutip.org/
- **Pygame Documentation**: https://www.pygame.org/docs/

### 9.3 Artículos Científicos

1. Glauber, R. J. (1963). "Coherent and Incoherent States of the Radiation Field". *Physical Review*, 131(6), 2766.

2. Loudon, R., & Knight, P. L. (1987). "Squeezed Light". *Journal of Modern Optics*, 34(6-7), 709-759.

3. Mandel, L. (1979). "Sub-Poissonian photon statistics in resonance fluorescence". *Optics Letters*, 4(7), 205-207.

---

## Apéndice A: Unidades y Constantes

En este proyecto se utilizan **unidades naturales** donde:
- ℏ = 1 (constante de Planck reducida)
- m = 1 (masa del oscilador)
- ω = 1 (frecuencia angular por defecto)

### Conversión a Unidades SI:

Para un oscilador real con frecuencia ω₀:
- **Longitud**: x_SI = x_adim × √(ℏ/mω₀)
- **Momento**: p_SI = p_adim × √(ℏmω₀)
- **Energía**: E_SI = E_adim × ℏω₀
- **Tiempo**: t_SI = t_adim / ω₀

---

## Apéndice B: Troubleshooting

### Problema: La simulación va muy lenta
**Solución**: Desactivar el guardado automático (tecla G) o reducir max_trail

### Problema: Los CSV son muy grandes
**Solución**: Ejecutar simulaciones más cortas o diezmar los datos post-procesamiento

### Problema: Error al importar scipy
**Solución**: Instalar con `pip install scipy`

### Problema: El estado no se mantiene al desactivar fuerza (Green)
**Solución**: Esto es correcto, el estado persiste y solo rota libremente

---

## Contacto y Contribuciones

**Autor**: Carlos  
**Proyecto**: Simulación QHO Interactiva  
**Fecha**: Noviembre 2025  
**Versión**: 2.0 (6 pestañas)

Para reportar bugs o sugerir mejoras, por favor documentar:
1. Pestaña en la que ocurre
2. Pasos para reproducir
3. Comportamiento esperado vs. observado
4. Configuración de parámetros

---

**¡Gracias por usar esta simulación!** 🚀✨