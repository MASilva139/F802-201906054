# Guía Práctica: Función de Green del QHO

## 🔬 Ecuaciones Implementadas en el Código

### 1. Función `alpha_analitico(t)` - Caso No Resonante

```python
factor = F0 / (√2 * (ω² - ν²))
alpha_forced = factor * cos(νt)      # Componente a frecuencia ν
alpha_free = -factor * cos(ωt)       # Componente a frecuencia ω
alpha(t) = (alpha_forced + alpha_free) * exp(-iωt)
```

**Forma expandida:**
```
α(t) = (F₀/√2) * [cos(νt) - cos(ωt)]/(ω² - ν²) * exp(-iωt)
```

### 2. Valores Esperados de Posición y Momento

```python
⟨X⟩(t) = √2 * Re[α(t)]
⟨P⟩(t) = √2 * Im[α(t)]
```

**Forma explícita:**
```
⟨X⟩(t) = √2 * Re[(F₀/√2) * [cos(νt) - cos(ωt)]/(ω² - ν²) * exp(-iωt)]
       = F₀ * Re{[cos(νt) - cos(ωt)]/(ω² - ν²) * [cos(ωt) - i·sin(ωt)]}
       = F₀/(ω² - ν²) * [cos(νt)cos(ωt) - cos²(ωt)]
```

### 3. Matriz de Covarianza Rotante

```python
c, s = cos(ωt), sin(ωt)
R = [[c, -s],
     [s,  c]]
     
Σ(t) = R @ Σ₀ @ Rᵀ
```

Para estado fundamental **Σ₀ = 0.5·I**:
```
Σ(t) = 0.5 * [[cos²(ωt) + sin²(ωt),  (cos²(ωt) - sin²(ωt))/2],
              [(cos²(ωt) - sin²(ωt))/2,  cos²(ωt) + sin²(ωt)]]
              
     = 0.5 * [[1,        sin(2ωt)/2],
              [sin(2ωt)/2,        1]]
```

### 4. Energía Total

```python
X² = Σ_XX + ⟨X⟩²
P² = Σ_PP + ⟨P⟩²
E = (P² + ω²X²)/2
```

Para estado coherente:
```
E(t) = ℏω(|α(t)|² + 1/2)
```

### 5. Trabajo Instantáneo

```python
W_inst(t) = F(t) * ⟨X⟩(t)
         = F₀cos(νt) * [F₀/(ω² - ν²)] * [cos(νt)cos(ωt) - cos²(ωt)]
```

### 6. Potencia Entregada

```python
P(t) = F(t) * ⟨P⟩(t)
     = F₀cos(νt) * √2 * Im[α(t)]
```

---

## 🎮 Experimentos Sugeridos

### Experimento 1: Respuesta Fuera de Resonancia

**Parámetros:**
- ω = 1.0
- ν = 0.7
- F₀ = 0.8

**Qué observar:**
- Trayectoria estable con dos frecuencias
- Batimientos en el espacio de fase
- Energía oscilante pero acotada

**Análisis FFT esperado:**
- Pico en f = ω/(2π) ≈ 0.159 Hz
- Pico en f = ν/(2π) ≈ 0.111 Hz
- Picos de batimiento en |ω ± ν|/(2π)

### Experimento 2: Cerca de Resonancia

**Parámetros:**
- ω = 1.0
- ν = 0.95 (muy cerca de ω)
- F₀ = 0.5

**Qué observar:**
- Batimientos lentos con periodo T_beat = 2π/|ω - ν| ≈ 125.7 s
- Amplitud modulada sinusoidalmente
- Máximos locales cada T_beat/2

**Predicción teórica:**
```
|α(t)| ≈ (F₀/√2) * |sin((ω-ν)t/2)|/|ω-ν|
```

### Experimento 3: Cambio de Frecuencia en Tiempo Real

**Procedimiento:**
1. Iniciar con ν = 0.5
2. Presionar `+` gradualmente hasta ν → ω
3. Observar cómo la amplitud aumenta

**Comportamiento esperado:**
```
Amplitud máxima ∝ 1/|ω² - ν²|
```

A medida que ν → ω, la amplitud diverge.

### Experimento 4: Comparación de Métodos

**Procedimiento:**
1. Correr simulación con método analítico (M para cambiar)
2. Exportar datos
3. Cambiar a split-operator
4. Correr misma simulación
5. Comparar resultados en `analisis_qho.py`

**Diferencias esperadas:**
- Split-operator: pequeños errores numéricos O(dt²)
- Analítico: exacto hasta precisión de máquina

### Experimento 5: Amplitud de Fuerza Variable

**Procedimiento:**
1. F₀ = 0.1: Perturbación débil
2. F₀ = 1.0: Perturbación moderada
3. F₀ = 5.0: Perturbación fuerte

**Observar:**
- ⟨X⟩ y ⟨P⟩ escalan linealmente con F₀
- Incertezas (ΔX, ΔP) permanecen constantes
- Comportamiento siempre lineal (QHO es sistema lineal)

---

## 📊 Análisis de Datos Exportados

### Columnas Clave en el CSV

1. **`tiempo`**: Tiempo de simulación
2. **`X_avg`, `P_avg`**: Trayectoria en espacio de fase
3. **`alpha_real`, `alpha_imag`**: Componentes de α(t)
4. **`alpha_magnitud`**: |α(t)| = amplitud de desplazamiento
5. **`fuerza_externa`**: F(t) en cada instante
6. **`energia_total`**: E(t) = ⟨Ĥ⟩(t)
7. **`trabajo_instantaneo`**: W_inst = F·⟨X⟩
8. **`potencia`**: P = F·⟨P⟩

### Análisis FFT Recomendado

```python
import numpy as np
import pandas as pd
from scipy.fft import fft, fftfreq
import matplotlib.pyplot as plt

# Cargar datos
df = pd.read_csv('funcion_green_qho_YYYYMMDD_HHMMSS.csv')

# FFT de ⟨X⟩
dt = df['tiempo'].iloc[1] - df['tiempo'].iloc[0]
N = len(df)
X_fft = fft(df['X_avg'].values)
freqs = fftfreq(N, dt)

# Graficar espectro
plt.figure(figsize=(12, 6))
plt.plot(freqs[:N//2], np.abs(X_fft)[:N//2])
plt.xlabel('Frecuencia (Hz)')
plt.ylabel('|FFT(⟨X⟩)|')
plt.title('Espectro de Frecuencias')
plt.grid(True)
plt.show()
```

### Cálculo de Energía Transferida

```python
# Energía inicial
E0 = df['energia_total'].iloc[0]

# Energía final
Ef = df['energia_total'].iloc[-1]

# Energía transferida
ΔE = Ef - E0

# Trabajo total (integral numérica)
W_total = np.trapz(df['trabajo_instantaneo'], df['tiempo'])

print(f"ΔE = {ΔE:.6f}")
print(f"W = {W_total:.6f}")
print(f"Error: {abs(ΔE - W_total):.6e}")
```

### Verificar Teorema Trabajo-Energía

El teorema trabajo-energía establece:
```
ΔE = ∫ F(t)·v(t) dt = ∫ F(t)·⟨P⟩(t) dt
```

---

## 🔢 Fórmulas de Verificación

### 1. Principio de Incertidumbre

```python
assert all(df['producto_incerteza'] >= 0.49)  # Tolerancia numérica
```

**Siempre debe cumplirse:** ΔX·ΔP ≥ ℏ/2 = 0.5 (en unidades ℏ=1)

### 2. Conservación de Área

```python
area_teorica = np.pi
area_numerica = df['area_elipse'].mean()
assert abs(area_numerica - area_teorica) < 0.01
```

### 3. Pureza del Estado

```python
assert all(df['pureza'] > 0.99)  # Estado puro
```

Para estado coherente: **pureza = 1**

### 4. Solución Analítica vs Numérica

Si usas split-operator, compara con solución analítica:

```python
# Calcular α analítico
omega = df['omega_oscilador'].iloc[0]
nu = df['nu_frecuencia'].iloc[0]
F0 = df['F0_amplitud'].iloc[0]
t = df['tiempo'].values

factor = F0 / (np.sqrt(2) * (omega**2 - nu**2))
alpha_teorico = factor * (np.cos(nu*t) - np.cos(omega*t)) * np.exp(-1j*omega*t)

# Comparar con datos
alpha_numerico = df['alpha_real'].values + 1j*df['alpha_imag'].values
error = np.abs(alpha_teorico - alpha_numerico).mean()

print(f"Error promedio: {error:.6e}")
```

---

## 🎯 Casos Límite Importantes

### Caso 1: Fuerza Muy Débil (F₀ → 0)

```
α(t) → 0
⟨X⟩(t) → 0
⟨P⟩(t) → 0
```

Sistema permanece esencialmente en estado fundamental.

### Caso 2: Frecuencia Muy Alta (ν >> ω)

```
α(t) ≈ (F₀/√2) * cos(νt)/(ν²) * exp(-iωt)
```

Respuesta muy pequeña (denominador grande).

### Caso 3: Frecuencia Muy Baja (ν << ω)

```
α(t) ≈ -(F₀/√2) * cos(νt)/(ω²) * exp(-iωt)
```

Respuesta cuasi-estática.

### Caso 4: Fuerza Constante (ν = 0)

```
F(t) = F₀
α(t) = (F₀/√2ω²) * [1 - cos(ωt)] * exp(-iωt)
```

Oscilación alrededor de nueva posición de equilibrio.

---

## 🧪 Extensiones Experimentales

### Extensión 1: Pulso Cuadrado

Modificar `fuerza_externa(t)`:

```python
def fuerza_externa(self, t):
    if 5.0 < t < 10.0:
        return self.F0
    else:
        return 0.0
```

**Resultado esperado:** Excitación impulsiva del oscilador.

### Extensión 2: Chirp Lineal

```python
def fuerza_externa(self, t):
    nu_t = self.nu + 0.1 * t  # Frecuencia variable
    return self.F0 * np.cos(nu_t * t)
```

**Observar:** Resonancia cuando ν(t) = ω.

### Extensión 3: Fuerza Aleatoria

```python
def fuerza_externa(self, t):
    return self.F0 * np.random.randn()
```

**Requiere:** Split-operator (no hay solución analítica).

---

## 📈 Gráficas Recomendadas

### Gráfica 1: Retrato de Fase Completo

```python
plt.figure(figsize=(10, 10))
plt.plot(df['X_avg'], df['P_avg'], 'c-', linewidth=0.5)
plt.xlabel('⟨X⟩')
plt.ylabel('⟨P⟩')
plt.title('Espacio de Fase')
plt.axis('equal')
plt.grid(True)
```

### Gráfica 2: Energía vs Tiempo

```python
plt.figure(figsize=(12, 6))
plt.plot(df['tiempo'], df['energia_total'], 'g-', label='Total')
plt.plot(df['tiempo'], df['energia_cinetica'], 'b--', label='Cinética')
plt.plot(df['tiempo'], df['energia_potencial'], 'r--', label='Potencial')
plt.xlabel('Tiempo')
plt.ylabel('Energía')
plt.legend()
plt.grid(True)
```

### Gráfica 3: |α(t)| vs Tiempo

```python
plt.figure(figsize=(12, 6))
plt.plot(df['tiempo'], df['alpha_magnitud'], 'purple', linewidth=2)
plt.xlabel('Tiempo')
plt.ylabel('|α(t)|')
plt.title('Amplitud de Desplazamiento')
plt.grid(True)
```

### Gráfica 4: Fuerza y Respuesta

```python
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

ax1.plot(df['tiempo'], df['fuerza_externa'], 'r-')
ax1.set_ylabel('F(t)')
ax1.set_title('Fuerza Externa')
ax1.grid(True)

ax2.plot(df['tiempo'], df['X_avg'], 'b-')
ax2.set_xlabel('Tiempo')
ax2.set_ylabel('⟨X⟩(t)')
ax2.set_title('Respuesta del Sistema')
ax2.grid(True)
```

---

## 💡 Tips de Optimización

### 1. Resolución Temporal

- **dt = 0.016** (60 FPS): Bueno para visualización
- **dt = 0.001**: Mejor para análisis FFT preciso
- **dt < π/ω**: Criterio de Nyquist

### 2. Duración de Simulación

Para capturar batimientos:
```
T_min = 2π/|ω - ν|
```

Simular al menos 5-10 periodos de batimiento.

### 3. Tamaño de Base (Split-Operator)

Para energías E ≈ n·ℏω:
```
n_basis ≥ 2n + 5
```

Regla general: **n_basis = 20** es suficiente para F₀ < 2.

### 4. Condición de Estabilidad

Split-operator es incondicional estable, pero para precisión:
```
ω·dt < 0.1  ⟹  dt < 0.1/ω
```

---

## 🎓 Preguntas para Investigar

1. **¿Qué pasa si ω = ν exactamente?**
   - Implementar detección de resonancia
   - Comparar crecimiento lineal vs simulación

2. **¿Cómo afecta el estado inicial?**
   - Cambiar de |0⟩ a |1⟩ o |2⟩
   - Observar diferentes trayectorias

3. **¿Se puede observar interferencia cuántica?**
   - Usar superposición inicial
   - Mirar evolución de coherencias

4. **¿Cómo simular amortiguamiento?**
   - Agregar término γ·â a ecuación de movimiento
   - Observar decaimiento exponencial

5. **¿Qué pasa con fuerzas no-lineales?**
   - F(t) = F₀·X²: Requiere Kerr Hamiltonian
   - Split-operator es necesario

---

## 🏆 Desafío Final

**Objetivo:** Reproducir fenómeno de **resonancia paramétrica**

**Setup:**
- Modular la frecuencia: ω → ω(t) = ω₀(1 + ε·cos(2ω₀t))
- Observar amplificación exponencial cuando se cumple condición de resonancia

**Requiere:** Modificar Hamiltoniano en split-operator para incluir modulación de frecuencia.

---

Esta guía te da todas las herramientas para explorar a fondo la física de la función de Green y validar tus simulaciones experimentalmente. ¡Disfruta la exploración! 🚀
