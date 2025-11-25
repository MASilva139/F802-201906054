# SIMULACIONES DEL OSCILADOR ARMÓNICO CUÁNTICO
## Comparación de Estados en Espacio de Fase

---

## 🎯 OBJETIVO
Visualizar y comparar la evolución temporal de diferentes estados cuánticos vs. el oscilador clásico en el espacio de fase (X, P), mostrando las diferencias fundamentales entre mecánica cuántica y clásica.

---

## 📐 FUNDAMENTOS TEÓRICOS

### **Hamiltoniano del Oscilador Armónico**
$$\hat{H} = \omega\left(\hat{a}^\dagger\hat{a} + \frac{1}{2}\right) = \frac{\omega}{2}(\hat{X}^2 + \hat{P}^2)$$

### **Operadores de Cuadratura**
$$\hat{X} = \frac{\hat{a} + \hat{a}^\dagger}{\sqrt{2}}, \quad \hat{P} = \frac{\hat{a} - \hat{a}^\dagger}{i\sqrt{2}}$$

$$[\hat{X}, \hat{P}] = i \quad \Rightarrow \quad \Delta X \cdot \Delta P \geq \frac{1}{2}$$

### **Estados de Fock (Base energética)**
$$\hat{H}|n\rangle = \omega(n + 1/2)|n\rangle$$

---

## 1️⃣ ESTADO COHERENTE CUÁNTICO

### **Definición**
Estado más "clásico" del oscilador cuántico. Minimiza el principio de incertidumbre:

$$|\alpha\rangle = e^{-|\alpha|^2/2} \sum_{n=0}^{\infty} \frac{\alpha^n}{\sqrt{n!}}|n\rangle = \hat{D}(\alpha)|0\rangle$$

donde $\hat{D}(\alpha) = e^{\alpha\hat{a}^\dagger - \alpha^*\hat{a}}$ es el operador de desplazamiento.

### **Evolución Temporal**
$$|\alpha(t)\rangle = |\alpha_0 e^{-i\omega t}\rangle$$

El parámetro complejo $\alpha$ simplemente rota en el plano complejo.

### **Valores Esperados**
$$\langle X \rangle(t) = \sqrt{2}\text{Re}[\alpha(t)] = \sqrt{2}|\alpha_0|\cos(\omega t + \phi_0)$$
$$\langle P \rangle(t) = \sqrt{2}\text{Im}[\alpha(t)] = \sqrt{2}|\alpha_0|\sin(\omega t + \phi_0)$$

### **Incertezas (CONSTANTES)**
$$\Delta X = \Delta P = \frac{1}{\sqrt{2}} \quad \text{(estado de mínima incerteza)}$$

$$\Delta X \cdot \Delta P = \frac{1}{2} \quad \text{(límite cuántico)}$$

### **Matriz de Covarianza**
$$\Sigma = \frac{1}{2}\begin{pmatrix} 1 & 0 \\ 0 & 1 \end{pmatrix}$$

**Rotada en el tiempo:**
$$\Sigma(t) = R(\omega t) \Sigma_0 R(\omega t)^T = \Sigma_0 \quad \text{(¡círculo!)}$$

### **Características Visuales**
✓ Trayectoria **circular perfecta** en (X, P)
✓ Elipse de incerteza **rígida** (círculo que rota sin deformarse)
✓ Radio de trayectoria: $r = \sqrt{2}|\alpha_0|$
✓ Comportamiento más parecido al clásico

### **Parámetros de Simulación**
- $\omega = 1.0$ rad/s
- $|\alpha_0| = 1.5$
- $\phi_0 = \pi/6$

---

## 2️⃣ ESTADO COMPRIMIDO (SQUEEZED)

### **Definición**
Estado con incerteza **reducida** en una cuadratura a costa de aumentarla en la otra:

$$|r, \theta, \alpha\rangle = \hat{D}(\alpha)\hat{S}(r,\theta)|0\rangle$$

**Operador de squeezing:**
$$\hat{S}(r,\theta) = \exp\left[\frac{r}{2}(e^{-2i\theta}\hat{a}^2 - e^{2i\theta}\hat{a}^{\dagger 2})\right]$$

### **Incertezas (VARIABLES)**
$$\Delta X_\theta = e^{-r}/\sqrt{2} < \frac{1}{\sqrt{2}} \quad \text{(comprimida)}$$
$$\Delta P_\theta = e^{+r}/\sqrt{2} > \frac{1}{\sqrt{2}} \quad \text{(expandida)}$$

$$\Delta X_\theta \cdot \Delta P_\theta = \frac{1}{2} \quad \text{(sigue siendo mínima)}$$

### **Matriz de Covarianza Inicial**
$$\Sigma_0 = R(\theta) \begin{pmatrix} e^{2r} & 0 \\ 0 & e^{-2r} \end{pmatrix} R(\theta)^T$$

donde $R(\theta) = \begin{pmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{pmatrix}$

### **Evolución Temporal (ROTACIÓN)**
$$\Sigma(t) = R(\omega t) \Sigma_0 R(\omega t)^T$$

**Ángulo de la elipse:**
$$\theta_{\text{elipse}}(t) = \theta_0 + \omega t$$

La elipse **rota** con periodo $T = \pi/\omega$

### **Squeezing en Decibelios**
$$S_{dB} = -10\log_{10}(\Delta X_{\min}) = 10r\log_{10}(e) \approx 4.34r \text{ dB}$$

### **Características Visuales**
✓ Trayectoria **circular** en valores esperados (como coherente)
✓ Elipse de incerteza **rígida pero alargada**
✓ **Rotación visible** de la elipse (periodo π/ω)
✓ Incerteza oscila entre ejes: $\Delta X(t)$ y $\Delta P(t)$ intercambian valores

### **Parámetros de Simulación**
- $r = 0.7$ (squeezing parameter)
- $\theta_0 = \pi/4$ (ángulo inicial)
- $|\alpha_0| = 1.5$

---

## 3️⃣ SUPERPOSICIÓN DE ESTADOS DE FOCK

### **Definición**
Superposición discreta de autoestados de energía:

$$|\psi\rangle = \sum_{n=0}^{N} c_n |n\rangle, \quad \sum_n |c_n|^2 = 1$$

### **Evolución Temporal**
$$|\psi(t)\rangle = \sum_{n=0}^{N} c_n e^{-i\omega(n+1/2)t}|n\rangle$$

Cada nivel evoluciona con **frecuencia diferente**: $\omega_n = \omega(n + 1/2)$

### **Valores Esperados**
$$\langle X \rangle = \sum_{n,m} c_n^* c_m \langle n|\hat{X}|m\rangle e^{-i\omega(m-n)t}$$

**Elementos de matriz:**
$$\langle n|\hat{X}|m\rangle = \frac{1}{\sqrt{2}}\left(\sqrt{m}\delta_{n,m-1} + \sqrt{n+1}\delta_{n,m+1}\right)$$

$$\langle n|\hat{P}|m\rangle = \frac{i}{\sqrt{2}}\left(\sqrt{n+1}\delta_{n,m+1} - \sqrt{m}\delta_{n,m-1}\right)$$

### **Matriz de Covarianza (DINÁMICA)**
$$\Sigma_{XX}(t) = \langle X^2 \rangle(t) - \langle X \rangle(t)^2$$
$$\Sigma_{PP}(t) = \langle P^2 \rangle(t) - \langle P \rangle(t)^2$$
$$\Sigma_{XP}(t) = \langle XP \rangle(t) - \langle X \rangle(t)\langle P \rangle(t)$$

**¡La matriz completa cambia con el tiempo!**

### **Probabilidades de Fock**
$$P_n(t) = |c_n|^2 \quad \text{(constantes en el tiempo)}$$

### **Coherencias Cuánticas**
$$\rho_{nm}(t) = c_n c_m^* e^{-i\omega(n-m)t}$$

Oscilan con frecuencias $\omega(n-m)$

### **Características Visuales**
✓ Trayectoria **compleja** (no circular)
✓ Elipse de incerteza **SE DEFORMA** continuamente
✓ $\Delta X(t)$ y $\Delta P(t)$ varían de forma **no-periódica simple**
✓ La elipse cambia de forma, orientación y tamaño
✓ Comportamiento **genuinamente cuántico**

### **Parámetros de Simulación**
- Estados: $|0\rangle, |1\rangle, |2\rangle, |3\rangle, |4\rangle, |5\rangle$
- Coeficientes: $c = [0.5, 0.5, 0.3, 0.2, 0.1, 0.1]$ (normalizados)

---

## 4️⃣ OSCILADOR CLÁSICO

### **Ecuaciones de Movimiento**
$$x(t) = A\cos(\omega t + \phi_0)$$
$$v(t) = \dot{x}(t) = -A\omega\sin(\omega t + \phi_0)$$

### **Espacio de Fase**
Trayectoria en el plano $(x, v)$:
$$x^2 + \frac{v^2}{\omega^2} = A^2 \quad \text{(elipse → círculo si } \omega=1\text{)}$$

### **Energía (Conservada)**
$$E = \frac{1}{2}(v^2 + \omega^2 x^2) = \frac{1}{2}\omega^2 A^2 = \text{constante}$$

$$T = \frac{v^2}{2}, \quad V = \frac{\omega^2 x^2}{2}$$

### **Características Visuales**
✓ **Punto material** (sin incerteza cuántica)
✓ Trayectoria **circular perfecta**
✓ Radio: $A$
✓ Periodo: $T = 2\pi/\omega$
✓ Sistema determinista (sin fluctuaciones)

### **Parámetros de Simulación**
- $A = 3.0$ (amplitud)
- $\omega = 1.0$ rad/s
- $\phi_0 = \pi/6$

---

## 🔄 COMPARACIÓN DE TRAYECTORIAS

| Estado | Valores ⟨X⟩, ⟨P⟩ | Incertezas ΔX, ΔP | Elipse |
|--------|------------------|-------------------|--------|
| **Coherente** | Circular | Constantes (1/√2) | Círculo rígido que rota |
| **Comprimido** | Circular | Oscilan entre ejes | Elipse rígida que rota |
| **Superposición** | Compleja | Variables dinámicas | Se DEFORMA continuamente |
| **Clásico** | Circular | ❌ Sin incerteza | Punto (sin extensión) |

---

## 🎨 VISUALIZACIÓN EN ESPACIO DE FASE

### **Coordenadas Adimensionales**
$$X = \frac{x}{x_0}, \quad P = \frac{p}{p_0}$$

donde $x_0 = \sqrt{\hbar/(m\omega)}$ y $p_0 = \sqrt{m\hbar\omega}$

### **Elipse de Incerteza**
Representación visual de la matriz de covarianza $\Sigma$:

**Autovalores:** $\lambda_1, \lambda_2$ → semiejes $a = \sqrt{\lambda_1}$, $b = \sqrt{\lambda_2}$

**Autovectores:** orientación de la elipse

**Puntos de la elipse:**
$$\begin{pmatrix} X \\ P \end{pmatrix} = \begin{pmatrix} \langle X \rangle \\ \langle P \rangle \end{pmatrix} + R(\theta) \begin{pmatrix} a\cos\phi \\ b\sin\phi \end{pmatrix}$$

con $\phi \in [0, 2\pi]$

### **Área de la Elipse (Liouville)**
Para estados puros:
$$A = \pi\sqrt{\det(\Sigma)} = \pi \cdot \frac{1}{2} = \frac{\pi}{2}$$

---

## ⚡ PRINCIPIOS VERIFICADOS

### **1. Principio de Incertidumbre de Heisenberg**
$$\boxed{\Delta X \cdot \Delta P \geq \frac{1}{2}}$$

Todos los estados cuánticos lo cumplen en todo momento.

### **2. Conservación de Energía**
$$E = \frac{\langle P^2 \rangle + \omega^2\langle X^2 \rangle}{2} = \text{constante}$$

### **3. Teorema de Liouville**
El área en espacio de fase se conserva:
$$\frac{dA}{dt} = 0$$

---

## 🔬 IMPLEMENTACIÓN

**Tecnología:** Python + Pygame (visualización en tiempo real)

**Parámetros globales:**
- $\omega = 1.0$ rad/s
- $dt = 0.016$ s (60 FPS)
- Integración temporal: trapezoidal

**Características:**
- 4 simulaciones simultáneas con pestañas
- Exportación de datos a CSV (40-70 observables)
- Visualización interactiva en tiempo real

---

## 📊 CONCLUSIONES VISUALES

### **Estado Coherente**
- Comportamiento más "clásico"
- Incerteza mínima pero constante
- Elipse rígida (círculo)

### **Estado Comprimido**
- Incerteza reducida en una dirección
- Útil para metrología de precisión
- Rotación visible de elipse

### **Superposición de Fock**
- Comportamiento genuinamente cuántico
- Deformación continua de incertezas
- Interferencia entre niveles energéticos

### **Oscilador Clásico**
- Referencia determinista
- Sin incerteza cuántica
- Límite $\hbar \to 0$

---

**Las diferencias entre mecánica cuántica y clásica son visibles en tiempo real en el espacio de fase.**