import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.animation as animation
from pathlib import Path
import sys
import os
from datetime import datetime
from scipy.optimize import curve_fit
from scipy.fft import fft, fftfreq
from scipy.signal import find_peaks, welch

# =============================================================================
# CONFIGURACIÓN GLOBAL
# =============================================================================

plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['figure.figsize'] = (14, 10)
plt.rcParams['font.size'] = 11
plt.rcParams['lines.linewidth'] = 2
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12

# Carpeta base de resultados
RESULTADOS_DIR = Path('Proyecto/Final/resultados1')
FIGURAS_DIR = RESULTADOS_DIR / 'figuras'
SALIDAS_DIR = RESULTADOS_DIR / 'salidas_txt'
COMPARACIONES_DIR = RESULTADOS_DIR / 'comparaciones'
ANIMACIONES_DIR = RESULTADOS_DIR / 'animaciones'
FFT_DIR = FIGURAS_DIR / 'fft'
AJUSTES_DIR = FIGURAS_DIR / 'ajustes'

GENERAR_ANIMACION_AUTO = False

# Colores consistentes para cada tipo de simulación
COLORES_SIMULACION = {
    'estado_coherente': '#1f77b4',      # Azul
    'estado_comprimido': '#2ca02c',     # Verde
    'superposicion_fock': '#9467bd',    # Púrpura
    'oscilador_clasico': '#d62728',     # Rojo
    'green_split_operator': '#00bcd4',  # Cyan
    'green_analitico': '#ff7f0e'        # Naranja
}

NOMBRES_SIMULACION = {
    'estado_coherente': 'Estado Coherente',
    'estado_comprimido': 'Estado Comprimido',
    'superposicion_fock': 'Superposición Fock',
    'oscilador_clasico': 'Oscilador Clásico',
    'green_split_operator': 'Green Split-Operator',
    'green_analitico': 'Green Analítico'
}


def crear_directorios():
    """Crea la estructura de directorios para resultados"""
    dirs = [RESULTADOS_DIR, FIGURAS_DIR, SALIDAS_DIR, 
            COMPARACIONES_DIR, ANIMACIONES_DIR, FFT_DIR, AJUSTES_DIR]
    
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
    
    print(f"📁 Estructura de directorios creada en: {RESULTADOS_DIR.absolute()}")


# =============================================================================
# CLASE PRINCIPAL: AnalizadorQHO
# =============================================================================

class AnalizadorQHO:
    """
    Clase para analizar datos de simulaciones QHO
    Soporta todos los tipos de simulación del Proyecto.py
    """
    
    def __init__(self, archivo_csv):
        """
        Inicializa el analizador con un archivo CSV
        
        Args:
            archivo_csv (str): Ruta al archivo CSV
        """
        self.archivo = Path(archivo_csv)
        self.df = pd.read_csv(archivo_csv)
        self.nombre = self.archivo.stem
        self.tipo = self._detectar_tipo()
        
        # Inicializar atributos de figuras
        self.fig_principal = None
        self.fig_fft = None
        self.fig_ajuste = None
        
        print(f"\n{'='*70}")
        print(f"📊 Analizando: {self.nombre}")
        print(f"   Tipo detectado: {NOMBRES_SIMULACION.get(self.tipo, self.tipo)}")
        print(f"   Puntos de datos: {len(self.df)}")
        print(f"   Tiempo total: {self.df['tiempo'].max():.2f} s")
        print(f"{'='*70}\n")
    
    def _detectar_tipo(self):
        """
        Detecta el tipo de simulación basándose en las columnas del CSV
        VERSIÓN MEJORADA con múltiples criterios de detección
        """
        columnas = set(self.df.columns)
        nombre_lower = self.nombre.lower()
        
        # =====================================================================
        # DETECCIÓN POR NOMBRE DE ARCHIVO (más confiable)
        # =====================================================================
        if 'comprimido' in nombre_lower or 'squeezed' in nombre_lower or 'squeeze' in nombre_lower:
            print("   📌 Detectado por nombre: Estado Comprimido")
            return 'estado_comprimido'
        
        if 'coherente' in nombre_lower and 'comprimido' not in nombre_lower:
            print("   📌 Detectado por nombre: Estado Coherente")
            return 'estado_coherente'
        
        if 'superposicion' in nombre_lower or 'fock' in nombre_lower:
            print("   📌 Detectado por nombre: Superposición Fock")
            return 'superposicion_fock'
        
        if 'clasico' in nombre_lower or 'classical' in nombre_lower:
            print("   📌 Detectado por nombre: Oscilador Clásico")
            return 'oscilador_clasico'
        
        if 'green' in nombre_lower:
            if 'split' in nombre_lower or 'numerico' in nombre_lower:
                print("   📌 Detectado por nombre: Green Split-Operator")
                return 'green_split_operator'
            else:
                print("   📌 Detectado por nombre: Green Analítico")
                return 'green_analitico'
        
        # =====================================================================
        # DETECCIÓN POR COLUMNAS ESPECÍFICAS
        # =====================================================================
        
        # Green's functions tienen columna 'metodo'
        if 'metodo' in columnas:
            if 'n_basis' in columnas:
                print("   📌 Detectado por columnas: Green Split-Operator")
                return 'green_split_operator'
            else:
                print("   📌 Detectado por columnas: Green Analítico")
                return 'green_analitico'
        
        # Fuerza externa sin método = Green analítico
        if 'fuerza_externa' in columnas and 'metodo' not in columnas:
            if 'n_basis' in columnas:
                print("   📌 Detectado por columnas: Green Split-Operator")
                return 'green_split_operator'
            print("   📌 Detectado por columnas: Green Analítico")
            return 'green_analitico'
        
        # Superposición tiene probabilidades de Fock
        if 'prob_n0' in columnas:
            print("   📌 Detectado por columnas: Superposición Fock")
            return 'superposicion_fock'
        
        # Oscilador clásico tiene posición/velocidad/aceleración
        if 'aceleracion' in columnas or ('posicion' in columnas and 'velocidad' in columnas):
            print("   📌 Detectado por columnas: Oscilador Clásico")
            return 'oscilador_clasico'
        
        # =====================================================================
        # DETECCIÓN DE SQUEEZING (múltiples métodos)
        # =====================================================================
        
        # Método 1: Columna explícita squeezing_r
        if 'squeezing_r' in columnas:
            print("   📌 Detectado por squeezing_r: Estado Comprimido")
            return 'estado_comprimido'
        
        # Método 2: Columna squeezing_dB con valores significativos
        if 'squeezing_dB' in columnas:
            sq_db_mean = abs(self.df['squeezing_dB'].mean())
            if sq_db_mean > 0.5:  # Más de 0.5 dB de squeezing
                print(f"   📌 Detectado por squeezing_dB ({sq_db_mean:.2f} dB): Estado Comprimido")
                return 'estado_comprimido'
        
        # Método 3: Anisotropía en matriz de covarianza
        if 'Sigma_XX' in columnas and 'Sigma_PP' in columnas:
            # Calcular ratio de varianzas
            sigma_xx = self.df['Sigma_XX'].values
            sigma_pp = self.df['Sigma_PP'].values
            
            # Evitar división por cero
            with np.errstate(divide='ignore', invalid='ignore'):
                ratios = sigma_xx / (sigma_pp + 1e-10)
            
            # Para estado coherente: ratio ≈ 1 siempre
            # Para comprimido: ratio oscila significativamente
            ratio_std = np.std(ratios)
            ratio_range = np.max(ratios) - np.min(ratios)
            
            # También verificar si hay excentricidad significativa
            if 'excentricidad' in columnas:
                exc_mean = self.df['excentricidad'].mean()
                if exc_mean > 0.3:  # Elipse significativamente no circular
                    print(f"   📌 Detectado por excentricidad ({exc_mean:.3f}): Estado Comprimido")
                    return 'estado_comprimido'
            
            # Verificar por ratio de varianzas
            if ratio_range > 0.5 or ratio_std > 0.2:
                print(f"   📌 Detectado por anisotropía (rango={ratio_range:.3f}): Estado Comprimido")
                return 'estado_comprimido'
            
            # Método 4: Verificar si ΔX ≠ ΔP significativamente
            if 'delta_X' in columnas and 'delta_P' in columnas:
                dx = self.df['delta_X'].values
                dp = self.df['delta_P'].values
                
                # Para coherente: ΔX ≈ ΔP ≈ 1/√2 ≈ 0.707
                dx_mean = np.mean(dx)
                dp_mean = np.mean(dp)
                
                # Diferencia relativa entre incertezas
                diff_rel = abs(dx_mean - dp_mean) / (0.5 * (dx_mean + dp_mean) + 1e-10)
                
                # Si la diferencia es mayor al 10%, es squeezing
                if diff_rel > 0.1:
                    print(f"   📌 Detectado por ΔX≠ΔP (diff={diff_rel*100:.1f}%): Estado Comprimido")
                    return 'estado_comprimido'
                
                # Verificar también los valores extremos
                dx_range = np.max(dx) - np.min(dx)
                dp_range = np.max(dp) - np.min(dp)
                
                if dx_range > 0.1 or dp_range > 0.1:
                    print(f"   📌 Detectado por variación de ΔX/ΔP: Estado Comprimido")
                    return 'estado_comprimido'
        
        # Método 5: Verificar autovalores de Σ
        if 'lambda_1' in columnas and 'lambda_2' in columnas:
            l1 = self.df['lambda_1'].values
            l2 = self.df['lambda_2'].values
            
            # Para coherente: λ1 ≈ λ2 ≈ 0.5
            # Para comprimido: λ1 >> λ2 o viceversa
            ratio_eigenvalues = l1 / (l2 + 1e-10)
            
            if np.mean(ratio_eigenvalues) > 1.5 or np.mean(ratio_eigenvalues) < 0.67:
                print(f"   📌 Detectado por autovalores (ratio={np.mean(ratio_eigenvalues):.2f}): Estado Comprimido")
                return 'estado_comprimido'
        
        # =====================================================================
        # POR DEFECTO: Estado coherente si tiene alpha
        # =====================================================================
        if 'alpha_real' in columnas or 'alpha_magnitud' in columnas:
            print("   📌 Detectado por defecto: Estado Coherente")
            return 'estado_coherente'
        
        print("   ⚠️  Tipo no reconocido")
        return 'desconocido'
    
    def analisis_completo(self, guardar=True):
        """
        Realiza un análisis completo de la simulación
        
        Args:
            guardar (bool): Si True, guarda las figuras generadas
        """
        print("📊 Iniciando análisis completo...\n")
        
        # 1. Resumen estadístico
        self.resumen_estadistico()
        
        # 2. Verificaciones físicas
        self.verificar_principios_fisicos()
        
        # 3. Gráficas principales según el tipo
        metodos_analisis = {
            'estado_coherente': self.analizar_coherente,
            'estado_comprimido': self.analizar_comprimido,
            'superposicion_fock': self.analizar_superposicion,
            'oscilador_clasico': self.analizar_clasico,
            'green_split_operator': self.analizar_green_split,
            'green_analitico': self.analizar_green_analitico
        }
        
        if self.tipo in metodos_analisis:
            metodos_analisis[self.tipo]()
        else:
            print(f"⚠️  Tipo de simulación '{self.tipo}' no reconocido")
            self.analizar_generico()
        
        # 4. Guardar figura principal
        if guardar and self.fig_principal is not None:
            self._guardar_figura_principal()
        
        print("\n✅ Análisis completo finalizado!")
    
    def resumen_estadistico(self):
        """Imprime resumen estadístico de los datos"""
        print("📈 RESUMEN ESTADÍSTICO")
        print("-" * 70)
        
        if self.tipo == 'oscilador_clasico':
            self._resumen_clasico()
        else:
            self._resumen_cuantico()
        
        print("-" * 70 + "\n")
    
    def _resumen_cuantico(self):
        """Resumen para simulaciones cuánticas"""
        df = self.df
        
        print(f"\n🔵 Valores esperados:")
        if 'X_avg' in df.columns:
            print(f"  ⟨X⟩: {df['X_avg'].mean():.4f} ± {df['X_avg'].std():.4f}")
            print(f"  ⟨P⟩: {df['P_avg'].mean():.4f} ± {df['P_avg'].std():.4f}")
        
        print(f"\n🔵 Incertezas:")
        if 'delta_X' in df.columns:
            print(f"  ΔX: {df['delta_X'].mean():.4f} ± {df['delta_X'].std():.4f}")
            print(f"  ΔP: {df['delta_P'].mean():.4f} ± {df['delta_P'].std():.4f}")
            print(f"  ΔX·ΔP: {df['producto_incerteza'].mean():.4f} ± {df['producto_incerteza'].std():.4f}")
        
        print(f"\n🔵 Número de fotones:")
        if 'n_promedio' in df.columns:
            print(f"  ⟨n⟩: {df['n_promedio'].mean():.4f} ± {df['n_promedio'].std():.4f}")
        if 'mandel_Q' in df.columns:
            print(f"  Mandel Q: {df['mandel_Q'].mean():.4f} ± {df['mandel_Q'].std():.4f}")
        
        print(f"\n🔵 Información cuántica:")
        if 'pureza' in df.columns:
            print(f"  Pureza: {df['pureza'].mean():.6f} ± {df['pureza'].std():.6f}")
        if 'entropia' in df.columns:
            print(f"  Entropía: {df['entropia'].mean():.6f} ± {df['entropia'].std():.6f}")
        
        # Específico para estado comprimido
        if self.tipo == 'estado_comprimido':
            print(f"\n🔵 Squeezing:")
            if 'squeezing_dB' in df.columns:
                print(f"  Squeezing: {df['squeezing_dB'].mean():.2f} ± {df['squeezing_dB'].std():.2f} dB")
            elif 'Sigma_XX' in df.columns and 'Sigma_PP' in df.columns:
                min_var = np.minimum(df['Sigma_XX'], df['Sigma_PP']).mean()
                sq_dB = -10 * np.log10(min_var * 2 + 1e-10)
                print(f"  Squeezing estimado: {sq_dB:.2f} dB")
            if 'excentricidad' in df.columns:
                print(f"  Excentricidad: {df['excentricidad'].mean():.4f}")
        
        # Específico para Green's functions
        if self.tipo in ['green_split_operator', 'green_analitico']:
            print(f"\n🔵 Fuerza externa:")
            if 'fuerza_externa' in df.columns:
                print(f"  F(t) max: {df['fuerza_externa'].max():.4f}")
                print(f"  F(t) min: {df['fuerza_externa'].min():.4f}")
            if 'F0_amplitud' in df.columns:
                print(f"  F₀: {df['F0_amplitud'].iloc[0]:.4f}")
            if 'nu_frecuencia' in df.columns:
                print(f"  ν: {df['nu_frecuencia'].iloc[0]:.4f}")
            if 'omega_oscilador' in df.columns:
                print(f"  ω: {df['omega_oscilador'].iloc[0]:.4f}")
    
    def _resumen_clasico(self):
        """Resumen para oscilador clásico"""
        df = self.df
        
        print(f"\n🔴 Oscilador Clásico:")
        if 'amplitud' in df.columns:
            print(f"  Amplitud: {df['amplitud'].iloc[0]:.4f}")
        if 'omega' in df.columns:
            print(f"  ω: {df['omega'].iloc[0]:.4f}")
        if 'periodo' in df.columns:
            print(f"  Período: {df['periodo'].iloc[0]:.4f}")
        if 'energia_total' in df.columns:
            print(f"  Energía: {df['energia_total'].mean():.4f} ± {df['energia_total'].std():.6f}")
    
    def verificar_principios_fisicos(self):
        """Verifica que se cumplan principios físicos fundamentales"""
        print("🔬 VERIFICACIÓN DE PRINCIPIOS FÍSICOS")
        print("-" * 70)
        
        if self.tipo != 'oscilador_clasico':
            self._verificar_incertidumbre()
            self._verificar_area_liouville()
            self._verificar_pureza()
        
        self._verificar_energia()
        
        print("-" * 70 + "\n")
    
    def _verificar_incertidumbre(self):
        """Verifica principio de incertidumbre"""
        if 'producto_incerteza' not in self.df.columns:
            return
        
        min_producto = self.df['producto_incerteza'].min()
        print(f"\n✓ Principio de incertidumbre:")
        print(f"  ΔX·ΔP mínimo = {min_producto:.6f}")
        
        if min_producto >= 0.49:
            print(f"  ✅ CUMPLE (≥ 0.5)")
        else:
            print(f"  ❌ NO CUMPLE (debería ser ≥ 0.5)")
    
    def _verificar_area_liouville(self):
        """Verifica conservación de área (Liouville)"""
        if 'area_elipse' not in self.df.columns:
            return
        
        area_media = self.df['area_elipse'].mean()
        area_std = self.df['area_elipse'].std()
        
        print(f"\n✓ Conservación de área (Liouville):")
        print(f"  Área = {area_media:.6f} ± {area_std:.6f}")
        
        if area_std / (area_media + 1e-10) < 0.01:
            print(f"  ✅ ÁREA CONSERVADA (variación < 1%)")
        else:
            print(f"  ⚠️  Variación: {area_std/(area_media+1e-10)*100:.2f}%")
    
    def _verificar_pureza(self):
        """Verifica pureza de estado"""
        if 'pureza' not in self.df.columns:
            return
        
        pureza_media = self.df['pureza'].mean()
        pureza_std = self.df['pureza'].std()
        
        print(f"\n✓ Estado puro:")
        print(f"  Pureza = {pureza_media:.6f} ± {pureza_std:.6f}")
        
        if pureza_media > 0.99:
            print(f"  ✅ Estado puro (≈ 1)")
        elif pureza_media > 0.9:
            print(f"  ⚠️  Estado casi puro")
        else:
            print(f"  ⚠️  Estado mixto (pureza < 0.9)")
    
    def _verificar_energia(self):
        """Verifica conservación de energía"""
        if 'energia_total' not in self.df.columns:
            return
        
        E_media = self.df['energia_total'].mean()
        E_std = self.df['energia_total'].std()
        E_var_rel = (E_std / (E_media + 1e-10)) * 100
        
        print(f"\n✓ Conservación de energía:")
        print(f"  E = {E_media:.6f} ± {E_std:.6f}")
        print(f"  Variación relativa: {E_var_rel:.4f}%")
        
        if self.tipo in ['green_split_operator', 'green_analitico']:
            if 'force_activa' in self.df.columns:
                drive_on = self.df['force_activa'].sum() > 0
                if drive_on:
                    print(f"  ℹ️  Sistema con fuerza externa (energía no conservada)")
                    return
        
        if E_var_rel < 1.0:
            print(f"  ✅ CONSERVADA (< 1%)")
        else:
            print(f"  ⚠️  Variación significativa")
    
    # =========================================================================
    # MÉTODOS DE ANÁLISIS ESPECÍFICOS POR TIPO
    # =========================================================================
    
    def analizar_coherente(self):
        """Análisis específico para estado coherente"""
        print("📊 Generando gráficas para Estado Coherente...")
        
        fig = plt.figure(figsize=(18, 14))
        fig.suptitle('Estado Coherente Cuántico', fontsize=16, fontweight='bold', 
                    color=COLORES_SIMULACION['estado_coherente'])
        
        df = self.df
        color_principal = COLORES_SIMULACION['estado_coherente']
        
        # 1. Trayectoria en espacio de fase
        ax1 = plt.subplot(3, 3, 1)
        scatter = ax1.scatter(df['X_avg'], df['P_avg'], 
                             c=df['tiempo'], cmap='viridis', s=5, alpha=0.7)
        ax1.plot(df['X_avg'].iloc[0], df['P_avg'].iloc[0], 'go', 
                markersize=12, label='Inicio', zorder=5)
        ax1.plot(df['X_avg'].iloc[-1], df['P_avg'].iloc[-1], 'r*', 
                markersize=15, label='Fin', zorder=5)
        ax1.set_xlabel('⟨X⟩')
        ax1.set_ylabel('⟨P⟩')
        ax1.set_title('Trayectoria en Espacio de Fase')
        ax1.legend(loc='upper right')
        ax1.grid(True, alpha=0.3)
        ax1.axis('equal')
        plt.colorbar(scatter, ax=ax1, label='Tiempo (s)')
        
        # 2. Evolución temporal de cuadraturas
        ax2 = plt.subplot(3, 3, 2)
        ax2.plot(df['tiempo'], df['X_avg'], color=color_principal, label='⟨X⟩', linewidth=1.5)
        ax2.plot(df['tiempo'], df['P_avg'], 'r-', label='⟨P⟩', linewidth=1.5)
        ax2.set_xlabel('Tiempo (s)')
        ax2.set_ylabel('Valor esperado')
        ax2.set_title('Cuadraturas vs Tiempo')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Incertezas
        ax3 = plt.subplot(3, 3, 3)
        ax3.plot(df['tiempo'], df['delta_X'], color=color_principal, label='ΔX', linewidth=1.5)
        ax3.plot(df['tiempo'], df['delta_P'], 'r-', label='ΔP', linewidth=1.5)
        ax3.axhline(y=1/np.sqrt(2), color='g', linestyle='--', 
                   label=f'Teórico = {1/np.sqrt(2):.4f}', alpha=0.7)
        ax3.set_xlabel('Tiempo (s)')
        ax3.set_ylabel('Incerteza')
        ax3.set_title('Incertezas (Estado Mínimo)')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Producto de incertezas
        ax4 = plt.subplot(3, 3, 4)
        ax4.plot(df['tiempo'], df['producto_incerteza'], 'purple', linewidth=1.5)
        ax4.axhline(y=0.5, color='r', linestyle='--', 
                   label='Límite Heisenberg = 0.5', alpha=0.7)
        ax4.set_xlabel('Tiempo (s)')
        ax4.set_ylabel('ΔX·ΔP')
        ax4.set_title('Principio de Incertidumbre')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        ax4.set_ylim(0.4, max(0.6, df['producto_incerteza'].max() * 1.1))
        
        # 5. Número de fotones
        ax5 = plt.subplot(3, 3, 5)
        ax5.plot(df['tiempo'], df['n_promedio'], 'orange', linewidth=1.5)
        ax5.set_xlabel('Tiempo (s)')
        ax5.set_ylabel('⟨n⟩')
        ax5.set_title('Número Promedio de Fotones')
        ax5.grid(True, alpha=0.3)
        
        # 6. Parámetro de Mandel Q
        ax6 = plt.subplot(3, 3, 6)
        ax6.plot(df['tiempo'], df['mandel_Q'], 'brown', linewidth=1.5)
        ax6.axhline(y=0, color='k', linestyle='--', alpha=0.5, label='Poisson (Q=0)')
        ax6.fill_between(df['tiempo'], 0, df['mandel_Q'], 
                        where=df['mandel_Q'] < 0, alpha=0.3, color='blue', 
                        label='Sub-Poisson')
        ax6.fill_between(df['tiempo'], 0, df['mandel_Q'], 
                        where=df['mandel_Q'] > 0, alpha=0.3, color='red', 
                        label='Super-Poisson')
        ax6.set_xlabel('Tiempo (s)')
        ax6.set_ylabel('Mandel Q')
        ax6.set_title('Estadística de Fotones')
        ax6.legend(loc='best', fontsize=8)
        ax6.grid(True, alpha=0.3)
        
        # 7. Energía
        ax7 = plt.subplot(3, 3, 7)
        ax7.plot(df['tiempo'], df['energia_total'], 'g-', 
                label='Total', linewidth=2)
        ax7.plot(df['tiempo'], df['energia_cinetica'], 'b--', 
                alpha=0.7, label='Cinética', linewidth=1.5)
        ax7.plot(df['tiempo'], df['energia_potencial'], 'r--', 
                alpha=0.7, label='Potencial', linewidth=1.5)
        ax7.set_xlabel('Tiempo (s)')
        ax7.set_ylabel('Energía')
        ax7.set_title('Conservación de Energía')
        ax7.legend()
        ax7.grid(True, alpha=0.3)
        
        # 8. Pureza y Entropía
        ax8 = plt.subplot(3, 3, 8)
        ax8_twin = ax8.twinx()
        l1 = ax8.plot(df['tiempo'], df['pureza'], 'b-', 
                     label='Pureza', linewidth=1.5)
        l2 = ax8_twin.plot(df['tiempo'], df['entropia'], 'r-', 
                          label='Entropía', linewidth=1.5)
        ax8.set_xlabel('Tiempo (s)')
        ax8.set_ylabel('Pureza', color='b')
        ax8_twin.set_ylabel('Entropía', color='r')
        ax8.tick_params(axis='y', labelcolor='b')
        ax8_twin.tick_params(axis='y', labelcolor='r')
        ax8.set_title('Pureza y Entropía')
        lines = l1 + l2
        labels = [l.get_label() for l in lines]
        ax8.legend(lines, labels, loc='best')
        ax8.grid(True, alpha=0.3)
        
        # 9. Parámetro α en plano complejo
        ax9 = plt.subplot(3, 3, 9, projection='polar')
        theta = df['alpha_fase'].values
        r = df['alpha_magnitud'].values
        scatter = ax9.scatter(theta, r, c=df['tiempo'], cmap='plasma', s=8, alpha=0.7)
        ax9.set_title('α en Plano Complejo')
        plt.colorbar(scatter, ax=ax9, label='Tiempo (s)', pad=0.1)
        
        plt.tight_layout()
        self.fig_principal = fig
    
    def analizar_comprimido(self):
        """Análisis específico para estado comprimido (squeezed)"""
        print("📊 Generando gráficas para Estado Comprimido...")
        
        fig = plt.figure(figsize=(18, 14))
        fig.suptitle('Estado Coherente Comprimido (Squeezed)', fontsize=16, fontweight='bold',
                    color=COLORES_SIMULACION['estado_comprimido'])
        
        df = self.df
        color_principal = COLORES_SIMULACION['estado_comprimido']
        
        # 1. Trayectoria en espacio de fase
        ax1 = plt.subplot(3, 3, 1)
        scatter = ax1.scatter(df['X_avg'], df['P_avg'], 
                             c=df['tiempo'], cmap='Greens', s=5, alpha=0.7)
        ax1.plot(df['X_avg'].iloc[0], df['P_avg'].iloc[0], 'go', 
                markersize=12, label='Inicio', zorder=5)
        ax1.set_xlabel('⟨X⟩')
        ax1.set_ylabel('⟨P⟩')
        ax1.set_title('Trayectoria en Espacio de Fase')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.axis('equal')
        plt.colorbar(scatter, ax=ax1, label='Tiempo (s)')
        
        # 2. Incertezas (squeezing)
        ax2 = plt.subplot(3, 3, 2)
        ax2.plot(df['tiempo'], df['delta_X'], color=color_principal, label='ΔX', linewidth=1.5)
        ax2.plot(df['tiempo'], df['delta_P'], 'r-', label='ΔP', linewidth=1.5)
        ax2.axhline(y=1/np.sqrt(2), color='gray', linestyle='--', 
                   label='Estado coherente', alpha=0.7)
        ax2.set_xlabel('Tiempo (s)')
        ax2.set_ylabel('Incerteza')
        ax2.set_title('Incertezas (Squeezing Visible)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Squeezing en dB
        ax3 = plt.subplot(3, 3, 3)
        if 'squeezing_dB' in df.columns:
            sq_dB = df['squeezing_dB']
        else:
            min_var = np.minimum(df['Sigma_XX'], df['Sigma_PP'])
            sq_dB = -10 * np.log10(min_var * 2 + 1e-10)
        ax3.plot(df['tiempo'], sq_dB, color=color_principal, linewidth=2)
        ax3.axhline(y=0, color='k', linestyle='--', alpha=0.5, label='Sin squeezing')
        ax3.set_xlabel('Tiempo (s)')
        ax3.set_ylabel('Squeezing (dB)')
        ax3.set_title('Nivel de Squeezing')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Ángulo de la elipse
        ax4 = plt.subplot(3, 3, 4)
        ax4.plot(df['tiempo'], np.degrees(df['theta_ellipse']), 'brown', linewidth=1.5)
        ax4.set_xlabel('Tiempo (s)')
        ax4.set_ylabel('Ángulo (grados)')
        ax4.set_title('Rotación de la Elipse de Incerteza')
        ax4.grid(True, alpha=0.3)
        
        # 5. Autovalores de Σ
        ax5 = plt.subplot(3, 3, 5)
        ax5.plot(df['tiempo'], df['lambda_1'], color=color_principal, label='λ₁ (mayor)', linewidth=1.5)
        ax5.plot(df['tiempo'], df['lambda_2'], 'r-', label='λ₂ (menor)', linewidth=1.5)
        ax5.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Estado coherente')
        ax5.set_xlabel('Tiempo (s)')
        ax5.set_ylabel('Autovalor')
        ax5.set_title('Autovalores de Matriz de Covarianza')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        # 6. Excentricidad
        ax6 = plt.subplot(3, 3, 6)
        ax6.plot(df['tiempo'], df['excentricidad'], color=color_principal, linewidth=1.5)
        ax6.set_xlabel('Tiempo (s)')
        ax6.set_ylabel('Excentricidad')
        ax6.set_title('Excentricidad de la Elipse')
        ax6.grid(True, alpha=0.3)
        ax6.set_ylim(0, 1)
        
        # 7. Estadística de Mandel
        ax7 = plt.subplot(3, 3, 7)
        ax7.plot(df['tiempo'], df['mandel_Q'], 'orange', linewidth=1.5)
        ax7.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        ax7.fill_between(df['tiempo'], 0, df['mandel_Q'], 
                        where=df['mandel_Q'] > 0, alpha=0.3, color='red', 
                        label='Super-Poisson')
        ax7.set_xlabel('Tiempo (s)')
        ax7.set_ylabel('Mandel Q')
        ax7.set_title('Estadística No-Clásica')
        ax7.legend()
        ax7.grid(True, alpha=0.3)
        
        # 8. Energía
        ax8 = plt.subplot(3, 3, 8)
        ax8.plot(df['tiempo'], df['energia_total'], color=color_principal, linewidth=2)
        ax8.set_xlabel('Tiempo (s)')
        ax8.set_ylabel('Energía Total')
        ax8.set_title('Conservación de Energía')
        ax8.grid(True, alpha=0.3)
        
        # 9. Área de la elipse (Liouville)
        ax9 = plt.subplot(3, 3, 9)
        ax9.plot(df['tiempo'], df['area_elipse'], color=color_principal, linewidth=1.5)
        ax9.axhline(y=np.pi/2, color='r', linestyle='--', 
                   label=f'Teórico = π/2 ≈ {np.pi/2:.4f}')
        ax9.set_xlabel('Tiempo (s)')
        ax9.set_ylabel('Área')
        ax9.set_title('Área de Elipse (Teorema de Liouville)')
        ax9.legend()
        ax9.grid(True, alpha=0.3)
        
        plt.tight_layout()
        self.fig_principal = fig
    
    def analizar_superposicion(self):
        """Análisis específico para superposición de estados de Fock"""
        print("📊 Generando gráficas para Superposición de Fock...")
        
        fig = plt.figure(figsize=(18, 16))
        fig.suptitle('Superposición de Estados de Fock', fontsize=16, fontweight='bold',
                    color=COLORES_SIMULACION['superposicion_fock'])
        
        df = self.df
        color_principal = COLORES_SIMULACION['superposicion_fock']
        
        # 1. Trayectoria en espacio de fase
        ax1 = plt.subplot(3, 3, 1)
        scatter = ax1.scatter(df['X_avg'], df['P_avg'], 
                             c=df['tiempo'], cmap='Purples', s=5, alpha=0.7)
        ax1.set_xlabel('⟨X⟩')
        ax1.set_ylabel('⟨P⟩')
        ax1.set_title('Trayectoria (color = tiempo)')
        ax1.grid(True, alpha=0.3)
        ax1.axis('equal')
        plt.colorbar(scatter, ax=ax1, label='Tiempo (s)')
        
        # 2. Incertezas variables
        ax2 = plt.subplot(3, 3, 2)
        ax2.plot(df['tiempo'], df['delta_X'], color=color_principal, label='ΔX', linewidth=1.5)
        ax2.plot(df['tiempo'], df['delta_P'], 'r-', label='ΔP', linewidth=1.5)
        ax2.set_xlabel('Tiempo (s)')
        ax2.set_ylabel('Incerteza')
        ax2.set_title('Incertezas (Deformación)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Probabilidades de Fock
        ax3 = plt.subplot(3, 3, 3)
        prob_cols = sorted([col for col in df.columns if col.startswith('prob_n')])
        colors = plt.cm.viridis(np.linspace(0, 1, len(prob_cols)))
        for col, color in zip(prob_cols, colors):
            n = col.replace('prob_n', '')
            ax3.plot(df['tiempo'], df[col], color=color, 
                    label=f'P(n={n})', linewidth=1.5)
        ax3.set_xlabel('Tiempo (s)')
        ax3.set_ylabel('Probabilidad')
        ax3.set_title('Evolución de P(n)')
        ax3.legend(loc='upper right', fontsize=8)
        ax3.grid(True, alpha=0.3)
        
        # 4. Producto de incertezas
        ax4 = plt.subplot(3, 3, 4)
        ax4.plot(df['tiempo'], df['producto_incerteza'], color=color_principal, linewidth=1.5)
        ax4.axhline(y=0.5, color='r', linestyle='--', label='Límite Heisenberg')
        ax4.set_xlabel('Tiempo (s)')
        ax4.set_ylabel('ΔX·ΔP')
        ax4.set_title('Principio de Incertidumbre')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # 5. Número de fotones
        ax5 = plt.subplot(3, 3, 5)
        ax5.plot(df['tiempo'], df['n_promedio'], 'orange', linewidth=1.5)
        ax5.set_xlabel('Tiempo (s)')
        ax5.set_ylabel('⟨n⟩')
        ax5.set_title('Número Promedio de Fotones')
        ax5.grid(True, alpha=0.3)
        
        # 6. Parámetro de Mandel
        ax6 = plt.subplot(3, 3, 6)
        ax6.plot(df['tiempo'], df['mandel_Q'], 'brown', linewidth=1.5)
        ax6.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        ax6.set_xlabel('Tiempo (s)')
        ax6.set_ylabel('Mandel Q')
        ax6.set_title('Estadística de Fotones')
        ax6.grid(True, alpha=0.3)
        
        # 7. Energía
        ax7 = plt.subplot(3, 3, 7)
        ax7.plot(df['tiempo'], df['energia_total'], color=color_principal, linewidth=2, label='Total')
        if 'energia_cinetica' in df.columns:
            ax7.plot(df['tiempo'], df['energia_cinetica'], 'b--', 
                    alpha=0.7, label='Cinética', linewidth=1.5)
            ax7.plot(df['tiempo'], df['energia_potencial'], 'r--', 
                    alpha=0.7, label='Potencial', linewidth=1.5)
        ax7.set_xlabel('Tiempo (s)')
        ax7.set_ylabel('Energía')
        ax7.set_title('Energía Total')
        ax7.legend()
        ax7.grid(True, alpha=0.3)
        
        # 8. Pureza
        ax8 = plt.subplot(3, 3, 8)
        ax8.plot(df['tiempo'], df['pureza'], color=color_principal, linewidth=1.5)
        ax8.set_xlabel('Tiempo (s)')
        ax8.set_ylabel('Pureza')
        ax8.set_title('Pureza del Estado')
        ax8.grid(True, alpha=0.3)
        ax8.set_ylim(0, 1.1)
        
        # 9. Elementos de matriz de covarianza
        ax9 = plt.subplot(3, 3, 9)
        ax9.plot(df['tiempo'], df['Sigma_XX'], 'b-', label='Σ_XX', linewidth=1.5)
        ax9.plot(df['tiempo'], df['Sigma_PP'], 'r-', label='Σ_PP', linewidth=1.5)
        ax9.plot(df['tiempo'], df['Sigma_XP'], 'g-', label='Σ_XP', linewidth=1.5)
        ax9.set_xlabel('Tiempo (s)')
        ax9.set_ylabel('Elemento de Σ')
        ax9.set_title('Matriz de Covarianza')
        ax9.legend()
        ax9.grid(True, alpha=0.3)
        
        plt.tight_layout()
        self.fig_principal = fig
    
    def analizar_clasico(self):
        """Análisis específico para oscilador clásico"""
        print("📊 Generando gráficas para Oscilador Clásico...")
        
        fig = plt.figure(figsize=(16, 12))
        fig.suptitle('Oscilador Armónico Clásico', fontsize=16, fontweight='bold',
                    color=COLORES_SIMULACION['oscilador_clasico'])
        
        df = self.df
        color_principal = COLORES_SIMULACION['oscilador_clasico']
        
        # 1. Espacio de fase
        ax1 = plt.subplot(2, 3, 1)
        ax1.plot(df['posicion'], df['velocidad'], color=color_principal, linewidth=1.5, alpha=0.8)
        ax1.plot(df['posicion'].iloc[0], df['velocidad'].iloc[0], 'go', 
                markersize=12, label='Inicio', zorder=5)
        ax1.set_xlabel('Posición x')
        ax1.set_ylabel('Velocidad v')
        ax1.set_title('Espacio de Fase (x, v)')
        ax1.grid(True, alpha=0.3)
        ax1.axis('equal')
        
        if 'amplitud' in df.columns and 'omega' in df.columns:
            A = df['amplitud'].iloc[0]
            omega = df['omega'].iloc[0]
            theta = np.linspace(0, 2*np.pi, 100)
            x_circ = A * np.cos(theta)
            v_circ = -A * omega * np.sin(theta)
            ax1.plot(x_circ, v_circ, 'b--', alpha=0.5, linewidth=2, label='Teórico')
        ax1.legend()
        
        # 2. Posición vs tiempo
        ax2 = plt.subplot(2, 3, 2)
        ax2.plot(df['tiempo'], df['posicion'], color=color_principal, linewidth=1.5)
        ax2.set_xlabel('Tiempo (s)')
        ax2.set_ylabel('Posición x')
        ax2.set_title('Posición vs Tiempo')
        ax2.grid(True, alpha=0.3)
        
        # 3. Velocidad vs tiempo
        ax3 = plt.subplot(2, 3, 3)
        ax3.plot(df['tiempo'], df['velocidad'], 'b-', linewidth=1.5)
        ax3.set_xlabel('Tiempo (s)')
        ax3.set_ylabel('Velocidad v')
        ax3.set_title('Velocidad vs Tiempo')
        ax3.grid(True, alpha=0.3)
        
        # 4. Energía
        ax4 = plt.subplot(2, 3, 4)
        ax4.plot(df['tiempo'], df['energia_total'], 'g-', 
                label='Total', linewidth=2)
        ax4.plot(df['tiempo'], df['energia_cinetica'], 'b--', 
                alpha=0.7, label='Cinética', linewidth=1.5)
        ax4.plot(df['tiempo'], df['energia_potencial'], 'r--', 
                alpha=0.7, label='Potencial', linewidth=1.5)
        ax4.set_xlabel('Tiempo (s)')
        ax4.set_ylabel('Energía')
        ax4.set_title('Conservación de Energía')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # 5. Fase instantánea
        ax5 = plt.subplot(2, 3, 5)
        ax5.plot(df['tiempo'], np.degrees(df['fase_instantanea']), color=color_principal, linewidth=1.5)
        ax5.set_xlabel('Tiempo (s)')
        ax5.set_ylabel('Fase (grados)')
        ax5.set_title('Fase Instantánea')
        ax5.grid(True, alpha=0.3)
        
        # 6. Distancia al origen
        ax6 = plt.subplot(2, 3, 6)
        ax6.plot(df['tiempo'], df['distancia_origen'], 'orange', linewidth=1.5)
        if 'amplitud' in df.columns:
            ax6.axhline(y=df['amplitud'].iloc[0], color='r', linestyle='--', 
                       label=f'Amplitud = {df["amplitud"].iloc[0]:.3f}')
        ax6.set_xlabel('Tiempo (s)')
        ax6.set_ylabel('r = √(x² + v²)')
        ax6.set_title('Distancia al Origen en Espacio de Fase')
        ax6.legend()
        ax6.grid(True, alpha=0.3)
        
        plt.tight_layout()
        self.fig_principal = fig
    
    def analizar_green_split(self):
        """Análisis específico para Green's function - Split Operator"""
        print("📊 Generando gráficas para Green Split-Operator...")
        
        fig = plt.figure(figsize=(18, 14))
        fig.suptitle("Green's Function: Split-Operator (Numérico)", 
                    fontsize=16, fontweight='bold',
                    color=COLORES_SIMULACION['green_split_operator'])
        
        df = self.df
        color_principal = COLORES_SIMULACION['green_split_operator']
        
        # 1. Trayectoria en espacio de fase
        ax1 = plt.subplot(3, 3, 1)
        if 'force_activa' in df.columns:
            mask_on = df['force_activa'] == 1
            mask_off = df['force_activa'] == 0
            if mask_on.any():
                ax1.plot(df.loc[mask_on, 'X_avg'], df.loc[mask_on, 'P_avg'], 
                        color=color_principal, alpha=0.7, linewidth=1.5, label='Fuerza ON')
            if mask_off.any():
                ax1.plot(df.loc[mask_off, 'X_avg'], df.loc[mask_off, 'P_avg'], 
                        'gray', alpha=0.5, linewidth=1.5, label='Fuerza OFF')
        else:
            scatter = ax1.scatter(df['X_avg'], df['P_avg'], 
                                 c=df['tiempo'], cmap='cool', s=5)
            plt.colorbar(scatter, ax=ax1, label='Tiempo')
        ax1.plot(df['X_avg'].iloc[0], df['P_avg'].iloc[0], 'go', 
                markersize=12, label='Inicio', zorder=5)
        ax1.set_xlabel('⟨X⟩')
        ax1.set_ylabel('⟨P⟩')
        ax1.set_title('Trayectoria en Espacio de Fase')
        ax1.legend(loc='upper right')
        ax1.grid(True, alpha=0.3)
        ax1.axis('equal')
        
        # 2. Fuerza externa
        ax2 = plt.subplot(3, 3, 2)
        if 'fuerza_externa' in df.columns:
            ax2.plot(df['tiempo'], df['fuerza_externa'], color=color_principal, linewidth=1.5)
            ax2.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        ax2.set_xlabel('Tiempo (s)')
        ax2.set_ylabel('F(t)')
        ax2.set_title('Fuerza Externa')
        ax2.grid(True, alpha=0.3)
        
        # 3. Cuadraturas
        ax3 = plt.subplot(3, 3, 3)
        ax3.plot(df['tiempo'], df['X_avg'], color=color_principal, label='⟨X⟩', linewidth=1.5)
        ax3.plot(df['tiempo'], df['P_avg'], 'r-', label='⟨P⟩', linewidth=1.5)
        ax3.set_xlabel('Tiempo (s)')
        ax3.set_ylabel('Valor esperado')
        ax3.set_title('Cuadraturas vs Tiempo')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Incertezas
        ax4 = plt.subplot(3, 3, 4)
        ax4.plot(df['tiempo'], df['delta_X'], color=color_principal, label='ΔX', linewidth=1.5)
        ax4.plot(df['tiempo'], df['delta_P'], 'r-', label='ΔP', linewidth=1.5)
        ax4.axhline(y=1/np.sqrt(2), color='g', linestyle='--', alpha=0.5)
        ax4.set_xlabel('Tiempo (s)')
        ax4.set_ylabel('Incerteza')
        ax4.set_title('Incertezas')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # 5. Energía
        ax5 = plt.subplot(3, 3, 5)
        ax5.plot(df['tiempo'], df['energia_total'], 'g-', 
                label='Total', linewidth=2)
        ax5.plot(df['tiempo'], df['energia_cinetica'], 'b--', 
                alpha=0.7, label='Cinética', linewidth=1.5)
        ax5.plot(df['tiempo'], df['energia_potencial'], 'r--', 
                alpha=0.7, label='Potencial', linewidth=1.5)
        ax5.set_xlabel('Tiempo (s)')
        ax5.set_ylabel('Energía')
        ax5.set_title('Energía del Sistema')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        # 6. Trabajo y potencia
        ax6 = plt.subplot(3, 3, 6)
        if 'trabajo_instantaneo' in df.columns:
            ax6.plot(df['tiempo'], df['trabajo_instantaneo'], 'orange', 
                    label='Trabajo', linewidth=1.5)
        if 'potencia' in df.columns:
            ax6.plot(df['tiempo'], df['potencia'], 'purple', 
                    label='Potencia', linewidth=1.5)
        ax6.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        ax6.set_xlabel('Tiempo (s)')
        ax6.set_ylabel('Trabajo / Potencia')
        ax6.set_title('Trabajo y Potencia')
        ax6.legend()
        ax6.grid(True, alpha=0.3)
        
        # 7. |α| magnitud
        ax7 = plt.subplot(3, 3, 7)
        ax7.plot(df['tiempo'], df['alpha_magnitud'], color=color_principal, linewidth=1.5)
        ax7.set_xlabel('Tiempo (s)')
        ax7.set_ylabel('|α|')
        ax7.set_title('Magnitud del Desplazamiento')
        ax7.grid(True, alpha=0.3)
        
        # 8. Pureza
        ax8 = plt.subplot(3, 3, 8)
        ax8.plot(df['tiempo'], df['pureza'], 'magenta', linewidth=1.5)
        ax8.set_xlabel('Tiempo (s)')
        ax8.set_ylabel('Pureza')
        ax8.set_title('Pureza del Estado')
        ax8.grid(True, alpha=0.3)
        ax8.set_ylim(0, 1.1)
        
        # 9. Producto de incertezas
        ax9 = plt.subplot(3, 3, 9)
        ax9.plot(df['tiempo'], df['producto_incerteza'], 'purple', linewidth=1.5)
        ax9.axhline(y=0.5, color='r', linestyle='--', label='Límite Heisenberg')
        ax9.set_xlabel('Tiempo (s)')
        ax9.set_ylabel('ΔX·ΔP')
        ax9.set_title('Principio de Incertidumbre')
        ax9.legend()
        ax9.grid(True, alpha=0.3)
        
        plt.tight_layout()
        self.fig_principal = fig
    
    def analizar_green_analitico(self):
        """Análisis específico para Green's function - Analítico"""
        print("📊 Generando gráficas para Green Analítico...")
        
        fig = plt.figure(figsize=(18, 14))
        fig.suptitle("Green's Function: Solución Analítica", 
                    fontsize=16, fontweight='bold',
                    color=COLORES_SIMULACION['green_analitico'])
        
        df = self.df
        color_principal = COLORES_SIMULACION['green_analitico']
        
        # 1. Trayectoria en espacio de fase
        ax1 = plt.subplot(3, 3, 1)
        if 'force_activa' in df.columns:
            mask_on = df['force_activa'] == 1
            mask_off = df['force_activa'] == 0
            if mask_on.any():
                ax1.plot(df.loc[mask_on, 'X_avg'], df.loc[mask_on, 'P_avg'], 
                        color=color_principal, alpha=0.8, linewidth=1.5, label='Fuerza ON')
            if mask_off.any():
                ax1.plot(df.loc[mask_off, 'X_avg'], df.loc[mask_off, 'P_avg'], 
                        'gray', alpha=0.5, linewidth=1.5, label='Fuerza OFF')
        else:
            scatter = ax1.scatter(df['X_avg'], df['P_avg'], 
                                 c=df['tiempo'], cmap='hot', s=5)
            plt.colorbar(scatter, ax=ax1, label='Tiempo')
        ax1.plot(df['X_avg'].iloc[0], df['P_avg'].iloc[0], 'go', 
                markersize=12, label='Inicio', zorder=5)
        ax1.set_xlabel('⟨X⟩')
        ax1.set_ylabel('⟨P⟩')
        ax1.set_title('Trayectoria en Espacio de Fase')
        ax1.legend(loc='upper right')
        ax1.grid(True, alpha=0.3)
        ax1.axis('equal')
        
        # 2. Fuerza externa
        ax2 = plt.subplot(3, 3, 2)
        if 'fuerza_externa' in df.columns:
            ax2.plot(df['tiempo'], df['fuerza_externa'], color=color_principal, linewidth=1.5)
            ax2.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        ax2.set_xlabel('Tiempo (s)')
        ax2.set_ylabel('F(t)')
        ax2.set_title('Fuerza Externa F(t) = F₀cos(νt)')
        ax2.grid(True, alpha=0.3)
        
        # 3. Cuadraturas
        ax3 = plt.subplot(3, 3, 3)
        ax3.plot(df['tiempo'], df['X_avg'], color=color_principal, label='⟨X⟩', linewidth=1.5)
        ax3.plot(df['tiempo'], df['P_avg'], 'r-', label='⟨P⟩', linewidth=1.5)
        ax3.set_xlabel('Tiempo (s)')
        ax3.set_ylabel('Valor esperado')
        ax3.set_title('Cuadraturas vs Tiempo')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Incertezas
        ax4 = plt.subplot(3, 3, 4)
        ax4.plot(df['tiempo'], df['delta_X'], color=color_principal, label='ΔX', linewidth=1.5)
        ax4.plot(df['tiempo'], df['delta_P'], 'r-', label='ΔP', linewidth=1.5)
        ax4.axhline(y=1/np.sqrt(2), color='g', linestyle='--', 
                   alpha=0.5, label='Estado coherente')
        ax4.set_xlabel('Tiempo (s)')
        ax4.set_ylabel('Incerteza')
        ax4.set_title('Incertezas (Estado Mínimo)')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # 5. Energía
        ax5 = plt.subplot(3, 3, 5)
        ax5.plot(df['tiempo'], df['energia_total'], 'g-', 
                label='Total', linewidth=2)
        ax5.plot(df['tiempo'], df['energia_cinetica'], 'b--', 
                alpha=0.7, label='Cinética', linewidth=1.5)
        ax5.plot(df['tiempo'], df['energia_potencial'], 'r--', 
                alpha=0.7, label='Potencial', linewidth=1.5)
        ax5.set_xlabel('Tiempo (s)')
        ax5.set_ylabel('Energía')
        ax5.set_title('Energía del Sistema')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        # 6. Trabajo y potencia
        ax6 = plt.subplot(3, 3, 6)
        if 'trabajo_instantaneo' in df.columns:
            ax6.plot(df['tiempo'], df['trabajo_instantaneo'], 'orange', 
                    label='Trabajo', linewidth=1.5)
        if 'potencia' in df.columns:
            ax6.plot(df['tiempo'], df['potencia'], 'purple', 
                    label='Potencia', linewidth=1.5)
        ax6.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        ax6.set_xlabel('Tiempo (s)')
        ax6.set_ylabel('Trabajo / Potencia')
        ax6.set_title('Trabajo y Potencia')
        ax6.legend()
        ax6.grid(True, alpha=0.3)
        
        # 7. |α| y fase
        ax7 = plt.subplot(3, 3, 7)
        ax7.plot(df['tiempo'], df['alpha_magnitud'], color=color_principal, 
                linewidth=1.5, label='|α|')
        ax7.set_xlabel('Tiempo (s)')
        ax7.set_ylabel('|α|')
        ax7.set_title('Magnitud del Estado Coherente')
        ax7.grid(True, alpha=0.3)
        
        # 8. α en plano complejo
        ax8 = plt.subplot(3, 3, 8, projection='polar')
        theta = df['alpha_fase'].values
        r = df['alpha_magnitud'].values
        scatter = ax8.scatter(theta, r, c=df['tiempo'], cmap='hot', s=8)
        ax8.set_title('α en Plano Complejo')
        plt.colorbar(scatter, ax=ax8, label='Tiempo', pad=0.1)
        
        # 9. Comparación ω vs ν
        ax9 = plt.subplot(3, 3, 9)
        if 'omega_oscilador' in df.columns and 'nu_frecuencia' in df.columns:
            omega = df['omega_oscilador'].iloc[0]
            nu = df['nu_frecuencia'].iloc[0]
            detuning = omega - nu
            
            info_text = f'ω = {omega:.4f}\nν = {nu:.4f}\nΔ = ω - ν = {detuning:.4f}'
            ax9.text(0.5, 0.7, info_text, transform=ax9.transAxes,
                    fontsize=14, verticalalignment='top', horizontalalignment='center',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            
            if abs(detuning) < 0.1:
                ax9.text(0.5, 0.3, '⚠️ CERCA DE RESONANCIA', transform=ax9.transAxes,
                        fontsize=12, color='red', fontweight='bold',
                        horizontalalignment='center')
            
        ax9.set_title('Parámetros del Sistema')
        ax9.axis('off')
        
        plt.tight_layout()
        self.fig_principal = fig
    
    def analizar_generico(self):
        """Análisis genérico para tipos no reconocidos"""
        print("📊 Generando gráficas genéricas...")
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f'Análisis: {self.nombre}', fontsize=16, fontweight='bold')
        
        df = self.df
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if 'tiempo' in numeric_cols:
            t = df['tiempo']
            numeric_cols.remove('tiempo')
            
            for ax, col in zip(axes.flat, numeric_cols[:4]):
                ax.plot(t, df[col], linewidth=1.5)
                ax.set_xlabel('Tiempo (s)')
                ax.set_ylabel(col)
                ax.set_title(col)
                ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        self.fig_principal = fig
    
    # =========================================================================
    # ANÁLISIS FFT
    # =========================================================================
    
    def analisis_fft(self, guardar=True):
        """Análisis de frecuencias usando FFT"""
        print("\n🎵 ANÁLISIS DE FRECUENCIAS (FFT)")
        print("-" * 70)
        
        if self.tipo == 'oscilador_clasico':
            señales = {
                'Posición': self.df['posicion'].values,
                'Velocidad': self.df['velocidad'].values
            }
        else:
            señales = {}
            if 'X_avg' in self.df.columns:
                señales['⟨X⟩'] = self.df['X_avg'].values
                señales['⟨P⟩'] = self.df['P_avg'].values
            if 'delta_X' in self.df.columns:
                señales['ΔX'] = self.df['delta_X'].values
            if 'fuerza_externa' in self.df.columns:
                señales['F(t)'] = self.df['fuerza_externa'].values
        
        if not señales:
            print("⚠️  No hay señales para análisis FFT")
            return {}
        
        dt = self.df['tiempo'].iloc[1] - self.df['tiempo'].iloc[0]
        N = len(self.df)
        
        n_signals = len(señales)
        fig, axes = plt.subplots(n_signals, 2, figsize=(16, 4*n_signals))
        fig.suptitle(f'Análisis FFT: {self.nombre} ({NOMBRES_SIMULACION.get(self.tipo, self.tipo)})', 
                    fontsize=14, fontweight='bold',
                    color=COLORES_SIMULACION.get(self.tipo, 'black'))
        
        if n_signals == 1:
            axes = axes.reshape(1, -1)
        
        resultados_fft = {}
        color = COLORES_SIMULACION.get(self.tipo, 'blue')
        
        for idx, (nombre, señal) in enumerate(señales.items()):
            fft_vals = fft(señal)
            fft_freq = fftfreq(N, dt)
            
            pos_mask = fft_freq > 0
            fft_freq_pos = fft_freq[pos_mask]
            fft_mag = np.abs(fft_vals[pos_mask])
            fft_power = fft_mag**2
            
            peaks, properties = find_peaks(fft_power, height=np.max(fft_power)*0.05)
            
            if len(peaks) > 0:
                freq_dominante = fft_freq_pos[peaks[0]]
                potencia_dominante = fft_power[peaks[0]]
                resultados_fft[nombre] = {
                    'frecuencia': freq_dominante,
                    'omega': 2*np.pi*freq_dominante,
                    'periodo': 1/freq_dominante,
                    'potencia': potencia_dominante
                }
                
                print(f"\n{nombre}:")
                print(f"  Frecuencia dominante: {freq_dominante:.6f} Hz")
                print(f"  ω = {2*np.pi*freq_dominante:.6f} rad/s")
                print(f"  Período: {1/freq_dominante:.6f} s")
            
            axes[idx, 0].plot(self.df['tiempo'], señal, color=color, linewidth=1)
            axes[idx, 0].set_xlabel('Tiempo (s)')
            axes[idx, 0].set_ylabel(nombre)
            axes[idx, 0].set_title(f'Señal: {nombre}')
            axes[idx, 0].grid(True, alpha=0.3)
            
            axes[idx, 1].semilogy(fft_freq_pos, fft_power, color=color, linewidth=1)
            if len(peaks) > 0:
                axes[idx, 1].plot(fft_freq_pos[peaks], fft_power[peaks], 
                                 'go', markersize=10, label='Picos')
                axes[idx, 1].annotate(f'f={fft_freq_pos[peaks[0]]:.4f} Hz',
                                     xy=(fft_freq_pos[peaks[0]], fft_power[peaks[0]]),
                                     xytext=(10, 10), textcoords='offset points',
                                     fontsize=9, color='green',
                                     bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
            axes[idx, 1].set_xlabel('Frecuencia (Hz)')
            axes[idx, 1].set_ylabel('Potencia')
            axes[idx, 1].set_title(f'Espectro: {nombre}')
            axes[idx, 1].grid(True, alpha=0.3)
            axes[idx, 1].legend()
            axes[idx, 1].set_xlim(0, min(5, fft_freq_pos.max()))
        
        plt.tight_layout()
        self.fig_fft = fig
        
        if guardar:
            self._guardar_figura_fft()
        
        print("\n" + "-" * 70)
        return resultados_fft
    
    # =========================================================================
    # AJUSTE DE CURVAS
    # =========================================================================
    
    def ajustar_curvas(self, guardar=True):
        """Ajuste de curvas a funciones teóricas"""
        print("\n🔍 AJUSTE DE CURVAS")
        print("-" * 70)
        
        def oscilacion(t, A, omega, phi, offset):
            return A * np.cos(omega * t + phi) + offset
        
        t = self.df['tiempo'].values
        
        if self.tipo == 'oscilador_clasico':
            y = self.df['posicion'].values
            y_name = 'Posición x(t)'
        elif 'X_avg' in self.df.columns:
            y = self.df['X_avg'].values
            y_name = '⟨X⟩(t)'
        else:
            print("⚠️  No hay señal para ajustar")
            return
        
        A_guess = np.std(y) * np.sqrt(2)
        omega_guess = 1.0
        phi_guess = 0
        offset_guess = np.mean(y)
        
        try:
            popt, pcov = curve_fit(oscilacion, t, y, 
                                  p0=[A_guess, omega_guess, phi_guess, offset_guess],
                                  maxfev=10000)
            perr = np.sqrt(np.diag(pcov))
            
            A_fit, omega_fit, phi_fit, offset_fit = popt
            
            print(f"\n🎯 Ajuste de {y_name} = A·cos(ω·t + φ) + offset:")
            print(f"  A = {A_fit:.6f} ± {perr[0]:.6f}")
            print(f"  ω = {omega_fit:.6f} ± {perr[1]:.6f} rad/s")
            print(f"  φ = {phi_fit:.6f} ± {perr[2]:.6f} rad")
            print(f"  offset = {offset_fit:.6f} ± {perr[3]:.6f}")
            print(f"  Período ajustado: {2*np.pi/omega_fit:.6f} s")
            
            y_fit = oscilacion(t, *popt)
            residuos = y - y_fit
            ss_res = np.sum(residuos**2)
            ss_tot = np.sum((y - np.mean(y))**2)
            r_squared = 1 - (ss_res / ss_tot)
            print(f"  R² = {r_squared:.8f}")
            
            color = COLORES_SIMULACION.get(self.tipo, 'blue')
            
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8))
            fig.suptitle(f'Ajuste de Curvas: {self.nombre} ({NOMBRES_SIMULACION.get(self.tipo, self.tipo)})', 
                        fontsize=14, fontweight='bold', color=color)
            
            ax1.plot(t, y, '.', color=color, markersize=2, alpha=0.5, label='Datos')
            ax1.plot(t, y_fit, 'r-', linewidth=2, label='Ajuste')
            ax1.set_xlabel('Tiempo (s)')
            ax1.set_ylabel(y_name)
            ax1.set_title(f'{y_name} = {A_fit:.3f}·cos({omega_fit:.3f}·t + {phi_fit:.3f}) + {offset_fit:.3f}')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            ax2.plot(t, residuos, 'g-', linewidth=1)
            ax2.axhline(y=0, color='k', linestyle='--', alpha=0.3)
            ax2.set_xlabel('Tiempo (s)')
            ax2.set_ylabel('Residuos')
            ax2.set_title(f'Residuos (R² = {r_squared:.6f})')
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            self.fig_ajuste = fig
            
            if guardar:
                self._guardar_figura_ajuste()
            
        except Exception as e:
            print(f"  ❌ Error en ajuste: {e}")
        
        print("-" * 70)
    
    # =========================================================================
    # MÉTODOS DE GUARDADO
    # =========================================================================
    
    def _guardar_figura_principal(self):
        """Guarda la figura principal"""
        if self.fig_principal is not None:
            filename = FIGURAS_DIR / f'{self.nombre}_analisis.png'
            self.fig_principal.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"💾 Figura principal guardada: {filename}")
    
    def _guardar_figura_fft(self):
        """Guarda la figura FFT"""
        if self.fig_fft is not None:
            filename = FFT_DIR / f'{self.nombre}_fft.png'
            self.fig_fft.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"💾 FFT guardado: {filename}")
    
    def _guardar_figura_ajuste(self):
        """Guarda la figura de ajuste"""
        if self.fig_ajuste is not None:
            filename = AJUSTES_DIR / f'{self.nombre}_ajuste.png'
            self.fig_ajuste.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"💾 Ajuste guardado: {filename}")
    
    def guardar_todas_figuras(self):
        """Guarda todas las figuras generadas"""
        print("\n💾 Guardando todas las figuras...")
        self._guardar_figura_principal()
        self._guardar_figura_fft()
        self._guardar_figura_ajuste()
    
    def exportar_resumen(self):
        """Exporta un resumen del análisis a archivo de texto"""
        filename = SALIDAS_DIR / f'resumen_{self.nombre}.txt'
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(f"{'='*70}\n")
            f.write(f"RESUMEN DE ANÁLISIS - {self.nombre}\n")
            f.write(f"{'='*70}\n\n")
            
            f.write(f"Tipo de simulación: {NOMBRES_SIMULACION.get(self.tipo, self.tipo)}\n")
            f.write(f"Puntos de datos: {len(self.df)}\n")
            f.write(f"Tiempo total: {self.df['tiempo'].max():.4f} s\n")
            f.write(f"Fecha de análisis: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("-" * 70 + "\n")
            f.write("ESTADÍSTICAS\n")
            f.write("-" * 70 + "\n\n")
            
            if self.tipo != 'oscilador_clasico':
                if 'X_avg' in self.df.columns:
                    f.write(f"⟨X⟩ promedio: {self.df['X_avg'].mean():.6f} ± {self.df['X_avg'].std():.6f}\n")
                    f.write(f"⟨P⟩ promedio: {self.df['P_avg'].mean():.6f} ± {self.df['P_avg'].std():.6f}\n")
                if 'delta_X' in self.df.columns:
                    f.write(f"ΔX promedio: {self.df['delta_X'].mean():.6f} ± {self.df['delta_X'].std():.6f}\n")
                    f.write(f"ΔP promedio: {self.df['delta_P'].mean():.6f} ± {self.df['delta_P'].std():.6f}\n")
                    f.write(f"ΔX·ΔP promedio: {self.df['producto_incerteza'].mean():.6f}\n")
                if 'energia_total' in self.df.columns:
                    f.write(f"Energía promedio: {self.df['energia_total'].mean():.6f}\n")
                if 'pureza' in self.df.columns:
                    f.write(f"Pureza promedio: {self.df['pureza'].mean():.6f}\n")
                if 'entropia' in self.df.columns:
                    f.write(f"Entropía promedio: {self.df['entropia'].mean():.6f}\n")
            else:
                if 'posicion' in self.df.columns:
                    f.write(f"Posición máx: {self.df['posicion'].max():.6f}\n")
                    f.write(f"Velocidad máx: {self.df['velocidad'].max():.6f}\n")
                if 'energia_total' in self.df.columns:
                    f.write(f"Energía: {self.df['energia_total'].mean():.6f} ± {self.df['energia_total'].std():.6f}\n")
            
            f.write("\n" + "=" * 70 + "\n")
        
        print(f"📄 Resumen exportado: {filename}")


# =============================================================================
# CLASE: ComparadorSimulaciones
# =============================================================================

class ComparadorSimulaciones:
    """Clase para comparar múltiples simulaciones entre sí"""
    
    def __init__(self, archivos):
        self.archivos = archivos
        self.analizadores = []
        
        print(f"\n{'='*70}")
        print(f"🔄 COMPARADOR DE SIMULACIONES")
        print(f"   Cargando {len(archivos)} archivos...")
        print(f"{'='*70}\n")
        
        for archivo in archivos:
            try:
                analizador = AnalizadorQHO(archivo)
                self.analizadores.append(analizador)
            except Exception as e:
                print(f"❌ Error cargando {archivo}: {e}")
        
        print(f"\n✅ {len(self.analizadores)} simulaciones cargadas\n")
        
        # Mostrar tipos detectados
        print("📋 Tipos detectados:")
        for a in self.analizadores:
            print(f"   • {a.nombre}: {NOMBRES_SIMULACION.get(a.tipo, a.tipo)}")
        print()
    
    def _get_color(self, tipo):
        return COLORES_SIMULACION.get(tipo, 'gray')
    
    def _get_nombre(self, tipo):
        return NOMBRES_SIMULACION.get(tipo, tipo)
    
    def comparar_trayectorias(self):
        """Compara trayectorias en espacio de fase"""
        print("\n📍 COMPARACIÓN DE TRAYECTORIAS EN ESPACIO DE FASE")
        print("-" * 70)
        
        fig, ax = plt.subplots(figsize=(12, 10))
        
        for analizador in self.analizadores:
            df = analizador.df
            color = self._get_color(analizador.tipo)
            label = self._get_nombre(analizador.tipo)
            
            if analizador.tipo == 'oscilador_clasico':
                x = df['posicion'].values
                p = df['velocidad'].values
            else:
                x = df['X_avg'].values
                p = df['P_avg'].values
            
            ax.plot(x, p, '-', color=color, linewidth=2, alpha=0.8, label=label)
            ax.plot(x[0], p[0], 'o', color=color, markersize=10, zorder=5)
            ax.plot(x[-1], p[-1], 's', color=color, markersize=8, zorder=5)
        
        ax.set_xlabel('X / x', fontsize=12)
        ax.set_ylabel('P / v', fontsize=12)
        ax.set_title('Comparación de Trayectorias en Espacio de Fase', fontsize=14)
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        
        plt.tight_layout()
        self.fig_trayectorias = fig
        print("✅ Gráfica de trayectorias generada")
    
    def comparar_energias(self):
        """Compara conservación de energía"""
        print("\n⚡ COMPARACIÓN DE ENERGÍAS")
        print("-" * 70)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle('Comparación de Energías', fontsize=14, fontweight='bold')
        
        for analizador in self.analizadores:
            df = analizador.df
            color = self._get_color(analizador.tipo)
            label = self._get_nombre(analizador.tipo)
            
            t = df['tiempo'].values
            E = df['energia_total'].values
            
            ax1.plot(t, E, '-', color=color, linewidth=2, label=label)
            
            E_mean = np.mean(E)
            E_rel = (E - E_mean) / (E_mean + 1e-10) * 100
            ax2.plot(t, E_rel, '-', color=color, linewidth=2, label=label)
            
            print(f"\n{label}:")
            print(f"  E promedio: {E_mean:.6f}")
            print(f"  Desv. estándar: {np.std(E):.6f}")
            print(f"  Variación rel: {np.std(E)/(E_mean+1e-10)*100:.4f}%")
        
        ax1.set_xlabel('Tiempo (s)')
        ax1.set_ylabel('Energía Total')
        ax1.set_title('Energía vs Tiempo')
        ax1.legend(loc='best', fontsize=9)
        ax1.grid(True, alpha=0.3)
        
        ax2.set_xlabel('Tiempo (s)')
        ax2.set_ylabel('Variación Relativa (%)')
        ax2.set_title('Variación Relativa de Energía')
        ax2.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        ax2.legend(loc='best', fontsize=9)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        self.fig_energias = fig
        print("\n✅ Gráfica de energías generada")
    
    def comparar_incertezas(self):
        """Compara incertezas entre simulaciones cuánticas"""
        print("\n📊 COMPARACIÓN DE INCERTEZAS")
        print("-" * 70)
        
        cuanticos = [a for a in self.analizadores if a.tipo != 'oscilador_clasico']
        
        if not cuanticos:
            print("⚠️  No hay simulaciones cuánticas para comparar")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Comparación de Incertezas Cuánticas', fontsize=14, fontweight='bold')
        
        for analizador in cuanticos:
            df = analizador.df
            color = self._get_color(analizador.tipo)
            label = self._get_nombre(analizador.tipo)
            t = df['tiempo'].values
            
            if 'delta_X' in df.columns:
                axes[0, 0].plot(t, df['delta_X'], '-', color=color, 
                               linewidth=1.5, label=label)
            if 'delta_P' in df.columns:
                axes[0, 1].plot(t, df['delta_P'], '-', color=color, 
                               linewidth=1.5, label=label)
            if 'producto_incerteza' in df.columns:
                axes[1, 0].plot(t, df['producto_incerteza'], '-', color=color, 
                               linewidth=1.5, label=label)
            if 'pureza' in df.columns:
                axes[1, 1].plot(t, df['pureza'], '-', color=color, 
                               linewidth=1.5, label=label)
        
        axes[0, 0].axhline(y=1/np.sqrt(2), color='k', linestyle='--', alpha=0.5)
        axes[0, 1].axhline(y=1/np.sqrt(2), color='k', linestyle='--', alpha=0.5)
        axes[1, 0].axhline(y=0.5, color='r', linestyle='--', alpha=0.7, label='Heisenberg')
        
        axes[0, 0].set_xlabel('Tiempo (s)')
        axes[0, 0].set_ylabel('ΔX')
        axes[0, 0].set_title('Incerteza en X')
        axes[0, 0].legend(loc='best', fontsize=8)
        axes[0, 0].grid(True, alpha=0.3)
        
        axes[0, 1].set_xlabel('Tiempo (s)')
        axes[0, 1].set_ylabel('ΔP')
        axes[0, 1].set_title('Incerteza en P')
        axes[0, 1].legend(loc='best', fontsize=8)
        axes[0, 1].grid(True, alpha=0.3)
        
        axes[1, 0].set_xlabel('Tiempo (s)')
        axes[1, 0].set_ylabel('ΔX·ΔP')
        axes[1, 0].set_title('Producto de Incertezas')
        axes[1, 0].legend(loc='best', fontsize=8)
        axes[1, 0].grid(True, alpha=0.3)
        
        axes[1, 1].set_xlabel('Tiempo (s)')
        axes[1, 1].set_ylabel('Pureza')
        axes[1, 1].set_title('Pureza del Estado')
        axes[1, 1].legend(loc='best', fontsize=8)
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].set_ylim(0, 1.1)
        
        plt.tight_layout()
        self.fig_incertezas = fig
        print("✅ Gráfica de incertezas generada")
    
    def tabla_comparativa(self):
        """Genera tabla comparativa"""
        print("\n📋 TABLA COMPARATIVA")
        print("=" * 100)
        
        datos = []
        
        for analizador in self.analizadores:
            df = analizador.df
            tipo = self._get_nombre(analizador.tipo)
            
            fila = {
                'Simulación': tipo,
                'Puntos': len(df),
                'Tiempo (s)': f"{df['tiempo'].max():.2f}",
                'E promedio': f"{df['energia_total'].mean():.4f}",
                'E σ': f"{df['energia_total'].std():.6f}",
            }
            
            if analizador.tipo != 'oscilador_clasico':
                if 'producto_incerteza' in df.columns:
                    fila['ΔX·ΔP mín'] = f"{df['producto_incerteza'].min():.4f}"
                else:
                    fila['ΔX·ΔP mín'] = 'N/A'
                
                if 'pureza' in df.columns:
                    fila['Pureza'] = f"{df['pureza'].mean():.4f}"
                else:
                    fila['Pureza'] = 'N/A'
            else:
                fila['ΔX·ΔP mín'] = 'N/A (clásico)'
                fila['Pureza'] = 'N/A (clásico)'
            
            datos.append(fila)
        
        tabla = pd.DataFrame(datos)
        print(tabla.to_string(index=False))
        print("=" * 100)
        
        csv_file = COMPARACIONES_DIR / 'tabla_comparativa.csv'
        tabla.to_csv(csv_file, index=False)
        print(f"\n💾 Tabla CSV: {csv_file}")
        
        self.tabla = tabla
    
    def guardar_figuras(self):
        """Guarda todas las figuras comparativas"""
        print(f"\n💾 Guardando figuras comparativas en: {COMPARACIONES_DIR}")
        
        figuras = {
            'fig_trayectorias': 'comparacion_trayectorias.png',
            'fig_energias': 'comparacion_energias.png',
            'fig_incertezas': 'comparacion_incertezas.png',
        }
        
        for attr, filename in figuras.items():
            if hasattr(self, attr):
                fig = getattr(self, attr)
                if fig is not None:
                    filepath = COMPARACIONES_DIR / filename
                    fig.savefig(filepath, dpi=300, bbox_inches='tight')
                    print(f"  ✅ {filename}")
    
    def ejecutar_comparacion_completa(self):
        """Ejecuta todas las comparaciones disponibles"""
        print("\n" + "="*70)
        print("🔄 EJECUTANDO COMPARACIÓN COMPLETA")
        print("="*70)
        
        self.comparar_trayectorias()
        self.comparar_energias()
        self.comparar_incertezas()
        self.tabla_comparativa()
        self.guardar_figuras()
        
        print("\n✅ Comparación completa finalizada!")


# =============================================================================
# FUNCIONES DE UTILIDAD
# =============================================================================

def analizar_archivo(archivo_csv, mostrar=True):
    """Analiza un archivo CSV individual"""
    crear_directorios()
    
    analizador = AnalizadorQHO(archivo_csv)
    analizador.analisis_completo(guardar=True)
    analizador.analisis_fft(guardar=True)
    analizador.ajustar_curvas(guardar=True)
    analizador.exportar_resumen()
    
    if GENERAR_ANIMACION_AUTO and analizador.tipo != 'oscilador_clasico':
        analizador.crear_animacion()
    
    if mostrar:
        plt.show()
    
    return analizador


def analizar_directorio(directorio='Proyecto/Final/CSV1'):
    """Analiza todos los archivos CSV en un directorio"""
    crear_directorios()
    
    archivos_csv = list(Path(directorio).glob('*.csv'))
    
    if not archivos_csv:
        print(f"❌ No se encontraron archivos CSV en {directorio}")
        return []
    
    print(f"\n📁 Encontrados {len(archivos_csv)} archivos CSV\n")
    
    analizadores = []
    
    for idx, archivo in enumerate(archivos_csv, 1):
        print(f"\n{'='*70}")
        print(f"📊 Archivo {idx}/{len(archivos_csv)}: {archivo.name}")
        print(f"{'='*70}")
        
        try:
            analizador = AnalizadorQHO(str(archivo))
            analizador.analisis_completo(guardar=True)
            analizador.analisis_fft(guardar=True)
            analizador.ajustar_curvas(guardar=True)
            analizador.exportar_resumen()
            
            analizadores.append(analizador)
            print(f"\n✅ {archivo.name} completado")
            
        except Exception as e:
            print(f"❌ Error al analizar {archivo.name}: {e}")
            import traceback
            traceback.print_exc()
    
    if len(analizadores) > 1:
        print("\n" + "="*70)
        print("🔄 INICIANDO COMPARACIÓN ENTRE SIMULACIONES")
        print("="*70)
        
        comparador = ComparadorSimulaciones([str(a.archivo) for a in analizadores])
        comparador.ejecutar_comparacion_completa()
    
    print("\n" + "="*70)
    print("✅ ANÁLISIS COMPLETO FINALIZADO")
    print(f"   Resultados guardados en: {RESULTADOS_DIR.absolute()}")
    print("="*70)
    
    return analizadores


def comparar_archivos(archivos):
    """Compara múltiples archivos CSV"""
    crear_directorios()
    
    comparador = ComparadorSimulaciones(archivos)
    comparador.ejecutar_comparacion_completa()
    
    plt.show()
    return comparador


# =============================================================================
# MAIN
# =============================================================================

if __name__ == '__main__':
    print("""
    ╔═══════════════════════════════════════════════════════════════════╗
    ║  ANÁLISIS DE SIMULACIONES QHO - v2.0                              ║
    ║  Oscilador Armónico Cuántico                                      ║
    ╠═══════════════════════════════════════════════════════════════════╣
    ║  DETECCIÓN MEJORADA:                                              ║
    ║    • Por nombre de archivo                                        ║
    ║    • Por columnas específicas                                     ║
    ║    • Por anisotropía de matriz de covarianza                      ║
    ║    • Por excentricidad de elipse                                  ║
    ║    • Por squeezing_dB                                             ║
    ╠═══════════════════════════════════════════════════════════════════╣
    ║  Tipos soportados:                                                ║
    ║    • Estado Coherente (azul)                                      ║
    ║    • Estado Comprimido (verde)                                    ║
    ║    • Superposición Fock (púrpura)                                 ║
    ║    • Oscilador Clásico (rojo)                                     ║
    ║    • Green Split-Operator (cyan)                                  ║
    ║    • Green Analítico (naranja)                                    ║
    ╚═══════════════════════════════════════════════════════════════════╝
    """)
    
    crear_directorios()
    
    if len(sys.argv) > 1:
        if sys.argv[1] == '--comparar':
            if len(sys.argv) < 4:
                print("❌ Uso: python analisis_qho.py --comparar archivo1.csv archivo2.csv ...")
                sys.exit(1)
            
            archivos = sys.argv[2:]
            print(f"\n🔄 Modo Comparación: {len(archivos)} archivos\n")
            comparar_archivos(archivos)
        
        elif sys.argv[1] == '--directorio':
            directorio = sys.argv[2] if len(sys.argv) > 2 else '.'
            print(f"\n📁 Modo Directorio: {directorio}\n")
            analizar_directorio(directorio)
            plt.show()
        
        else:
            archivo = sys.argv[1]
            if not os.path.exists(archivo):
                print(f"❌ Archivo no encontrado: {archivo}")
                sys.exit(1)
            
            print(f"\n📊 Modo Individual: {archivo}\n")
            analizar_archivo(archivo)
    
    else:
        print("\n📁 Analizando directorio actual...\n")
        analizar_directorio()
        plt.show()