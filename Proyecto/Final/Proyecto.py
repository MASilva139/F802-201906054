import math
import numpy as np
import pygame
import csv
from datetime import datetime
from scipy.linalg import expm
from pathlib import Path

WIDTH, HEIGHT = 1280, 700
TAB_HEIGHT = 50
SIM_SIZE = 600

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (200, 0, 0)
BLUE = (0, 0, 200)
GREEN = (0, 150, 0)
CYAN = (0, 180, 180)
MAGENTA = (180, 0, 180)
GRAY = (200, 200, 200)
LIGHT_GRAY = (240, 240, 240)
ORANGE = (255, 140, 0)

GUARDAR_AUTOMATICO = False

class SimulacionBase():
    def __init__(self, width, height, x_offset, y_offset):
        self.window_width = width
        self.window_height = height
        self.x_offset = x_offset
        self.y_offset = y_offset

        self.sim_size = SIM_SIZE
        self.margin = 50

        self.sim_x_offset = self.x_offset + (self.window_width - self.sim_size) // 2
        self.sim_y_offset = self.y_offset + (self.window_height - self.sim_size) // 2

        self.t = 0.0
        self.dt = 0.016
        self.trail = []
        self.max_trail = 2000

        self.datos_csv = []
        self.nombre_simulacion = "simulacion"

    def world_to_screen(self, x, y):
        xmin, xmax, ymin, ymax = self.world_bounds()
        sx = (self.sim_size - 2*self.margin) / (xmax - xmin)
        sy = (self.sim_size - 2*self.margin) / (ymax - ymin)
        px = self.sim_x_offset + self.margin + (x - xmin) * sx
        py = self.sim_y_offset + self.sim_size - (self.margin + (y - ymin) * sy)
        return int(px), int(py)

    def draw_axes(self, screen):
        ox, oy = self.world_to_screen(0.0, 0.0)
        rect = pygame.Rect(self.sim_x_offset, self.sim_y_offset, self.sim_size-1, self.sim_size-1)
        pygame.draw.rect(screen, BLACK, rect, 2)
        pygame.draw.line(screen, BLACK, (self.sim_x_offset, oy), (self.sim_x_offset + self.sim_size, oy), 1)
        pygame.draw.line(screen, BLACK, (ox, self.sim_y_offset), (ox, self.sim_y_offset + self.sim_size), 1)

    def update(self, speed):
        self.t += self.dt * speed

    def guardar_datos_frame(self):
        pass

    def exportar_csv(self):
        if not self.datos_csv:
            print(f"[{self.nombre_simulacion}] No hay datos para exportar")
            return
        CSV = Path('Proyecto/Final/CSV')
        CSV.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = CSV/f"{self.nombre_simulacion}_{timestamp}.csv"

        with open(filename, 'w', newline='') as csvfile:
            fieldnames = self.datos_csv[0].keys()
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for row in self.datos_csv:
                writer.writerow(row)

        print(f"[{self.nombre_simulacion}] Datos exportados a {filename}")
        print(f"[{self.nombre_simulacion}] Total de {len(self.datos_csv)} registros guardados")
        return filename

class EstadoCoherente(SimulacionBase):
    def __init__(self, width, height, x_offset, y_offset):
        super().__init__(width, height, x_offset, y_offset)
        self.omega = 1.0
        self.alpha0_mag = 1.5
        self.alpha0_phase = np.pi/6
        self.alpha0 = self.alpha0_mag * np.exp(1j * self.alpha0_phase)

        self.r = 0.0
        self.theta = 0.0
        self.R_world = np.sqrt(2) * abs(self.alpha0) + 3.0
        self.zoom = 1.0

        self.drive_on = False
        self.F0 = 0.6
        self.nu = 1.0
        self.kappa = 1/np.sqrt(2.0)
        self.I_drive = 0.0 + 0.0j
        self.drive_history = []

        self.Sigma0 = self.covariance_t0()
        self.nombre_simulacion = "estado_coherente"

    def world_bounds(self):
        r = self.R_world / self.zoom
        return (-r, r, -r, r)

    def covariance_t0(self):
        c, s = np.cos(self.theta), np.sin(self.theta)
        R = np.array([[c, -s], [s, c]])
        D = np.diag([np.exp(2*self.r), np.exp(-2*self.r)])
        return R @ D @ R.T

    def rotate_covariance(self, t):
        c, s = np.cos(self.omega * t), np.sin(self.omega * t)
        R = np.array([[c, -s], [s, c]])
        return R @ self.Sigma0 @ R.T

    def forcing(self, t):
        return self.F0 * np.cos(self.nu * t)

    def update(self, speed):
        t_next = self.t + self.dt * speed
        if self.drive_on:
            f1 = self.forcing(self.t)
            f2 = self.forcing(t_next)
            ph1 = np.exp(1j * self.omega * self.t)
            ph2 = np.exp(1j * self.omega * t_next)
            self.I_drive += 0.5 * ((ph1*f1) + (ph2*f2)) * (t_next - self.t)
        self.t = t_next

        self.drive_history.append(self.forcing(self.t) if self.drive_on else 0.0)
        if len(self.drive_history) > self.max_trail:
            self.drive_history.pop(0)

        if GUARDAR_AUTOMATICO:
            self.guardar_datos_frame()

    def get_alpha(self):
        return np.exp(-1j * self.omega * self.t) * (self.alpha0 + self.kappa * self.I_drive)

    def guardar_datos_frame(self):
        a = self.get_alpha()
        X = np.sqrt(2.0) * a.real
        P = np.sqrt(2.0) * a.imag
        Sigma = self.rotate_covariance(self.t)
        delta_X = np.sqrt(Sigma[0, 0])
        delta_P = np.sqrt(Sigma[1, 1])

        n_avg = abs(a)**2
        n_variance = abs(a)**2
        mandel_Q = (n_variance - n_avg) / (n_avg + 1e-10)
        fano_F = n_variance / (n_avg + 1e-10)

        X2 = Sigma[0,0] + X**2
        P2 = Sigma[1,1] + P**2
        T_cinetica = P2 / 2.0
        V_potencial = X2 / 2.0

        det_Sigma = np.linalg.det(Sigma)
        pureza = 1.0 / (2.0 * np.sqrt(det_Sigma + 1e-10))

        nu = np.sqrt(det_Sigma)
        entropia = (nu + 0.5) * np.log(nu + 0.5) - (nu - 0.5) * np.log(nu - 0.5) if nu > 0.5 else 0.0

        eigenvalues = np.linalg.eigvalsh(Sigma)
        lambda_1 = max(eigenvalues)
        lambda_2 = min(eigenvalues)

        if abs(Sigma[0,0] - Sigma[1,1]) > 1e-10:
            theta_ellipse = 0.5 * np.arctan2(2*Sigma[0,1], Sigma[0,0] - Sigma[1,1])
        else:
            theta_ellipse = 0.0

        excentricidad = np.sqrt(1.0 - (lambda_2 / (lambda_1 + 1e-10))**2)
        area_elipse = np.pi * np.sqrt(det_Sigma)

        frame_number = int(self.t / self.dt)
        periodo_numero = int(self.omega * self.t / (2 * np.pi))

        self.datos_csv.append({
            'tiempo': self.t,
            'frame_number': frame_number,
            'periodo_numero': periodo_numero,
            'X_avg': X,
            'P_avg': P,
            'delta_X': delta_X,
            'delta_P': delta_P,
            'producto_incerteza': delta_X * delta_P,
            'alpha_real': a.real,
            'alpha_imag': a.imag,
            'alpha_magnitud': abs(a),
            'alpha_fase': np.angle(a),
            'n_promedio': n_avg,
            'n_varianza': n_variance,
            'mandel_Q': mandel_Q,
            'fano_F': fano_F,
            'energia_cinetica': T_cinetica,
            'energia_potencial': V_potencial,
            'energia_total': T_cinetica + V_potencial,
            'pureza': pureza,
            'entropia': entropia,
            'drive_activo': 1 if self.drive_on else 0,
            'I_drive_real': self.I_drive.real,
            'I_drive_imag': self.I_drive.imag,
            'omega': self.omega,
            'Sigma_XX': Sigma[0, 0],
            'Sigma_XP': Sigma[0, 1],
            'Sigma_PP': Sigma[1, 1],
            'det_Sigma': det_Sigma,
            'traza_Sigma': np.trace(Sigma),
            'lambda_1': lambda_1,
            'lambda_2': lambda_2,
            'theta_ellipse': theta_ellipse,
            'excentricidad': excentricidad,
            'area_elipse': area_elipse
        })

    def draw(self, screen):
        # Panel lateral izquierdo
        panel_x = 20
        panel_y = self.sim_y_offset + 20
        panel_w = 300

        # Título en el panel lateral
        font = pygame.font.SysFont(None, 28)
        font_small = pygame.font.SysFont(None, 24)
        title = font.render("Estado Coherente Cuántico", True, BLACK)
        info = font.render(f"t={self.t:.2f}", True, BLACK)
        screen.blit(title, (panel_x, panel_y))
        screen.blit(info, (panel_x, panel_y + 30))

        # Área de simulación
        self.draw_axes(screen)

        a = self.get_alpha()
        X = np.sqrt(2.0) * a.real
        P = np.sqrt(2.0) * a.imag
        center = (X, P)

        px, py = self.world_to_screen(X, P)
        self.trail.append((px, py))
        if len(self.trail) > self.max_trail:
            self.trail.pop(0)

        if len(self.trail) > 1:
            pygame.draw.lines(screen, RED, False, self.trail, 2)

        Sigma = self.rotate_covariance(self.t)
        self.draw_sigma_ellipse(screen, Sigma, center, GREEN)

        pygame.draw.circle(screen, BLACK, (px, py), 5)

        # Datos en panel lateral
        delta_X = np.sqrt(Sigma[0, 0])
        delta_P = np.sqrt(Sigma[1, 1])

        table_y = panel_y + 80
        pygame.draw.rect(screen, (240, 240, 240), (panel_x, table_y, panel_w, 220))
        pygame.draw.rect(screen, BLACK, (panel_x, table_y, panel_w, 220), 2)

        table_title = font.render("Observables", True, BLACK)
        screen.blit(table_title, (panel_x + 10, table_y + 10))

        # Valores esperados
        esperado_x = font_small.render(f"⟨X⟩ = {X:.4f}", True, BLACK)
        esperado_p = font_small.render(f"⟨P⟩ = {P:.4f}", True, BLACK)
        incerteza_x = font_small.render(f"ΔX = {delta_X:.4f}", True, BLACK)
        incerteza_p = font_small.render(f"ΔP = {delta_P:.4f}", True, BLACK)
        producto = font_small.render(f"ΔX·ΔP = {delta_X*delta_P:.4f}", True, BLACK)

        n_avg = abs(a)**2
        energia = self.omega * (n_avg + 0.5)

        n_text = font_small.render(f"⟨n⟩ = {n_avg:.4f}", True, BLACK)
        energia_text = font_small.render(f"E = {energia:.4f}", True, BLACK)
        alpha_text = font_small.render(f"|α| = {abs(a):.4f}", True, BLACK)
        fase_text = font_small.render(f"φ = {np.angle(a):.4f} rad", True, BLACK)

        y_offset = table_y + 35
        screen.blit(esperado_x, (panel_x + 10, y_offset))
        screen.blit(esperado_p, (panel_x + 10, y_offset + 20))
        screen.blit(incerteza_x, (panel_x + 10, y_offset + 40))
        screen.blit(incerteza_p, (panel_x + 10, y_offset + 60))
        screen.blit(producto, (panel_x + 10, y_offset + 80))
        screen.blit(n_text, (panel_x + 10, y_offset + 100))
        screen.blit(energia_text, (panel_x + 10, y_offset + 120))
        screen.blit(alpha_text, (panel_x + 10, y_offset + 140))
        screen.blit(fase_text, (panel_x + 10, y_offset + 160))

        # Parámetros
        param_y = table_y + 230
        pygame.draw.rect(screen, (250, 240, 230), (panel_x, param_y, panel_w, 100))
        pygame.draw.rect(screen, BLACK, (panel_x, param_y, panel_w, 100), 2)

        param_title = font.render("Parámetros", True, BLACK)
        screen.blit(param_title, (panel_x + 10, param_y + 10))

        omega_text = font_small.render(f"ω = {self.omega:.4f}", True, BLACK)
        alpha0_text = font_small.render(f"|α₀| = {self.alpha0_mag:.4f}", True, BLACK)
        drive_text = font_small.render(f"Drive: {'ON' if self.drive_on else 'OFF'}", True, GREEN if self.drive_on else RED)

        screen.blit(omega_text, (panel_x + 10, param_y + 35))
        screen.blit(alpha0_text, (panel_x + 10, param_y + 55))
        screen.blit(drive_text, (panel_x + 10, param_y + 75))

        # ⭐ Gráfico del drive
        if len(self.drive_history) > 1:
            self._draw_drive_history_panel(screen, panel_x, param_y + 110)

        # font = pygame.font.SysFont(None, 24)
        # font_small = pygame.font.SysFont(None, 18)
        # title = font.render("Estado Coherente Cuántico", True, BLACK)
        # info = font.render(f"t={self.t:.2f} ω={self.omega:.2f}", True, BLACK)
        # screen.blit(title, (self.sim_x_offset + 10, self.sim_y_offset + 10))
        # screen.blit(info, (self.sim_x_offset + 10, self.sim_y_offset + 35))

        # table_y = self.sim_y_offset + self.sim_size - 240
        # pygame.draw.rect(screen, (240, 240, 240), (self.sim_x_offset + 10, table_y, 260, 140))
        # pygame.draw.rect(screen, BLACK, (self.sim_x_offset + 10, table_y, 260, 140), 2)

        # table_title = font.render("Valores del Sistema", True, BLACK)
        # screen.blit(table_title, (self.sim_x_offset + 20, table_y + 10))

        # esperado_x = font_small.render(f"⟨X⟩ = {X:.4f}", True, BLACK)
        # esperado_p = font_small.render(f"⟨P⟩ = {P:.4f}", True, BLACK)
        # incerteza_x = font_small.render(f"ΔX = {delta_X:.4f}", True, BLACK)
        # incerteza_p = font_small.render(f"ΔP = {delta_P:.4f}", True, BLACK)
        # drive_status = font_small.render(f"Drive: {'ON' if self.drive_on else 'OFF'}",
        #                                 True, GREEN if self.drive_on else RED)

        # screen.blit(esperado_x, (self.sim_x_offset + 20, table_y + 35))
        # screen.blit(esperado_p, (self.sim_x_offset + 20, table_y + 55))
        # screen.blit(incerteza_x, (self.sim_x_offset + 20, table_y + 75))
        # screen.blit(incerteza_p, (self.sim_x_offset + 20, table_y + 95))
        # screen.blit(drive_status, (self.sim_x_offset + 20, table_y + 115))

        # if len(self.drive_history) > 1:
        #     self._draw_drive_history_panel(screen, self.sim_x_offset + 10, table_y + 150)

    def _draw_drive_history_panel(self, screen, x, y):
        graph_w = 260
        graph_h = 80

        pygame.draw.rect(screen, (250, 250, 250), (x, y, graph_w, graph_h))
        pygame.draw.rect(screen, BLACK, (x, y, graph_w, graph_h), 2)

        font_tiny = pygame.font.SysFont(None, 16)
        title = font_tiny.render("Drive F(t) - Historia", True, BLACK)
        screen.blit(title, (x + 5, y + 5))

        drives = np.array(self.drive_history)
        if len(drives) > 0:
            d_max = max(abs(drives.max()), abs(drives.min()), 0.1)

            zero_y = y + graph_h // 2
            pygame.draw.line(screen, (180, 180, 180), (x, zero_y), (x + graph_w, zero_y), 1)

            points = []
            for i, d in enumerate(drives):
                px = x + int((i / len(drives)) * graph_w)
                py = zero_y - int((d / d_max) * (graph_h // 2 - 10))
                points.append((px, py))

            if len(points) > 1:
                color = GREEN if self.drive_on else GRAY
                pygame.draw.lines(screen, color, False, points, 2)

    def draw_sigma_ellipse(self, surface, Sigma, center_world, color, segs=120):
        vals, vecs = np.linalg.eigh(Sigma)
        vals = np.clip(vals, 1e-8, None)
        radii = np.sqrt(vals)
        phi = np.linspace(0, 2*np.pi, segs, endpoint=True)
        circle = np.stack([np.cos(phi), np.sin(phi)], axis=0)
        pts = (vecs @ (radii[:,None] * circle))
        pts = pts.T + np.array(center_world)[None,:]
        pts_px = [self.world_to_screen(px, py) for px, py in pts]
        pygame.draw.lines(surface, color, True, pts_px, 2)

class EstadoComprimido(EstadoCoherente):
    def __init__(self, width, height, x_offset, y_offset):
        super().__init__(width, height, x_offset, y_offset)
        self.r = 0.7
        self.theta = np.pi/4
        self.Sigma0 = self.covariance_t0()
        self.nombre_simulacion = "estado_comprimido"
        self.drive_history = []

    def draw(self, screen):
        # Panel lateral izquierdo
        panel_x = 20
        panel_y = self.sim_y_offset + 10
        panel_w = 300

        # Título
        font = pygame.font.SysFont(None, 28)
        font_small = pygame.font.SysFont(None, 24)
        title = font.render("Estado Comprimido", True, BLACK)
        info = font.render(f"t={self.t:.2f}", True, BLACK)
        screen.blit(title, (panel_x, panel_y))
        screen.blit(info, (panel_x, panel_y + 30))

        # Área de simulación
        self.draw_axes(screen)

        a = self.get_alpha()
        X = np.sqrt(2.0) * a.real
        P = np.sqrt(2.0) * a.imag
        center = (X, P)

        px, py = self.world_to_screen(X, P)
        self.trail.append((px, py))
        if len(self.trail) > self.max_trail:
            self.trail.pop(0)

        if len(self.trail) > 1:
            pygame.draw.lines(screen, RED, False, self.trail, 2)

        Sigma = self.rotate_covariance(self.t)
        self.draw_sigma_ellipse(screen, Sigma, center, GREEN)

        pygame.draw.circle(screen, BLACK, (px, py), 5)

        # font = pygame.font.SysFont(None, 24)
        # font_small = pygame.font.SysFont(None, 18)
        # title = font.render("Estado Coherente Comprimido", True, BLACK)
        # info = font.render(f"t={self.t:.2f} r={self.r:.2f}", True, BLACK)
        # screen.blit(title, (self.sim_x_offset + 10, self.sim_y_offset + 10))
        # screen.blit(info, (self.sim_x_offset + 10, self.sim_y_offset + 35))

        # Datos en panel lateral
        delta_X = np.sqrt(Sigma[0, 0])
        delta_P = np.sqrt(Sigma[1, 1])

        table_y = panel_y + 80
        pygame.draw.rect(screen, (240, 240, 240), (panel_x, table_y, panel_w, 260))
        pygame.draw.rect(screen, BLACK, (panel_x, table_y, panel_w, 260), 2)

        table_title = font.render("Observables", True, BLACK)
        screen.blit(table_title, (panel_x + 10, table_y + 10))

        # Valores
        esperado_x = font_small.render(f"⟨X⟩ = {X:.4f}", True, BLACK)
        esperado_p = font_small.render(f"⟨P⟩ = {P:.4f}", True, BLACK)
        incerteza_x = font_small.render(f"ΔX = {delta_X:.4f}", True, BLACK)
        incerteza_p = font_small.render(f"ΔP = {delta_P:.4f}", True, BLACK)
        producto = font_small.render(f"ΔX·ΔP = {delta_X*delta_P:.4f}", True, BLACK)

        n_avg = abs(a)**2 + np.sinh(self.r)**2
        energia = self.omega * (n_avg + 0.5)

        n_text = font_small.render(f"⟨n⟩ = {n_avg:.4f}", True, BLACK)
        energia_text = font_small.render(f"E = {energia:.4f}", True, BLACK)
        alpha_text = font_small.render(f"|α| = {abs(a):.4f}", True, BLACK)

        # Squeezing
        min_var = min(delta_X**2, delta_P**2)
        sq_dB = -10 * np.log10(min_var + 1e-10)
        sq_text = font_small.render(f"Sq = {sq_dB:.2f} dB", True, BLACK)

        y_offset = table_y + 40
        screen.blit(esperado_x, (panel_x + 10, y_offset))
        screen.blit(esperado_p, (panel_x + 10, y_offset + 24))
        screen.blit(incerteza_x, (panel_x + 10, y_offset + 48))
        screen.blit(incerteza_p, (panel_x + 10, y_offset + 72))
        screen.blit(producto, (panel_x + 10, y_offset + 96))
        screen.blit(n_text, (panel_x + 10, y_offset + 120))
        screen.blit(energia_text, (panel_x + 10, y_offset + 144))
        screen.blit(alpha_text, (panel_x + 10, y_offset + 168))
        screen.blit(sq_text, (panel_x + 10, y_offset + 192))

        # Parámetros
        param_y = table_y + 270
        pygame.draw.rect(screen, (250, 240, 230), (panel_x, param_y, panel_w, 120))
        pygame.draw.rect(screen, BLACK, (panel_x, param_y, panel_w, 120), 2)

        param_title = font.render("Parámetros", True, BLACK)
        screen.blit(param_title, (panel_x + 10, param_y + 10))

        omega_text = font_small.render(f"ω = {self.omega:.4f}", True, BLACK)
        r_text = font_small.render(f"r = {self.r:.4f}", True, BLACK)
        theta_text = font_small.render(f"θ = {np.degrees(self.theta):.1f}°", True, BLACK)
        drive_text = font_small.render(f"Drive: {'ON' if self.drive_on else 'OFF'}",
                                       True, GREEN if self.drive_on else RED)

        screen.blit(omega_text, (panel_x + 10, param_y + 35))
        screen.blit(r_text, (panel_x + 10, param_y + 55))
        screen.blit(theta_text, (panel_x + 10, param_y + 75))
        screen.blit(drive_text, (panel_x + 10, param_y + 95))

        # ⭐ Gráfico del drive
        if len(self.drive_history) > 1:
            self._draw_drive_history_panel(screen, panel_x, param_y + 130)

        # table_y = self.sim_y_offset + self.sim_size - 260
        # pygame.draw.rect(screen, (240, 240, 240), (self.sim_x_offset + 10, table_y, 260, 160))
        # pygame.draw.rect(screen, BLACK, (self.sim_x_offset + 10, table_y, 260, 160), 2)

        # table_title = font.render("Valores del Sistema", True, BLACK)
        # screen.blit(table_title, (self.sim_x_offset + 20, table_y + 10))

        # esperado_x = font_small.render(f"⟨X⟩ = {X:.4f}", True, BLACK)
        # esperado_p = font_small.render(f"⟨P⟩ = {P:.4f}", True, BLACK)
        # incerteza_x = font_small.render(f"ΔX = {delta_X:.4f}", True, BLACK)
        # incerteza_p = font_small.render(f"ΔP = {delta_P:.4f}", True, BLACK)
        # producto_text = font_small.render(f"ΔX·ΔP = {delta_X*delta_P:.4f}", True, BLACK)
        # drive_status = font_small.render(f"Drive: {'ON' if self.drive_on else 'OFF'}",
        #                                 True, GREEN if self.drive_on else RED)

        # screen.blit(esperado_x, (self.sim_x_offset + 20, table_y + 35))
        # screen.blit(esperado_p, (self.sim_x_offset + 20, table_y + 55))
        # screen.blit(incerteza_x, (self.sim_x_offset + 20, table_y + 75))
        # screen.blit(incerteza_p, (self.sim_x_offset + 20, table_y + 95))
        # screen.blit(producto_text, (self.sim_x_offset + 20, table_y + 115))
        # screen.blit(drive_status, (self.sim_x_offset + 20, table_y + 135))

        # if len(self.drive_history) > 1:
        #     self._draw_drive_history_panel(screen, self.sim_x_offset + 10, table_y + 170)

class SuperposicionEstados(SimulacionBase):
    def __init__(self, width, height, x_offset, y_offset):
        super().__init__(width, height, x_offset, y_offset)
        self.omega = 1.0
        self.R_world = 5.0
        self.zoom = 1.0

        # Superposición de estados de Fock |0⟩, |1⟩, |2⟩, |3⟩, |4⟩, |5⟩
        self.n_max = 5
        self.coeffs = np.array([0.5, 0.5, 0.3, 0.2, 0.1, 0.1], dtype=complex)
        norm = np.sqrt(np.sum(np.abs(self.coeffs)**2))
        self.coeffs = self.coeffs / norm

        self.nombre_simulacion = "superposicion_fock"

        # DRIVE
        self.drive_on = False
        self.F0 = 0.6
        self.nu = 1.0
        self.kappa = 1/np.sqrt(2.0)
        self.I_drive = 0.0 + 0.0j
        self.drive_history = []

        # Precálculo de matrices
        self._X_matrix = None
        self._P_matrix = None
        self._precalcular_matrices()

    def world_bounds(self):
        r = self.R_world / self.zoom
        return (-r, r, -r, r)

    def _precalcular_matrices(self):
        n_states = self.n_max + 1

        self._X_matrix = np.zeros((n_states, n_states), dtype=float)
        for n in range(n_states):
            if n > 0:
                self._X_matrix[n, n-1] = np.sqrt(n) / np.sqrt(2)
            if n < n_states - 1:
                self._X_matrix[n, n+1] = np.sqrt(n+1) / np.sqrt(2)

        self._P_matrix = np.zeros((n_states, n_states), dtype=complex)
        for n in range(n_states):
            if n > 0:
                self._P_matrix[n, n-1] = -1j * np.sqrt(n) / np.sqrt(2)
            if n < n_states - 1:
                self._P_matrix[n, n+1] = 1j * np.sqrt(n+1) / np.sqrt(2)

    def forcing(self, t):
        return self.F0 * np.cos(self.nu * t)

    def update(self, speed):
        t_next = self.t + self.dt * speed

        if self.drive_on:
            f1 = self.forcing(self.t)
            f2 = self.forcing(t_next)
            ph1 = np.exp(1j * self.omega * self.t)
            ph2 = np.exp(1j * self.omega * t_next)
            self.I_drive += 0.5 * ((ph1*f1) + (ph2*f2)) * (t_next - self.t)

        self.t = t_next

        # Guardar historia del drive
        self.drive_history.append(self.forcing(self.t) if self.drive_on else 0.0)
        if len(self.drive_history) > self.max_trail:
            self.drive_history.pop(0)

        if GUARDAR_AUTOMATICO:
            self.guardar_datos_frame()

    def get_coeffs_t(self):
        phases = np.exp(-1j * self.omega * (np.arange(len(self.coeffs)) + 0.5) * self.t)
        return self.coeffs * phases

    def get_observables(self):
        c_t = self.get_coeffs_t() # Observables del estado

        # Valores esperados usando matrices precalculadas
        X_avg = np.real(np.vdot(c_t, self._X_matrix @ c_t))
        P_avg = np.real(np.vdot(c_t, self._P_matrix @ c_t))

        alpha_drive = np.exp(-1j * self.omega * self.t) * self.kappa * self.I_drive
        X_avg += np.sqrt(2) * alpha_drive.real
        P_avg += np.sqrt(2) * alpha_drive.imag

        # Momentos cuadráticos
        X2_avg = np.real(np.vdot(c_t, self._X_matrix @ self._X_matrix @ c_t))
        P2_avg = np.real(np.vdot(c_t, self._P_matrix @ self._P_matrix @ c_t))
        XP_avg = np.real(np.vdot(c_t, self._X_matrix @ self._P_matrix @ c_t))

        # Matriz de covarianza
        Sigma = np.array([
            [X2_avg - X_avg**2, XP_avg - X_avg*P_avg],
            [XP_avg - X_avg*P_avg, P2_avg - P_avg**2]
        ])

        E = sum(self.omega * (n + 0.5) * np.abs(c_t[n])**2 for n in range(len(c_t)))
        if self.drive_on:
            E += self.omega * abs(alpha_drive)**2

        phase = np.angle(sum(c_t[n] * np.sqrt(n+1) for n in range(len(c_t)-1)))

        return X_avg, P_avg, Sigma, E, phase

    def guardar_datos_frame(self):
        X_avg, P_avg, Sigma, E, phase = self.get_observables()
        delta_X = np.sqrt(max(Sigma[0, 0], 1e-10))
        delta_P = np.sqrt(max(Sigma[1, 1], 1e-10))

        c_t = self.get_coeffs_t()

        # Probabilidades
        probs = {f'prob_n{n}': np.abs(c_t[n])**2 for n in range(len(c_t))}

        # Número de fotones
        n_avg = sum(n * np.abs(c_t[n])**2 for n in range(len(c_t)))
        n2_avg = sum(n**2 * np.abs(c_t[n])**2 for n in range(len(c_t)))
        n_variance = n2_avg - n_avg**2

        mandel_Q = (n_variance - n_avg) / (n_avg + 1e-10)
        fano_F = n_variance / (n_avg + 1e-10)

        # Energías
        X2 = Sigma[0,0] + X_avg**2
        P2 = Sigma[1,1] + P_avg**2
        T_cinetica = P2 / 2.0
        V_potencial = X2 / 2.0

        # Pureza y entropía
        det_Sigma = np.linalg.det(Sigma)
        pureza = 1.0 / (2.0 * np.sqrt(det_Sigma + 1e-10))

        nu = np.sqrt(det_Sigma)
        if nu > 0.5:
            entropia = (nu + 0.5) * np.log(nu + 0.5) - (nu - 0.5) * np.log(nu - 0.5)
        else:
            entropia = 0.0

        # Autovalores
        eigenvalues = np.linalg.eigvalsh(Sigma)
        lambda_1 = max(eigenvalues)
        lambda_2 = min(eigenvalues)

        if abs(Sigma[0,0] - Sigma[1,1]) > 1e-10:
            theta_ellipse = 0.5 * np.arctan2(2*Sigma[0,1], Sigma[0,0] - Sigma[1,1])
        else:
            theta_ellipse = 0.0

        excentricidad = np.sqrt(1.0 - (lambda_2 / (lambda_1 + 1e-10))**2)
        area_elipse = np.pi * np.sqrt(det_Sigma)

        frame_number = int(self.t / self.dt)
        periodo_numero = int(self.omega * self.t / (2 * np.pi))

        # self.datos_csv.append({
        #     'tiempo': self.t,
        #     'frame_number': frame_number,
        #     'periodo_numero': periodo_numero,
        #     'X_avg': X_avg,
        #     'P_avg': P_avg,
        #     'delta_X': delta_X,
        #     'delta_P': delta_P,
        #     'producto_incerteza': delta_X * delta_P,
        #     'n_promedio': n_avg,
        #     'n_varianza': n_variance,
        #     'mandel_Q': mandel_Q,
        #     'fano_F': fano_F,
        #     'energia_total': E,
        #     'energia_cinetica': T_cinetica,
        #     'energia_potencial': V_potencial,
        #     'pureza': pureza,
        #     'entropia': entropia,
        #     'fase': phase,
        #     'Sigma_XX': Sigma[0, 0],
        #     'Sigma_XP': Sigma[0, 1],
        #     'Sigma_PP': Sigma[1, 1],
        #     'det_Sigma': det_Sigma,
        #     'traza_Sigma': np.trace(Sigma),
        #     'lambda_1': lambda_1,
        #     'lambda_2': lambda_2,
        #     'theta_ellipse': theta_ellipse,
        #     'excentricidad': excentricidad,
        #     'area_elipse': area_elipse
        # })

        datos = {
            'tiempo': self.t,
            'frame_number': frame_number,
            'periodo_numero': periodo_numero,
            'X_avg': X_avg,
            'P_avg': P_avg,
            'delta_X': delta_X,
            'delta_P': delta_P,
            'producto_incerteza': delta_X * delta_P,
            'n_promedio': n_avg,
            'n_varianza': n_variance,
            'mandel_Q': mandel_Q,
            'fano_F': fano_F,
            'energia_total': E,
            'energia_cinetica': T_cinetica,
            'energia_potencial': V_potencial,
            'pureza': pureza,
            'entropia': entropia,
            'fase': phase,
            'Sigma_XX': Sigma[0, 0],
            'Sigma_XP': Sigma[0, 1],
            'Sigma_PP': Sigma[1, 1],
            'det_Sigma': det_Sigma,
            'traza_Sigma': np.trace(Sigma),
            'lambda_1': lambda_1,
            'lambda_2': lambda_2,
            'theta_ellipse': theta_ellipse,
            'excentricidad': excentricidad,
            'area_elipse': area_elipse
        }

        datos.update(probs)
        self.datos_csv.append(datos)

    def draw(self, screen):
        # Panel lateral izquierdo
        panel_x = 20
        panel_y = self.sim_y_offset + 20
        panel_w = 300

        # Título
        font = pygame.font.SysFont(None, 28)
        font_small = pygame.font.SysFont(None, 26)
        title = font.render("Superposición Fock", True, BLACK)
        info = font.render(f"t={self.t:.2f}", True, BLACK)
        screen.blit(title, (panel_x, panel_y))
        screen.blit(info, (panel_x, panel_y + 28))

        # Área de simulación
        self.draw_axes(screen)

        X_avg, P_avg, Sigma, E, phase = self.get_observables()
        center = (X_avg, P_avg)

        px, py = self.world_to_screen(X_avg, P_avg)
        self.trail.append((px, py))
        if len(self.trail) > self.max_trail:
            self.trail.pop(0)

        if len(self.trail) > 1:
            pygame.draw.lines(screen, (150, 0, 150), False, self.trail, 2)

        self.draw_sigma_ellipse(screen, Sigma, center, (100, 100, 200))

        pygame.draw.circle(screen, BLACK, (px, py), 5)

        # Datos en panel lateral
        delta_X = np.sqrt(max(Sigma[0, 0], 1e-10))
        delta_P = np.sqrt(max(Sigma[1, 1], 1e-10))
        producto = delta_X * delta_P

        table_y = panel_y + 70
        pygame.draw.rect(screen, (240, 240, 240), (panel_x, table_y, panel_w, 240))
        pygame.draw.rect(screen, BLACK, (panel_x, table_y, panel_w, 240), 2)

        table_title = font.render("Observables", True, BLACK)
        screen.blit(table_title, (panel_x + 10, table_y + 10))

        esperado_x = font_small.render(f"⟨X⟩ = {X_avg:.4f}", True, BLACK)
        esperado_p = font_small.render(f"⟨P⟩ = {P_avg:.4f}", True, BLACK)
        incerteza_x = font_small.render(f"ΔX = {delta_X:.4f}", True, BLACK)
        incerteza_p = font_small.render(f"ΔP = {delta_P:.4f}", True, BLACK)
        producto_text = font_small.render(f"ΔX·ΔP = {producto:.4f}", True, BLACK)
        energia_text = font_small.render(f"E = {E:.4f}", True, BLACK)

        c_t = self.get_coeffs_t()
        n_avg = sum(n * np.abs(c_t[n])**2 for n in range(len(c_t)))
        n_text = font_small.render(f"⟨n⟩ = {n_avg:.4f}", True, BLACK)

        pureza = 1.0 / (2.0 * np.sqrt(np.linalg.det(Sigma) + 1e-10))
        pureza_text = font_small.render(f"Pureza = {pureza:.4f}", True, BLACK)

        y_offset = table_y + 40
        screen.blit(esperado_x, (panel_x + 10, y_offset))
        screen.blit(esperado_p, (panel_x + 10, y_offset + 25))
        screen.blit(incerteza_x, (panel_x + 10, y_offset + 50))
        screen.blit(incerteza_p, (panel_x + 10, y_offset + 75))
        screen.blit(producto_text, (panel_x + 10, y_offset + 100))
        screen.blit(energia_text, (panel_x + 10, y_offset + 125))
        screen.blit(n_text, (panel_x + 10, y_offset + 150))
        screen.blit(pureza_text, (panel_x + 10, y_offset + 175))

        # Probabilidades de Fock
        prob_y = table_y + 250
        pygame.draw.rect(screen, (250, 240, 230), (panel_x, prob_y, panel_w, 190))
        pygame.draw.rect(screen, BLACK, (panel_x, prob_y, panel_w, 190), 2)

        prob_title = font.render("Probabilidades", True, BLACK)
        screen.blit(prob_title, (panel_x + 10, prob_y + 10))

        for n in range(min(6, len(c_t))):
            prob_n = np.abs(c_t[n])**2
            prob_text = font_small.render(f"P(n={n}) = {prob_n:.4f}", True, BLACK)
            screen.blit(prob_text, (panel_x + 10, prob_y + 35 + n*25))

        # Parámetros y Drive
        drive_x = panel_x + 935
        # drive_y = prob_y + 150
        drive_y = panel_y + 70
        pygame.draw.rect(screen, (240, 250, 230), (drive_x, drive_y, panel_w, 100))
        pygame.draw.rect(screen, BLACK, (drive_x, drive_y, panel_w, 100), 2)

        drive_title = font.render("Drive", True, BLACK)
        screen.blit(drive_title, (drive_x + 10, drive_y + 10))

        drive_status = font_small.render(f"Drive: {'ON' if self.drive_on else 'OFF'}",
                                        True, GREEN if self.drive_on else RED)
        f_val = self.forcing(self.t) if self.drive_on else 0.0
        drive_f = font_small.render(f"F(t) = {f_val:.4f}", True, BLACK)
        drive_nu = font_small.render(f"ν = {self.nu:.4f}", True, BLACK)
        drive_F0 = font_small.render(f"F₀ = {self.F0:.4f}", True, BLACK)

        screen.blit(drive_status, (drive_x + 10, drive_y + 35))
        screen.blit(drive_f, (drive_x + 10, drive_y + 55))
        screen.blit(drive_nu, (drive_x + 10, drive_y + 75))

        # Gráfico del drive
        if len(self.drive_history) > 1:
            self._draw_drive_history_panel(screen, drive_x, drive_y + 110)

        # font = pygame.font.SysFont(None, 22)
        # font_small = pygame.font.SysFont(None, 18)
        # title = font.render("Superposición de Estados de Fock", True, BLACK)
        # info = font.render(f"t={self.t:.2f} E={E:.3f}", True, BLACK)
        # screen.blit(title, (self.sim_x_offset + 10, self.sim_y_offset + 10))
        # screen.blit(info, (self.sim_x_offset + 10, self.sim_y_offset + 32))

        # table_y = self.sim_y_offset + self.sim_size - 280
        # pygame.draw.rect(screen, (240, 240, 240), (self.sim_x_offset + 10, table_y, 280, 180))
        # pygame.draw.rect(screen, BLACK, (self.sim_x_offset + 10, table_y, 280, 180), 2)

        # table_title = font.render("Valores del Sistema", True, BLACK)
        # screen.blit(table_title, (self.sim_x_offset + 20, table_y + 10))

        # esperado_x = font_small.render(f"⟨X⟩ = {X_avg:.4f}", True, BLACK)
        # esperado_p = font_small.render(f"⟨P⟩ = {P_avg:.4f}", True, BLACK)
        # incerteza_x = font_small.render(f"ΔX = {delta_X:.4f}", True, BLACK)
        # incerteza_p = font_small.render(f"ΔP = {delta_P:.4f}", True, BLACK)
        # producto_text = font_small.render(f"ΔX·ΔP = {delta_X*delta_P:.4f}", True, BLACK)
        # energia_text = font_small.render(f"E = {E:.4f}", True, BLACK)
        # drive_status = font_small.render(f"Drive: {'ON' if self.drive_on else 'OFF'}",
        #                                 True, GREEN if self.drive_on else RED)

        # screen.blit(esperado_x, (self.sim_x_offset + 20, table_y + 35))
        # screen.blit(esperado_p, (self.sim_x_offset + 20, table_y + 55))
        # screen.blit(incerteza_x, (self.sim_x_offset + 20, table_y + 75))
        # screen.blit(incerteza_p, (self.sim_x_offset + 20, table_y + 95))
        # screen.blit(producto_text, (self.sim_x_offset + 20, table_y + 115))
        # screen.blit(energia_text, (self.sim_x_offset + 20, table_y + 135))
        # screen.blit(drive_status, (self.sim_x_offset + 20, table_y + 155))

        # if len(self.drive_history) > 1:
        #     self._draw_drive_history_panel(screen, self.sim_x_offset + 10, table_y + 190)

    def _draw_drive_history_panel(self, screen, x, y):
        graph_w = 280
        graph_h = 80

        pygame.draw.rect(screen, (250, 250, 250), (x, y, graph_w, graph_h))
        pygame.draw.rect(screen, BLACK, (x, y, graph_w, graph_h), 2)

        font_tiny = pygame.font.SysFont(None, 16)
        title = font_tiny.render("Drive F(t) - Historia", True, BLACK)
        screen.blit(title, (x + 5, y + 5))

        drives = np.array(self.drive_history)
        if len(drives) > 0:
            d_max = max(abs(drives.max()), abs(drives.min()), 0.1)

            zero_y = y + graph_h // 2
            pygame.draw.line(screen, (180, 180, 180), (x, zero_y), (x + graph_w, zero_y), 1)

            points = []
            for i, d in enumerate(drives):
                px = x + int((i / len(drives)) * graph_w)
                py = zero_y - int((d / d_max) * (graph_h // 2 - 10))
                points.append((px, py))

            if len(points) > 1:
                color = GREEN if self.drive_on else GRAY
                pygame.draw.lines(screen, color, False, points, 2)

    def draw_sigma_ellipse(self, surface, Sigma, center_world, color, segs=120):
        vals, vecs = np.linalg.eigh(Sigma)
        vals = np.clip(vals, 1e-8, None)
        radii = np.sqrt(vals)
        phi = np.linspace(0, 2*np.pi, segs, endpoint=True)
        circle = np.stack([np.cos(phi), np.sin(phi)], axis=0)
        pts = (vecs @ (radii[:,None] * circle))
        pts = pts.T + np.array(center_world)[None,:]
        pts_px = [self.world_to_screen(px, py) for px, py in pts]
        pygame.draw.lines(surface, color, True, pts_px, 2)

class OsciladorClasico(SimulacionBase):
    def __init__(self, width, height, x_offset, y_offset):
        super().__init__(width, height, x_offset, y_offset)
        self.omega = 1.0
        self.A = 3.0
        self.phase = np.pi/6
        self.R_world = self.A + 1.0
        self.zoom = 1.0
        self.nombre_simulacion = "oscilador_clasico"

    def world_bounds(self):
        r = self.R_world / self.zoom
        return (-r, r, -r, r)

    def get_position(self):
        x = self.A * np.cos(self.omega * self.t + self.phase)
        v = -self.A * self.omega * np.sin(self.omega * self.t + self.phase)
        return x, v

    def update(self, speed):
        self.t += self.dt * speed
        if GUARDAR_AUTOMATICO:
            self.guardar_datos_frame()

    def guardar_datos_frame(self):
        x, v = self.get_position()
        E = 0.5 * (self.omega**2 * x**2 + v**2)
        T = 0.5 * v**2
        V = 0.5 * self.omega**2 * x**2
        a = -self.omega**2 * x
        p = v

        if abs(x) > 1e-10:
            fase_instantanea = np.arctan2(v, self.omega * x)
        else:
            fase_instantanea = np.pi/2 if v > 0 else -np.pi/2

        r_fase = np.sqrt(x**2 + v**2)
        frame_number = int(self.t / self.dt)
        periodo_numero = int(self.omega * self.t / (2 * np.pi))

        self.datos_csv.append({
            'tiempo': self.t,
            'frame_number': frame_number,
            'periodo_numero': periodo_numero,
            'posicion': x,
            'velocidad': v,
            'aceleracion': a,
            'momentum': p,
            'energia_total': E,
            'energia_cinetica': T,
            'energia_potencial': V,
            'amplitud': self.A,
            'fase_inicial': self.phase,
            'fase_instantanea': fase_instantanea,
            'omega': self.omega,
            'periodo': 2 * np.pi / self.omega,
            'frecuencia': self.omega / (2 * np.pi),
            'distancia_origen': r_fase
        })

    def draw(self, screen):
        # Panel lateral izquierdo
        panel_x = 20
        panel_y = self.sim_y_offset + 20
        panel_w = 300

        # Título
        font = pygame.font.SysFont(None, 28)
        font_small = pygame.font.SysFont(None, 24)
        title = font.render("Oscilador Clásico", True, BLACK)
        info = font.render(f"t={self.t:.2f}", True, BLACK)
        screen.blit(title, (panel_x, panel_y))
        screen.blit(info, (panel_x, panel_y + 30))

        # Área de simulación
        self.draw_axes(screen)

        x, v = self.get_position()

        px, py = self.world_to_screen(x, v)
        self.trail.append((px, py))
        if len(self.trail) > self.max_trail:
            self.trail.pop(0)

        if len(self.trail) > 1:
            pygame.draw.lines(screen, BLUE, False, self.trail, 2)

        pygame.draw.circle(screen, BLACK, (px, py), 7)

        radio_px = abs(self.world_to_screen(self.A, 0)[0] - self.world_to_screen(0, 0)[0])
        ox, oy = self.world_to_screen(0, 0)
        pygame.draw.circle(screen, (150, 150, 150), (ox, oy), radio_px, 1)

        # font = pygame.font.SysFont(None, 24)
        # font_small = pygame.font.SysFont(None, 18)
        # title = font.render("Oscilador Armónico Clásico", True, BLACK)
        # info = font.render(f"t={self.t:.2f} A={self.A:.2f}", True, BLACK)
        # screen.blit(title, (self.sim_x_offset + 10, self.sim_y_offset + 10))
        # screen.blit(info, (self.sim_x_offset + 10, self.sim_y_offset + 35))

        # Datos en panel lateral
        E = 0.5 * (self.omega**2 * x**2 + v**2)
        T = 0.5 * v**2
        V = 0.5 * self.omega**2 * x**2

        table_y = panel_y + 80
        pygame.draw.rect(screen, (240, 240, 240), (panel_x, table_y, panel_w, 180))
        pygame.draw.rect(screen, BLACK, (panel_x, table_y, panel_w, 180), 2)

        table_title = font.render("Observables", True, BLACK)
        screen.blit(table_title, (panel_x + 10, table_y + 10))

        pos_text = font_small.render(f"x = {x:.4f}", True, BLACK)
        vel_text = font_small.render(f"v = {v:.4f}", True, BLACK)
        energy_text = font_small.render(f"E = {E:.4f}", True, BLACK)
        T_text = font_small.render(f"T = {T:.4f}", True, BLACK)
        V_text = font_small.render(f"V = {V:.4f}", True, BLACK)
        r_fase = np.sqrt(x**2 + v**2)
        r_text = font_small.render(f"r = {r_fase:.4f}", True, BLACK)

        y_offset = table_y + 40
        screen.blit(pos_text, (panel_x + 10, y_offset))
        screen.blit(vel_text, (panel_x + 10, y_offset + 22))
        screen.blit(energy_text, (panel_x + 10, y_offset + 44))
        screen.blit(T_text, (panel_x + 10, y_offset + 66))
        screen.blit(V_text, (panel_x + 10, y_offset + 88))
        screen.blit(r_text, (panel_x + 10, y_offset + 110))

        # Parámetros
        param_y = table_y + 190
        pygame.draw.rect(screen, (250, 240, 230), (panel_x, param_y, panel_w, 100))
        pygame.draw.rect(screen, BLACK, (panel_x, param_y, panel_w, 100), 2)

        param_title = font.render("Parámetros", True, BLACK)
        screen.blit(param_title, (panel_x + 10, param_y + 10))

        omega_text = font_small.render(f"ω = {self.omega:.4f}", True, BLACK)
        A_text = font_small.render(f"A = {self.A:.4f}", True, BLACK)
        T_period = 2*np.pi/self.omega
        period_text = font_small.render(f"T = {T_period:.4f}", True, BLACK)

        screen.blit(omega_text, (panel_x + 10, param_y + 35))
        screen.blit(A_text, (panel_x + 10, param_y + 55))
        screen.blit(period_text, (panel_x + 10, param_y + 75))

        # table_y = self.sim_y_offset + self.sim_size - 120
        # pygame.draw.rect(screen, (240, 240, 240), (self.sim_x_offset + 10, table_y, 260, 100))
        # pygame.draw.rect(screen, BLACK, (self.sim_x_offset + 10, table_y, 260, 100), 2)

        # table_title = font.render("Valores del Sistema", True, BLACK)
        # screen.blit(table_title, (self.sim_x_offset + 20, table_y + 10))

        # pos_text = font_small.render(f"x = {x:.4f}", True, BLACK)
        # vel_text = font_small.render(f"v = {v:.4f}", True, BLACK)
        # energy_text = font_small.render(f"E = {E:.4f}", True, BLACK)

        # screen.blit(pos_text, (self.sim_x_offset + 20, table_y + 35))
        # screen.blit(vel_text, (self.sim_x_offset + 20, table_y + 55))
        # screen.blit(energy_text, (self.sim_x_offset + 20, table_y + 75))

class GreenSplitOperator(SimulacionBase):
    """
    Función de Green - Método Split-Operator
    Propagación numérica completa con estado inicial |0⟩
    """
    def __init__(self, width, height, x_offset, y_offset):
        super().__init__(width, height, x_offset, y_offset)
        self.omega = 1.0
        
        # Parámetros de la fuerza
        self.F0 = 0.8
        self.nu = 0.7
        
        # Control de fuerza
        self.force_on = True
        
        self.R_world = 4.0
        self.zoom = 1.0
        
        # Base de Fock
        self.n_basis = 30  # Más estados para mejor precisión
        self.psi = None
        self.inicializar_estado()
        
        self.nombre_simulacion = "green_split_operator"
        
        self.force_history = []
        self.energy_history = []
        
    def world_bounds(self):
        r = self.R_world / self.zoom
        return (-r, r, -r, r)
    
    def inicializar_estado(self):
        """Inicializa el estado fundamental |0⟩"""
        self.psi = np.zeros(self.n_basis, dtype=complex)
        self.psi[0] = 1.0  # Estado fundamental
        print(f"\n*** Estado inicial: |0⟩ (ground state) ***")
    
    def fuerza_externa(self, t):
        if self.force_on:
            return self.F0 * np.cos(self.nu * t)
        return 0.0
    
    def hamiltoniano_libre(self):
        """Hamiltoniano del oscilador libre"""
        H0 = np.zeros((self.n_basis, self.n_basis), dtype=complex)
        for n in range(self.n_basis):
            H0[n, n] = self.omega * (n + 0.5)
        return H0
    
    def operador_posicion_fock(self):
        """Operador posición en base de Fock"""
        X = np.zeros((self.n_basis, self.n_basis), dtype=complex)
        for n in range(self.n_basis - 1):
            X[n, n+1] = np.sqrt(n + 1) / np.sqrt(2)
            X[n+1, n] = np.sqrt(n + 1) / np.sqrt(2)
        return X
    
    def operador_momento_fock(self):
        """Operador momento en base de Fock"""
        P = np.zeros((self.n_basis, self.n_basis), dtype=complex)
        for n in range(self.n_basis - 1):
            P[n, n+1] = 1j * np.sqrt(n + 1) / np.sqrt(2)
            P[n+1, n] = -1j * np.sqrt(n + 1) / np.sqrt(2)
        return P
    
    def propagador_split_operator(self, dt):
        """Split-operator: U = e^(-iH₀dt/2) e^(-iFXdt) e^(-iH₀dt/2)"""
        H0 = self.hamiltoniano_libre()
        U_half = expm(-1j * H0 * dt / 2)
        
        F = self.fuerza_externa(self.t)
        X = self.operador_posicion_fock()
        U_interaction = expm(-1j * F * X * dt)
        
        # Propagación
        self.psi = U_half @ U_interaction @ U_half @ self.psi
        
        # Normalizar
        norm = np.sqrt(np.vdot(self.psi, self.psi))
        if norm > 1e-10:
            self.psi = self.psi / norm
    
    def calcular_observables(self):
        """Calcula ⟨X⟩, ⟨P⟩, Σ"""
        X_op = self.operador_posicion_fock()
        P_op = self.operador_momento_fock()
        
        X_avg = np.real(np.vdot(self.psi, X_op @ self.psi))
        P_avg = np.real(np.vdot(self.psi, P_op @ self.psi))
        
        X2_avg = np.real(np.vdot(self.psi, X_op @ X_op @ self.psi))
        P2_avg = np.real(np.vdot(self.psi, P_op @ P_op @ self.psi))
        XP_avg = np.real(np.vdot(self.psi, X_op @ P_op @ self.psi))
        
        Sigma = np.array([
            [X2_avg - X_avg**2, XP_avg - X_avg*P_avg],
            [XP_avg - X_avg*P_avg, P2_avg - P_avg**2]
        ])
        
        # Energía
        H = self.hamiltoniano_libre()
        F = self.fuerza_externa(self.t)
        H_total = H - F * X_op
        E = np.real(np.vdot(self.psi, H_total @ self.psi))
        
        return X_avg, P_avg, Sigma, E
    
    def update(self, speed):
        self.propagador_split_operator(self.dt * speed)
        self.t += self.dt * speed
        
        _, _, _, E = self.calcular_observables()
        
        self.force_history.append(self.fuerza_externa(self.t))
        self.energy_history.append(E)
        
        if len(self.force_history) > self.max_trail:
            self.force_history.pop(0)
            self.energy_history.pop(0)
        
        if GUARDAR_AUTOMATICO:
            self.guardar_datos_frame()
    
    def toggle_force(self):
        """Cambia el estado de la fuerza"""
        self.force_on = not self.force_on
        status = "ON" if self.force_on else "OFF"
        X, P, _, E = self.calcular_observables()
        print(f"\n*** Fuerza {status} en t={self.t:.2f} ***")
        print(f"    Estado: ⟨X⟩={X:.4f}, ⟨P⟩={P:.4f}, E={E:.4f}")
    
    def guardar_datos_frame(self):
        X, P, Sigma, E = self.calcular_observables()
        
        delta_X = np.sqrt(max(Sigma[0, 0], 1e-10))
        delta_P = np.sqrt(max(Sigma[1, 1], 1e-10))
        
        F_t = self.fuerza_externa(self.t)
        
        # Calcular alpha (para compatibilidad con analítico)
        alpha = (X + 1j*P) / np.sqrt(2)
        
        # Energías
        X2 = Sigma[0,0] + X**2
        P2 = Sigma[1,1] + P**2
        T_cinetica = P2 / 2.0
        V_potencial = (self.omega**2) * X2 / 2.0
        E_total = T_cinetica + V_potencial
        
        # Trabajo y potencia
        trabajo_instantaneo = F_t * X
        potencia = F_t * P
        
        # Pureza y entropía
        det_Sigma = np.linalg.det(Sigma)
        pureza = 1.0 / (2.0 * np.sqrt(det_Sigma + 1e-10))
        
        nu_entropy = np.sqrt(det_Sigma)
        if nu_entropy > 0.5:
            entropia = ((nu_entropy + 0.5) * np.log(nu_entropy + 0.5) - 
                       (nu_entropy - 0.5) * np.log(nu_entropy - 0.5))
        else:
            entropia = 0.0
        
        # Autovalores de Sigma
        eigenvalues = np.linalg.eigvalsh(Sigma)
        lambda_1 = max(eigenvalues)
        lambda_2 = min(eigenvalues)
        
        # Ángulo de la elipse
        if abs(Sigma[0,0] - Sigma[1,1]) > 1e-10:
            theta_ellipse = 0.5 * np.arctan2(2*Sigma[0,1], Sigma[0,0] - Sigma[1,1])
        else:
            theta_ellipse = 0.0
        
        # Excentricidad
        excentricidad = np.sqrt(1.0 - (lambda_2 / (lambda_1 + 1e-10))**2)
        
        # Área de la elipse
        area_elipse = np.pi * np.sqrt(det_Sigma)
        
        # Frame y periodos
        frame_number = int(self.t / self.dt)
        periodo_oscilador = int(self.omega * self.t / (2 * np.pi))
        periodo_fuerza = int(self.nu * self.t / (2 * np.pi))
        
        self.datos_csv.append({
            'tiempo': self.t,
            'frame_number': frame_number,
            'periodo_oscilador': periodo_oscilador,
            'periodo_fuerza': periodo_fuerza,
            'X_avg': X,
            'P_avg': P,
            'delta_X': delta_X,
            'delta_P': delta_P,
            'producto_incerteza': delta_X * delta_P,
            'alpha_real': alpha.real,
            'alpha_imag': alpha.imag,
            'alpha_magnitud': abs(alpha),
            'alpha_fase': np.angle(alpha),
            'fuerza_externa': F_t,
            'F0_amplitud': self.F0,
            'nu_frecuencia': self.nu,
            'omega_oscilador': self.omega,
            'energia_cinetica': T_cinetica,
            'energia_potencial': V_potencial,
            'energia_total': E_total,
            'trabajo_instantaneo': trabajo_instantaneo,
            'potencia': potencia,
            'pureza': pureza,
            'entropia': entropia,
            'Sigma_XX': Sigma[0, 0],
            'Sigma_XP': Sigma[0, 1],
            'Sigma_PP': Sigma[1, 1],
            'det_Sigma': det_Sigma,
            'traza_Sigma': np.trace(Sigma),
            'lambda_1': lambda_1,
            'lambda_2': lambda_2,
            'theta_ellipse': theta_ellipse,
            'excentricidad': excentricidad,
            'area_elipse': area_elipse,
            'force_activa': 1 if self.force_on else 0,
            'metodo': 'split_operator',
            'n_basis': self.n_basis
        })
    
    def draw(self, screen):
        # Panel lateral
        panel_x = 20
        panel_y = self.sim_y_offset + 20
        panel_w = 300

        # Titulo 
        font = pygame.font.SysFont(None, 28)
        font_small = pygame.font.SysFont(None, 24)
        title = font.render("Green: Split-Operator", True, BLACK)
        info = font.render(f"t={self.t:.2f} | n_basis={self.n_basis}", True, BLACK)
        screen.blit(title, (panel_x, panel_y))
        screen.blit(info, (panel_x, panel_y + 20))

        # Areá de simulación
        self.draw_axes(screen)
        
        X, P, Sigma, E = self.calcular_observables()
        center = (X, P)
        
        px, py = self.world_to_screen(X, P)
        self.trail.append((px, py))
        if len(self.trail) > self.max_trail:
            self.trail.pop(0)
        
        if len(self.trail) > 1:
            pygame.draw.lines(screen, CYAN, False, self.trail, 2)
        
        self.draw_sigma_ellipse(screen, Sigma, (X, P), GREEN)
        pygame.draw.circle(screen, BLACK, (px, py), 5)

        # Datos en panel lateral
        delta_X = np.sqrt(max(Sigma[0, 0], 1e-10))
        delta_P = np.sqrt(max(Sigma[1, 1], 1e-10))
        
        # Vector de fuerza
        F_t = self.fuerza_externa(self.t)
        if self.force_on and abs(F_t) > 0.01:
            scale_force = 50
            force_x = px + int(F_t * scale_force)
            pygame.draw.line(screen, RED, (px, py), (force_x, py), 3)
            pygame.draw.circle(screen, RED, (force_x, py), 4)
        
        esperado_x = font_small.render(f"⟨X⟩ = {X:.4f}", True, BLACK)
        esperado_p = font_small.render(f"⟨P⟩ = {P:.4f}", True, BLACK)
        incerteza_x = font_small.render(f"ΔX = {delta_X:.4f}", True, BLACK)
        incerteza_p = font_small.render(f"ΔP = {delta_P:.4f}", True, BLACK)
        producto = font_small.render(f"ΔX·ΔP = {delta_X*delta_P:.4f}", True, BLACK)
        energia_text = font_small.render(f"E = {E:.4f}", True, BLACK)
        force_text = font_small.render(f"F(t) = {F_t:.4f}", True, RED if self.force_on else GRAY)
        
        # font = pygame.font.SysFont(None, 24)
        # font_small = pygame.font.SysFont(None, 18)
        # title = font.render("Green: Split-Operator", True, BLACK)
        # info = font.render(f"t={self.t:.2f} | n_basis={self.n_basis}", True, BLACK)
        # screen.blit(title, (self.sim_x_offset + 10, self.sim_y_offset + 10))
        # screen.blit(info, (self.sim_x_offset + 10, self.sim_y_offset + 35))
        
        # table_y = self.sim_y_offset + self.sim_size - 340
        table_y = panel_y + 70
        pygame.draw.rect(screen, (240, 240, 240), (panel_x, table_y, panel_w, 190)) # Controla el tamaño del marco de información (x, y, ancho, alto)
        pygame.draw.rect(screen, BLACK, (panel_x, table_y, panel_w, 190), 2)
        
        table_title = font.render("Observables", True, BLACK)
        screen.blit(table_title, (panel_x + 20, table_y + 10))
        
        screen.blit(esperado_x, (panel_x + 20, table_y + 40))
        screen.blit(esperado_p, (panel_x + 20, table_y + 60))
        screen.blit(incerteza_x, (panel_x + 20, table_y + 80))
        screen.blit(incerteza_p, (panel_x + 20, table_y + 100))
        screen.blit(producto, (panel_x + 20, table_y + 120))
        screen.blit(energia_text, (panel_x + 20, table_y + 140))
        screen.blit(force_text, (panel_x + 20, table_y + 160))
        
        # pygame.draw.rect(screen, (240, 240, 240), (self.sim_x_offset + 10, table_y, 280, 180))
        # pygame.draw.rect(screen, BLACK, (self.sim_x_offset + 10, table_y, 280, 180), 2)

        # table_title = font.render("Observables", True, BLACK)
        # screen.blit(table_title, (self.sim_x_offset + 20, table_y + 10))
        
        # screen.blit(esperado_x, (self.sim_x_offset + 20, table_y + 40))
        # screen.blit(esperado_p, (self.sim_x_offset + 20, table_y + 60))
        # screen.blit(incerteza_x, (self.sim_x_offset + 20, table_y + 80))
        # screen.blit(incerteza_p, (self.sim_x_offset + 20, table_y + 100))
        # screen.blit(producto, (self.sim_x_offset + 20, table_y + 120))
        # screen.blit(energia_text, (self.sim_x_offset + 20, table_y + 140))
        # screen.blit(force_text, (self.sim_x_offset + 20, table_y + 160))
        
        # Parámetros
        param_y = table_y + 200
        pygame.draw.rect(screen, (250, 240, 230), (panel_x + 10, param_y, panel_w, 140))
        pygame.draw.rect(screen, BLACK, (panel_x, param_y, panel_w, 140), 2)
        
        param_title = font.render("Parámetros", True, BLACK)
        screen.blit(param_title, (panel_x + 20, param_y + 10))
        
        omega_text = font_small.render(f"ω = {self.omega:.4f}", True, BLACK)
        nu_text = font_small.render(f"ν = {self.nu:.4f}", True, BLACK)
        F0_text = font_small.render(f"F₀ = {self.F0:.4f}", True, BLACK)
        resonance = abs(self.omega - self.nu)
        res_text = font_small.render(f"|ω-ν| = {resonance:.4f}", True, BLACK)
        force_status = font_small.render(f"Fuerza: {'ON' if self.force_on else 'OFF'}", 
                                        True, GREEN if self.force_on else RED)
        
        screen.blit(omega_text, (panel_x + 20, param_y + 35))
        screen.blit(nu_text, (panel_x + 20, param_y + 55))
        screen.blit(F0_text, (panel_x + 20, param_y + 75))
        screen.blit(res_text, (panel_x + 20, param_y + 95))
        screen.blit(force_status, (panel_x + 20, param_y + 115))
        
        # Gráfico F(t) - Historia del Drive
        if len(self.force_history) > 1:
            self._draw_force_history(screen, panel_x, param_y + 150)
    
    def _draw_force_history(self, screen, x, y):
        """Dibuja gráfico F(t) en el panel - Estilo Drive"""
        graph_w = 280
        graph_h = 80
        
        pygame.draw.rect(screen, (250, 250, 250), (x, y, graph_w, graph_h))
        pygame.draw.rect(screen, BLACK, (x, y, graph_w, graph_h), 2)
        
        font_tiny = pygame.font.SysFont(None, 16)
        title = font_tiny.render("F(t) - Historia del Drive", True, BLACK)
        screen.blit(title, (x + 5, y + 5))
        
        forces = np.array(self.force_history)
        if len(forces) > 0:
            f_max = max(abs(forces.max()), abs(forces.min()), 0.1)
            
            zero_y = y + graph_h // 2
            pygame.draw.line(screen, (180, 180, 180), (x, zero_y), (x + graph_w, zero_y), 1)
            
            points = []
            for i, f in enumerate(forces):
                px = x + int((i / len(forces)) * graph_w)
                py = zero_y - int((f / f_max) * (graph_h // 2 - 10))
                points.append((px, py))
            
            if len(points) > 1:
                color = RED if self.force_on else GRAY
                pygame.draw.lines(screen, color, False, points, 2)
    
    def draw_sigma_ellipse(self, surface, Sigma, center_world, color, segs=120):
        vals, vecs = np.linalg.eigh(Sigma)
        vals = np.clip(vals, 1e-8, None)
        radii = np.sqrt(vals)
        phi = np.linspace(0, 2*np.pi, segs, endpoint=True)
        circle = np.stack([np.cos(phi), np.sin(phi)], axis=0)
        pts = (vecs @ (radii[:,None] * circle))
        pts = pts.T + np.array(center_world)[None,:]
        pts_px = [self.world_to_screen(px, py) for px, py in pts]
        pygame.draw.lines(surface, color, True, pts_px, 2)

class GreenAnalitico(SimulacionBase):
    """
    Función de Green - Método Analítico
    Solución analítica con acumulación correcta del estado
    """
    def __init__(self, width, height, x_offset, y_offset):
        super().__init__(width, height, x_offset, y_offset)
        self.omega = 1.0
        
        # Parámetros de la fuerza
        self.F0 = 0.8
        self.nu = 0.7
        
        # Control de fuerza
        self.force_on = True
        
        # Estado acumulado (física correcta)
        self.alpha_acumulado = 0.0 + 0.0j
        self.t_last_toggle = 0.0
        
        self.R_world = 4.0
        self.zoom = 1.0
        
        self.Sigma0 = np.eye(2) * 0.5
        self.nombre_simulacion = "green_analitico"
        
        self.force_history = []
        self.alpha_history = []
        
    def world_bounds(self):
        r = self.R_world / self.zoom
        return (-r, r, -r, r)
    
    def fuerza_externa(self, t):
        if self.force_on:
            return self.F0 * np.cos(self.nu * t)
        return 0.0
    
    def alpha_respuesta_fuerza(self, t_inicio, t_actual):
        """
        Respuesta de la función de Green desde t_inicio hasta t_actual
        """
        dt = t_actual - t_inicio
        
        if abs(self.omega**2 - self.nu**2) < 1e-6:
            # Caso resonante
            factor = self.F0 / (np.sqrt(2) * 2 * self.omega)
            alpha_forced = factor * dt * np.sin(self.omega * t_actual) * np.exp(-1j * self.omega * t_actual)
            return alpha_forced
        else:
            # Caso no resonante
            factor = self.F0 / (np.sqrt(2) * (self.omega**2 - self.nu**2))
            
            # Respuesta en t_actual
            alpha_forced_now = factor * np.cos(self.nu * t_actual)
            alpha_free_now = -factor * np.cos(self.omega * t_actual)
            
            # Respuesta en t_inicio
            alpha_forced_start = factor * np.cos(self.nu * t_inicio)
            alpha_free_start = -factor * np.cos(self.omega * t_inicio)
            
            # Diferencia (respuesta acumulada)
            delta_alpha = (alpha_forced_now - alpha_forced_start + 
                          alpha_free_now - alpha_free_start)
            
            return delta_alpha * np.exp(-1j * self.omega * t_actual)
    
    def update(self, speed):
        t_prev = self.t
        self.t += self.dt * speed
        
        if self.force_on:
            # Con fuerza: acumular respuesta desde último toggle
            # Fase libre + nueva contribución
            phase_evolution = np.exp(-1j * self.omega * self.dt * speed)
            new_contribution = self.alpha_respuesta_fuerza(t_prev, self.t)
            self.alpha_acumulado = self.alpha_acumulado * phase_evolution + new_contribution
        else:
            # Sin fuerza: solo evolución libre (rotación)
            self.alpha_acumulado = self.alpha_acumulado * np.exp(-1j * self.omega * self.dt * speed)
        
        self.alpha_history.append(self.alpha_acumulado)
        self.force_history.append(self.fuerza_externa(self.t))
        
        if len(self.alpha_history) > self.max_trail:
            self.alpha_history.pop(0)
            self.force_history.pop(0)
        
        if GUARDAR_AUTOMATICO:
            self.guardar_datos_frame()
    
    def toggle_force(self):
        """Cambia el estado de la fuerza"""
        self.force_on = not self.force_on
        self.t_last_toggle = self.t
        
        if self.force_on:
            print(f"\n*** Fuerza ACTIVADA en t={self.t:.2f} ***")
            print(f"    Estado actual: α = {abs(self.alpha_acumulado):.4f}")
        else:
            print(f"\n*** Fuerza DESACTIVADA en t={self.t:.2f} ***")
            print(f"    Estado acumulado: α = {abs(self.alpha_acumulado):.4f}")
            print(f"    → Evolución libre desde ahora")
    
    def get_position_momentum(self):
        X = np.sqrt(2) * self.alpha_acumulado.real
        P = np.sqrt(2) * self.alpha_acumulado.imag
        return X, P
    
    def get_covariance_matrix(self):
        c, s = np.cos(self.omega * self.t), np.sin(self.omega * self.t)
        R = np.array([[c, -s], [s, c]])
        return R @ self.Sigma0 @ R.T
    
    def guardar_datos_frame(self):
        X, P = self.get_position_momentum()
        Sigma = self.get_covariance_matrix()
        
        delta_X = np.sqrt(Sigma[0, 0])
        delta_P = np.sqrt(Sigma[1, 1])
        
        F_t = self.fuerza_externa(self.t)
        
        # Energías
        X2 = Sigma[0,0] + X**2
        P2 = Sigma[1,1] + P**2
        T_cinetica = P2 / 2.0
        V_potencial = (self.omega**2) * X2 / 2.0
        E_total = T_cinetica + V_potencial
        
        # Trabajo y potencia
        trabajo_instantaneo = F_t * X
        potencia = F_t * P
        
        # Pureza y entropía
        det_Sigma = np.linalg.det(Sigma)
        pureza = 1.0 / (2.0 * np.sqrt(det_Sigma + 1e-10))
        
        nu_entropy = np.sqrt(det_Sigma)
        if nu_entropy > 0.5:
            entropia = ((nu_entropy + 0.5) * np.log(nu_entropy + 0.5) - 
                       (nu_entropy - 0.5) * np.log(nu_entropy - 0.5))
        else:
            entropia = 0.0
        
        # Autovalores de Sigma
        eigenvalues = np.linalg.eigvalsh(Sigma)
        lambda_1 = max(eigenvalues)
        lambda_2 = min(eigenvalues)
        
        # Ángulo de la elipse
        if abs(Sigma[0,0] - Sigma[1,1]) > 1e-10:
            theta_ellipse = 0.5 * np.arctan2(2*Sigma[0,1], Sigma[0,0] - Sigma[1,1])
        else:
            theta_ellipse = 0.0
        
        # Excentricidad
        excentricidad = np.sqrt(1.0 - (lambda_2 / (lambda_1 + 1e-10))**2)
        
        # Área de la elipse
        area_elipse = np.pi * np.sqrt(det_Sigma)
        
        # Frame y periodos
        frame_number = int(self.t / self.dt)
        periodo_oscilador = int(self.omega * self.t / (2 * np.pi))
        periodo_fuerza = int(self.nu * self.t / (2 * np.pi))
        
        self.datos_csv.append({
            'tiempo': self.t,
            'frame_number': frame_number,
            'periodo_oscilador': periodo_oscilador,
            'periodo_fuerza': periodo_fuerza,
            'X_avg': X,
            'P_avg': P,
            'delta_X': delta_X,
            'delta_P': delta_P,
            'producto_incerteza': delta_X * delta_P,
            'alpha_real': self.alpha_acumulado.real,
            'alpha_imag': self.alpha_acumulado.imag,
            'alpha_magnitud': abs(self.alpha_acumulado),
            'alpha_fase': np.angle(self.alpha_acumulado),
            'fuerza_externa': F_t,
            'F0_amplitud': self.F0,
            'nu_frecuencia': self.nu,
            'omega_oscilador': self.omega,
            'energia_cinetica': T_cinetica,
            'energia_potencial': V_potencial,
            'energia_total': E_total,
            'trabajo_instantaneo': trabajo_instantaneo,
            'potencia': potencia,
            'pureza': pureza,
            'entropia': entropia,
            'Sigma_XX': Sigma[0, 0],
            'Sigma_XP': Sigma[0, 1],
            'Sigma_PP': Sigma[1, 1],
            'det_Sigma': det_Sigma,
            'traza_Sigma': np.trace(Sigma),
            'lambda_1': lambda_1,
            'lambda_2': lambda_2,
            'theta_ellipse': theta_ellipse,
            'excentricidad': excentricidad,
            'area_elipse': area_elipse,
            'force_activa': 1 if self.force_on else 0,
            'metodo': 'analitico'
        })
    
    def draw(self, screen):
        # Panel lateral
        panel_x = 20
        panel_y = self.sim_y_offset + 10
        panel_w = 300
        
        self.draw_axes(screen)
        X, P = self.get_position_momentum()
        Sigma = self.get_covariance_matrix()
        
        px, py = self.world_to_screen(X, P)
        self.trail.append((px, py))
        if len(self.trail) > self.max_trail:
            self.trail.pop(0)
        
        if len(self.trail) > 1:
            pygame.draw.lines(screen, ORANGE, False, self.trail, 2)
        
        self.draw_sigma_ellipse(screen, Sigma, (X, P), GREEN)
        pygame.draw.circle(screen, BLACK, (px, py), 5)
        
        # Vector de fuerza
        F_t = self.fuerza_externa(self.t)
        if self.force_on and abs(F_t) > 0.01:
            scale_force = 50
            force_x = px + int(F_t * scale_force)
            pygame.draw.line(screen, RED, (px, py), (force_x, py), 3)
            pygame.draw.circle(screen, RED, (force_x, py), 4)
        
        font = pygame.font.SysFont(None, 28)
        font_small = pygame.font.SysFont(None, 26)
        
        title = font.render("Green: Analítico", True, BLACK)
        info = font.render(f"t={self.t:.2f} | Método cerrado", True, BLACK)
        screen.blit(title, (panel_x, panel_y + 10))
        screen.blit(info, (panel_x, panel_y + 35))
        
        delta_X = np.sqrt(Sigma[0, 0])
        delta_P = np.sqrt(Sigma[1, 1])
        
        X2 = Sigma[0,0] + X**2
        P2 = Sigma[1,1] + P**2
        E = (P2 + (self.omega**2) * X2) / 2.0
        
        table_y = panel_y + 70
        pygame.draw.rect(screen, (240, 240, 240), (panel_x, table_y, panel_w, 240))
        pygame.draw.rect(screen, BLACK, (panel_x, table_y, panel_w, 240), 2)
        
        table_title = font.render("Observables", True, BLACK)
        screen.blit(table_title, (panel_x + 20, table_y + 10))
        
        esperado_x = font_small.render(f"⟨X⟩ = {X:.4f}", True, BLACK)
        esperado_p = font_small.render(f"⟨P⟩ = {P:.4f}", True, BLACK)
        incerteza_x = font_small.render(f"ΔX = {delta_X:.4f}", True, BLACK)
        incerteza_p = font_small.render(f"ΔP = {delta_P:.4f}", True, BLACK)
        producto = font_small.render(f"ΔX·ΔP = {delta_X*delta_P:.4f}", True, BLACK)
        energia_text = font_small.render(f"E = {E:.4f}", True, BLACK)
        alpha_text = font_small.render(f"|α| = {abs(self.alpha_acumulado):.4f}", True, BLACK)
        fase_text = font_small.render(f"φ = {np.angle(self.alpha_acumulado):.4f} rad", True, BLACK)
        force_text = font_small.render(f"F(t) = {F_t:.4f}", True, RED if self.force_on else GRAY)
        
        screen.blit(esperado_x, (panel_x + 20, table_y + 40))
        screen.blit(esperado_p, (panel_x + 20, table_y + 65))
        screen.blit(incerteza_x, (panel_x + 20, table_y + 90))
        screen.blit(incerteza_p, (panel_x + 20, table_y + 115))
        screen.blit(producto, (panel_x + 20, table_y + 140))
        screen.blit(energia_text, (panel_x + 20, table_y + 165))
        screen.blit(alpha_text, (panel_x + 20, table_y + 190))
        screen.blit(fase_text, (panel_x + 20, table_y + 215))
        
        # Parámetros
        param_y = table_y + 250
        pygame.draw.rect(screen, (250, 240, 230), (panel_x, param_y, 280, 140))
        pygame.draw.rect(screen, BLACK, (panel_x, param_y, panel_w, 140), 2)
        
        param_title = font.render("Parámetros", True, BLACK)
        screen.blit(param_title, (panel_x + 20, param_y + 10))
        
        omega_text = font_small.render(f"ω = {self.omega:.4f}", True, BLACK)
        nu_text = font_small.render(f"ν = {self.nu:.4f}", True, BLACK)
        F0_text = font_small.render(f"F₀ = {self.F0:.4f}", True, BLACK)
        resonance = abs(self.omega - self.nu)
        res_text = font_small.render(f"|ω-ν| = {resonance:.4f}", True, BLACK)
        force_status = font_small.render(f"Fuerza: {'ON' if self.force_on else 'OFF'}", 
                                        True, GREEN if self.force_on else RED)
        
        screen.blit(omega_text, (panel_x + 20, param_y + 35))
        screen.blit(nu_text, (panel_x + 20, param_y + 55))
        screen.blit(F0_text, (panel_x + 20, param_y + 75))
        screen.blit(res_text, (panel_x + 20, param_y + 95))
        screen.blit(force_status, (panel_x + 20, param_y + 115))
        
        # Gráfico F(t) - Historia del Drive
        if len(self.force_history) > 1:
            self._draw_force_history(screen, panel_x, param_y + 150)
    
    def _draw_force_history(self, screen, x, y):
        """Dibuja gráfico F(t) en el panel - Estilo Drive"""
        graph_w = 280
        graph_h = 80
        
        pygame.draw.rect(screen, (250, 250, 250), (x, y, graph_w, graph_h))
        pygame.draw.rect(screen, BLACK, (x, y, graph_w, graph_h), 2)
        
        font_tiny = pygame.font.SysFont(None, 16)
        title = font_tiny.render("F(t) - Historia del Drive", True, BLACK)
        screen.blit(title, (x + 5, y + 5))
        
        forces = np.array(self.force_history)
        if len(forces) > 0:
            f_max = max(abs(forces.max()), abs(forces.min()), 0.1)
            
            zero_y = y + graph_h // 2
            pygame.draw.line(screen, (180, 180, 180), (x, zero_y), (x + graph_w, zero_y), 1)
            
            points = []
            for i, f in enumerate(forces):
                px = x + int((i / len(forces)) * graph_w)
                py = zero_y - int((f / f_max) * (graph_h // 2 - 10))
                points.append((px, py))
            
            if len(points) > 1:
                color = RED if self.force_on else GRAY
                pygame.draw.lines(screen, color, False, points, 2)
    
    def draw_sigma_ellipse(self, surface, Sigma, center_world, color, segs=120):
        vals, vecs = np.linalg.eigh(Sigma)
        vals = np.clip(vals, 1e-8, None)
        radii = np.sqrt(vals)
        phi = np.linspace(0, 2*np.pi, segs, endpoint=True)
        circle = np.stack([np.cos(phi), np.sin(phi)], axis=0)
        pts = (vecs @ (radii[:,None] * circle))
        pts = pts.T + np.array(center_world)[None,:]
        pts_px = [self.world_to_screen(px, py) for px, py in pts]
        pygame.draw.lines(surface, color, True, pts_px, 2)

def draw_tabs(screen, active_tab):
    tab_width = WIDTH // 6
    font = pygame.font.SysFont(None, 16)

    tabs = [
        (1, "1. Coherente"),
        (2, "2. Comprimido"),
        (3, "3. Superposición"),
        (4, "4. Clásico"),
        (5, "5. Green Split-Op"),
        (6, "6. Green Analítico")
    ]

    for idx, (tab_num, label) in enumerate(tabs):
        tab_color = LIGHT_GRAY if active_tab == tab_num else GRAY
        x = idx * tab_width
        pygame.draw.rect(screen, tab_color, (x, 0, tab_width, TAB_HEIGHT))
        pygame.draw.rect(screen, BLACK, (x, 0, tab_width, TAB_HEIGHT), 2)

        text = font.render(label, True, BLACK)
        text_rect = text.get_rect(center=(x + tab_width // 2, TAB_HEIGHT // 2))
        screen.blit(text, text_rect)

def get_clicked_tab(mouse_pos):
    x, y = mouse_pos
    if y <= TAB_HEIGHT:
        tab_width = WIDTH // 6
        return min(6, (x // tab_width) + 1)
    return None

def main():
    global GUARDAR_AUTOMATICO

    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Simulación Oscilador Armónico Cuántico")
    clock = pygame.time.Clock()

    content_height = HEIGHT - TAB_HEIGHT
    estado_cuantico = EstadoCoherente(WIDTH, content_height, 0, TAB_HEIGHT)
    estado_comprimido = EstadoComprimido(WIDTH, content_height, 0, TAB_HEIGHT)
    superposicion = SuperposicionEstados(WIDTH, content_height, 0, TAB_HEIGHT)
    oscilador_clasico = OsciladorClasico(WIDTH, content_height, 0, TAB_HEIGHT)
    green_split = GreenSplitOperator(WIDTH, content_height, 0, TAB_HEIGHT)
    green_analitico = GreenAnalitico(WIDTH, content_height, 0, TAB_HEIGHT)

    simulaciones = [estado_cuantico, estado_comprimido, superposicion, oscilador_clasico, green_split, green_analitico]

    active_tab = 1
    prev_tab = 1
    running = True

    font = pygame.font.SysFont(None, 18)

    print("=== SIMULACIÓN QHO COMPLETA ===")
    print("G: guardado | E: exportar | D: drive (1-3) | F: fuerza (5-6)")
    print("+/-: ajustar ν | [/]: ajustar F₀ | R: reset (5-6)")
    print("1/2/3/4/5/6: pestañas | ESC: salir")
    print("\nPestaña 5: Split-Operator (numérico)")
    print("Pestaña 6: Analítico (solución cerrada)")

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_g:
                    GUARDAR_AUTOMATICO = not GUARDAR_AUTOMATICO
                    if GUARDAR_AUTOMATICO:
                        print("\n*** GUARDADO AUTOMÁTICO ACTIVADO ***")
                        for sim in simulaciones:
                            sim.datos_csv = []
                    else:
                        print("\n*** GUARDADO AUTOMÁTICO DESACTIVADO ***")

                elif event.key == pygame.K_e:
                    simulaciones[active_tab - 1].exportar_csv()

                elif event.key == pygame.K_d:
                    estado_cuantico.drive_on = not estado_cuantico.drive_on
                    estado_comprimido.drive_on = not estado_comprimido.drive_on
                    superposicion.drive_on = not superposicion.drive_on
                    status = "ON" if estado_cuantico.drive_on else "OFF"
                    print(f"\n*** Drive {status} (tabs 1-3) ***")

                elif event.key == pygame.K_f:
                    if active_tab == 5:
                        green_split.toggle_force()
                    elif active_tab == 6:
                        green_analitico.toggle_force()

                elif event.key == pygame.K_EQUALS or event.key == pygame.K_PLUS:
                    if active_tab == 5:
                        green_split.nu += 0.1
                        print(f"ν = {green_split.nu:.2f}")
                    elif active_tab == 6:
                        green_analitico.nu += 0.1
                        print(f"ν = {green_analitico.nu:.2f}")

                elif event.key == pygame.K_MINUS:
                    if active_tab == 5:
                        green_split.nu = max(0.1, green_split.nu - 0.1)
                        print(f"ν = {green_split.nu:.2f}")
                    elif active_tab == 6:
                        green_analitico.nu = max(0.1, green_analitico.nu - 0.1)
                        print(f"ν = {green_analitico.nu:.2f}")

                elif event.key == pygame.K_LEFTBRACKET:
                    if active_tab == 5:
                        green_split.F0 = max(0.0, green_split.F0 - 0.1)
                        print(f"F₀ = {green_split.F0:.2f}")
                    elif active_tab == 6:
                        green_analitico.F0 = max(0.0, green_analitico.F0 - 0.1)
                        print(f"F₀ = {green_analitico.F0:.2f}")

                elif event.key == pygame.K_RIGHTBRACKET:
                    if active_tab == 5:
                        green_split.F0 += 0.1
                        print(f"F₀ = {green_split.F0:.2f}")
                    elif active_tab == 6:
                        green_analitico.F0 += 0.1
                        print(f"F₀ = {green_analitico.F0:.2f}")

                elif event.key == pygame.K_r:
                    if active_tab == 5:
                        green_split.t = 0.0
                        green_split.trail = []
                        green_split.force_history = []
                        green_split.energy_history = []
                        green_split.inicializar_estado()
                        print("\n*** Sistema reseteado (Split-Op) ***")
                    elif active_tab == 6:
                        green_analitico.t = 0.0
                        green_analitico.trail = []
                        green_analitico.force_history = []
                        green_analitico.alpha_history = []
                        green_analitico.alpha_acumulado = 0.0 + 0.0j
                        green_analitico.t_last_toggle = 0.0
                        print("\n*** Sistema reseteado (Analítico) ***")

                elif event.key == pygame.K_ESCAPE:
                    running = False

                elif event.key == pygame.K_1:
                    prev_tab = active_tab
                    active_tab = 1
                elif event.key == pygame.K_2:
                    prev_tab = active_tab
                    active_tab = 2
                elif event.key == pygame.K_3:
                    prev_tab = active_tab
                    active_tab = 3
                elif event.key == pygame.K_4:
                    prev_tab = active_tab
                    active_tab = 4
                elif event.key == pygame.K_5:
                    prev_tab = active_tab
                    active_tab = 5
                elif event.key == pygame.K_6:
                    prev_tab = active_tab
                    active_tab = 6

            elif event.type == pygame.MOUSEBUTTONDOWN:
                clicked_tab = get_clicked_tab(event.pos)
                if clicked_tab:
                    prev_tab = active_tab
                    active_tab = clicked_tab

        if active_tab != prev_tab and GUARDAR_AUTOMATICO:
            sim_prev = simulaciones[prev_tab - 1]
            if sim_prev.datos_csv:
                print(f"\n>>> Exportando datos automáticamente...")
                sim_prev.exportar_csv()
                sim_prev.datos_csv = []
            prev_tab = active_tab

        simulaciones[active_tab - 1].update(1.0)

        screen.fill(WHITE)
        draw_tabs(screen, active_tab)
        simulaciones[active_tab - 1].draw(screen)

        sim_actual = simulaciones[active_tab - 1]
        registros = len(sim_actual.datos_csv)

        controls = font.render("G: guardado | E: exportar | D: drive | F: fuerza | +/-: ν | [/]: F₀ | R: reset | 1-5 | ESC", True, BLACK)

        status_parts = [f"Guardado: {'ON' if GUARDAR_AUTOMATICO else 'OFF'}", f"Registros: {registros}"]

        if active_tab in [1, 2, 3]:
            status_parts.append(f"Drive: {'ON' if estado_cuantico.drive_on else 'OFF'}")
        elif active_tab == 5:
            status_parts.append(f"Fuerza: {'ON' if green_split.force_on else 'OFF'}")
            status_parts.append(f"n_basis={green_split.n_basis}")
        elif active_tab == 6:
            status_parts.append(f"Fuerza: {'ON' if green_analitico.force_on else 'OFF'}")
            status_parts.append(f"|α|={abs(green_analitico.alpha_acumulado):.4f}")

        status = font.render(" | ".join(status_parts), True, BLACK)

        screen.blit(controls, (10, HEIGHT - 40))
        screen.blit(status, (10, HEIGHT - 20))

        pygame.display.flip()
        clock.tick(60)

    if GUARDAR_AUTOMATICO:
        print("\n>>> Exportando datos finales...")
        for i, sim in enumerate(simulaciones):
            if sim.datos_csv:
                print(f"\nExportando pestaña {i+1}...")
                sim.exportar_csv()

    print("\n=== SIMULACIÓN TERMINADA ===")
    pygame.quit()

if __name__ == "__main__":
    main()
