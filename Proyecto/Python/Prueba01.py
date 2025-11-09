import sys, math, cmath, pygame
from collections import deque

# ---------------- Parámetros (Unidades adimensionales) ----------------
# m = ħ = 1.0, inicio con omega = 1.0
# Varianzas para un estado coherente DeltaX = DeltaP = 1 (1-sigma).

omega = 1.0
alpha0_mag = 1.5
alpha0_phase = math.pi/6
alpha0 = alpha0_mag * cmath.exp(1j * alpha0_phase)

# Parámetros de fuerza [k = 1/sqrt(2)]
drive_on = False
F0 = 0.6
nu = 1.0   # frecuencia de la fuerza
kappa = 1/math.sqrt(2.0)

# Control de tiempo
speed = 1.0  # Multiplicador de velocidad de simulación
dt_real = 1/60.0  # FPS de tiempo real (aprox)
paused = False

# Renderizado
W, H = 700, 700
margin_px = 60
bg_color = (255, 255, 255)
fg_color = (0, 0, 0)
traj_color = (20, 20, 20)
disk_color = (0, 0, 0)
axis_color = (0, 0, 0)
text_color = (0, 0, 0)

# Ventana de cuadratura (X,P). Se auto ajusta |alpha0|
def initial_world_radius():
    # Radio inicial para cubrir la trayectoria y el disco 1-sigma cómodamente
    return math.sqrt(2) * abs(alpha0) + 3.0

R_world = initial_world_radius()
zoom = 1.0  # Ajuste de zoom automático desactivado por ahora

# Traza de la trayectoria
max_pts = 4000
trail = deque(maxlen=max_pts)

# ---------------- Pygame setup ----------------
pygame.init()
screen = pygame.display.set_mode((W, H))
pygame.display.set_caption("QHO Coherent State — Quadrature Space")
clock = pygame.time.Clock()
font = pygame.font.SysFont(None, 18)

# ---------------- Funciones de dibujo ----------------
def world_bounds():
    r = R_world / zoom
    return (-r, r, -r, r)

def world_to_screen(x, y):
    xmin, xmax, ymin, ymax = world_bounds()
    sx = (W - 2*margin_px) / (xmax - xmin)
    sy = (H - 2*margin_px) / (ymax - ymin)
    px = margin_px + (x - xmin) * sx
    py = H - (margin_px + (y - ymin) * sy)
    return int(px), int(py)

def draw_axes(surface):
    xmin, xmax, ymin, ymax = world_bounds()
    # dibuja la caja exterior
    pygame.draw.rect(surface, axis_color, (0,0,W-1,H-1), 1)
    # dibuja los ejes
    px0, py0 = world_to_screen(0.0, 0.0)
    pygame.draw.line(surface, axis_color, (0, py0), (W, py0), 1)
    pygame.draw.line(surface, axis_color, (px0, 0), (px0, H), 1)
    # ticks
    for val in range(-10, 11):
        if val == 0:
            continue
        tx, ty = world_to_screen(val, 0.0)
        pygame.draw.line(surface, axis_color, (tx, py0-3), (tx, py0+3), 1)
        tx, ty = world_to_screen(0.0, val)
        pygame.draw.line(surface, axis_color, (px0-3, ty), (px0+3, ty), 1)

def draw_text(surface, text, pos):
    img = font.render(text, True, text_color)
    surface.blit(img, pos)

def draw_trail(surface, pts):
    if len(pts) > 1:
        pygame.draw.lines(surface, traj_color, False, pts, 2)

def draw_uncertainty_disk(surface, center_px, radius_world):
    # Convertir el radio del mundo a píxeles usando la escala X
    cx, cy = center_px
    rx_px, _ = world_to_screen(radius_world, 0.0)
    ox, _ = world_to_screen(0.0, 0.0)
    rad_px = abs(rx_px - ox)
    pygame.draw.circle(surface, disk_color, center_px, rad_px, 1)

# ---------------- Simulación de Estado ----------------
t = 0.0
I_drive = 0.0 + 0.0j  # acumulador integral para el término de la fuerza en alpha(t)

def forcing(t):
    return F0 * math.cos(nu * t)

def alpha_of_t(t, alpha0, omega, drive_on):
    global I_drive
    if not drive_on:
        return alpha0 * cmath.exp(-1j * omega * t)
    # Actualización de la integral numérica usando regla del rectángulo (dt definido afuera)
    # El bucle actualizará I_drive externamente en cada paso; aquí solo reconstruimos alpha.
    return cmath.exp(-1j * omega * t) * (alpha0 + kappa * I_drive)

# ---------------- Bucle principal ----------------
running = True
prev_t = t
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                running = False
            elif event.key == pygame.K_SPACE:
                paused = not paused
            elif event.key == pygame.K_LEFT:
                speed = max(0.125, speed/2.0)
            elif event.key == pygame.K_RIGHT:
                speed = min(16.0, speed*2.0)
            elif event.key == pygame.K_UP:
                alpha0_mag *= 1.1
                alpha0 = alpha0_mag * cmath.exp(1j * alpha0_phase)
                R_world = initial_world_radius()
            elif event.key == pygame.K_DOWN:
                alpha0_mag *= 0.9
                alpha0 = alpha0_mag * cmath.exp(1j * alpha0_phase)
                R_world = initial_world_radius()
            elif event.key == pygame.K_o:
                omega *= 0.9
            elif event.key == pygame.K_p:
                omega *= 1.1
            elif event.key == pygame.K_d:
                drive_on = not drive_on
                I_drive = 0.0 + 0.0j
                t = 0.0
                trail.clear()
            elif event.key == pygame.K_r:
                # reseteo de tiempo y trayectoria
                t = 0.0
                I_drive = 0.0 + 0.0j
                trail.clear()

    dt = clock.tick(60) / 1000.0  # Secgundos reales desde último frame
    if not paused:
        t_next = t + speed * dt

        # Actualiazación de la integral de la fuerza si está activada: I += e^{i w t} F(t) dt
        if drive_on:
            # Incremento usando regla del trapecio
            f_now = forcing(t)
            f_next = forcing(t_next)
            phase_now = cmath.exp(1j * omega * t)
            phase_next = cmath.exp(1j * omega * t_next)
            I_drive += 0.5 * (phase_now * f_now + phase_next * f_next) * (t_next - t)

        t = t_next

    # Alpha y cuadraturas
    a = alpha_of_t(t, alpha0, omega, drive_on)
    X = math.sqrt(2.0) * a.real
    P = math.sqrt(2.0) * a.imag

    trail.append(world_to_screen(X, P))

    # ---------------- Draw ----------------
    screen.fill(bg_color)
    draw_axes(screen)
    draw_trail(screen, list(trail))

    # Incerteza disco 1-sigma [radio = 1 en (X,P)]
    center_px = world_to_screen(X, P)
    draw_uncertainty_disk(screen, center_px, radius_world=1.0)

    # Marcador central
    pygame.draw.circle(screen, (0,0,0), center_px, 4)

    # Texto HUD
    hud1 = f"t={t:6.2f}  omega={omega:.3f}  |alpha0|={alpha0_mag:.3f}  phase={alpha0_phase:.2f} rad"
    hud2 = f"speed={speed:.2f}x  drive={'ON' if drive_on else 'OFF'}  F0={F0:.2f}  nu={nu:.2f}  zoom={zoom:.2f}"
    draw_text(screen, hud1, (10, 10))
    draw_text(screen, hud2, (10, 30))
    draw_text(screen, "Keys: SPACE pause | LEFT/RIGHT speed | UP/DOWN |alpha0| | O/P omega | D drive | R reset", (10, H-28))

    pygame.display.flip()

pygame.quit()
sys.exit(0)