import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, FancyArrow, Wedge
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider, Button
import matplotlib.patches as mpatches

# Create interactive F-14A adverse yaw simulator
class F14AdverseYawSimulator:
    def __init__(self):
        self.fig = plt.figure(figsize=(16, 10))
        self.fig.suptitle('F-14A Lateral-Directional Dynamics: Adverse Yaw & Sideslip Excursion', 
                         fontsize=16, fontweight='bold')
        
        # Create subplots
        self.ax_aircraft = plt.subplot2grid((3, 3), (0, 0), colspan=2, rowspan=2)
        self.ax_beta = plt.subplot2grid((3, 3), (0, 2))
        self.ax_roll = plt.subplot2grid((3, 3), (1, 2))
        self.ax_yaw = plt.subplot2grid((3, 3), (2, 0))
        self.ax_forces = plt.subplot2grid((3, 3), (2, 1))
        self.ax_params = plt.subplot2grid((3, 3), (2, 2))
        
        # Aircraft parameters (from paper)
        self.b = 64.13  # wingspan (ft)
        self.V_KTAS = 135  # airspeed (knots)
        self.V_fps = self.V_KTAS * 1.68781  # convert to ft/s
        
        # Stability derivatives (landing configuration, Λ=22°)
        self.C_n_beta = 0.002  # per degree
        self.C_l_beta = -0.004  # per degree
        self.C_np_bare = -0.05  # per radian (adverse!)
        self.C_lp_bare = -0.4  # per radian
        
        # Roll damper creates synthetic derivatives
        self.C_np_synthetic = -0.05  # additional adverse yaw
        self.C_np_aug = self.C_np_bare + self.C_np_synthetic  # total = -0.10
        
        # Control surface characteristics
        self.dCn_dail = 0.003  # proverse yaw from aileron
        self.dCl_dail = 0.03   # roll power
        
        # Initial conditions
        self.roll_rate = 0  # deg/s
        self.beta = 0  # sideslip angle (deg)
        self.phi = 0  # roll angle (deg)
        self.time = 0
        self.dt = 0.05
        
        # Control input
        self.aileron_input = 0  # lateral stick (-1 to +1)
        self.roll_damper_on = True
        
        # Data storage
        self.time_history = []
        self.beta_history = []
        self.roll_rate_history = []
        self.yaw_moment_history = []
        
        # Setup plots
        self.setup_aircraft_view()
        self.setup_time_plots()
        self.setup_force_diagram()
        self.setup_parameter_display()
        
        # Animation
        self.anim = None
        self.running = False
        
    def setup_aircraft_view(self):
        """Setup top-down view of F-14A"""
        self.ax_aircraft.set_xlim(-50, 50)
        self.ax_aircraft.set_ylim(-50, 50)
        self.ax_aircraft.set_aspect('equal')
        self.ax_aircraft.set_title('Top View: F-14A Aircraft', fontsize=12, fontweight='bold')
        self.ax_aircraft.grid(True, alpha=0.3)
        self.ax_aircraft.set_xlabel('Lateral Position (ft)')
        self.ax_aircraft.set_ylabel('Longitudinal Position (ft)')
        
        # Draw F-14A from top (simplified)
        self.draw_f14()
        
    def draw_f14(self):
        """Draw simplified F-14A top view"""
        # Clear previous
        self.ax_aircraft.clear()
        self.ax_aircraft.set_xlim(-50, 50)
        self.ax_aircraft.set_ylim(-50, 50)
        self.ax_aircraft.set_aspect('equal')
        self.ax_aircraft.grid(True, alpha=0.3)
        
        # Rotate aircraft by roll angle and sideslip
        phi_rad = np.radians(self.phi)
        beta_rad = np.radians(self.beta)
        
        # Fuselage (elongated)
        fuselage_length = 25
        fuselage_width = 3
        fuse = np.array([
            [-fuselage_width/2, fuselage_length/2],
            [fuselage_width/2, fuselage_length/2],
            [fuselage_width, fuselage_length/3],
            [fuselage_width, -fuselage_length/2],
            [-fuselage_width, -fuselage_length/2],
            [-fuselage_width, fuselage_length/3]
        ])
        
        # Wings (variable sweep shown at 22°)
        wing_span = 30
        wing_chord_root = 8
        wing_chord_tip = 4
        wing_sweep = 10  # simplified
        
        left_wing = np.array([
            [0, 0],
            [-wing_span/2, -wing_sweep],
            [-wing_span/2, -wing_sweep-wing_chord_tip],
            [-wing_chord_root/2, -wing_chord_root]
        ])
        
        right_wing = np.array([
            [0, 0],
            [wing_span/2, -wing_sweep],
            [wing_span/2, -wing_sweep-wing_chord_tip],
            [wing_chord_root/2, -wing_chord_root]
        ])
        
        # Twin vertical tails
        tail_offset = 4
        tail_height = 8
        tail_width = 2
        
        left_tail = np.array([
            [-tail_offset, -fuselage_length/2],
            [-tail_offset-tail_width, -fuselage_length/2-tail_height],
            [-tail_offset, -fuselage_length/2-tail_height]
        ])
        
        right_tail = np.array([
            [tail_offset, -fuselage_length/2],
            [tail_offset+tail_width, -fuselage_length/2-tail_height],
            [tail_offset, -fuselage_length/2-tail_height]
        ])
        
        # Rotate all components for roll angle
        def rotate_points(points, angle):
            cos_a, sin_a = np.cos(angle), np.sin(angle)
            rotation_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
            return points @ rotation_matrix.T
        
        fuse_rot = rotate_points(fuse, phi_rad)
        left_wing_rot = rotate_points(left_wing, phi_rad)
        right_wing_rot = rotate_points(right_wing, phi_rad)
        left_tail_rot = rotate_points(left_tail, phi_rad)
        right_tail_rot = rotate_points(right_tail, phi_rad)
        
        # Draw components
        fuselage_patch = Polygon(fuse_rot, closed=True, 
                                facecolor='gray', edgecolor='black', linewidth=2)
        left_wing_patch = Polygon(left_wing_rot, closed=True, 
                                 facecolor='lightblue', edgecolor='black', linewidth=2)
        right_wing_patch = Polygon(right_wing_rot, closed=True, 
                                  facecolor='lightblue', edgecolor='black', linewidth=2)
        left_tail_patch = Polygon(left_tail_rot, closed=True, 
                                 facecolor='darkgray', edgecolor='black', linewidth=1.5)
        right_tail_patch = Polygon(right_tail_rot, closed=True, 
                                  facecolor='darkgray', edgecolor='black', linewidth=1.5)
        
        self.ax_aircraft.add_patch(fuselage_patch)
        self.ax_aircraft.add_patch(left_wing_patch)
        self.ax_aircraft.add_patch(right_wing_patch)
        self.ax_aircraft.add_patch(left_tail_patch)
        self.ax_aircraft.add_patch(right_tail_patch)
        
        # Draw velocity vector (showing sideslip)
        V_magnitude = 25
        V_body_x = V_magnitude * np.sin(beta_rad)  # sideways component
        V_body_y = V_magnitude * np.cos(beta_rad)  # forward component
        
        # Rotate by roll angle
        V_x = V_body_x * np.cos(phi_rad) - V_body_y * np.sin(phi_rad)
        V_y = V_body_x * np.sin(phi_rad) + V_body_y * np.cos(phi_rad)
        
        self.ax_aircraft.arrow(0, 0, V_x, V_y, head_width=3, head_length=2, 
                              fc='red', ec='darkred', linewidth=2, label='Velocity Vector')
        
        # Draw body axis
        body_x = 20 * np.sin(phi_rad)
        body_y = 20 * np.cos(phi_rad)
        self.ax_aircraft.arrow(0, 0, body_x, body_y, head_width=2, head_length=1.5,
                              fc='green', ec='darkgreen', linewidth=1.5, 
                              linestyle='--', alpha=0.7, label='Body Axis')
        
        # Show sideslip angle
        if abs(self.beta) > 0.5:
            arc = Wedge((0, 0), 15, 90-self.beta, 90, 
                       facecolor='yellow', alpha=0.3, edgecolor='orange', linewidth=2)
            self.ax_aircraft.add_patch(arc)
            self.ax_aircraft.text(8, 12, f'β = {self.beta:.1f}°', 
                                 fontsize=11, fontweight='bold', color='orange')
        
        # Show roll angle
        if abs(self.phi) > 1:
            self.ax_aircraft.text(-40, 40, f'φ = {self.phi:.1f}°', 
                                 fontsize=11, fontweight='bold', color='blue')
        
        # Add legend
        self.ax_aircraft.legend(loc='upper right', fontsize=9)
        self.ax_aircraft.set_title(f'Top View: F-14A | Time = {self.time:.2f}s', 
                                   fontsize=12, fontweight='bold')
        
    def setup_time_plots(self):
        """Setup time history plots"""
        # Sideslip angle plot
        self.ax_beta.set_xlim(0, 10)
        self.ax_beta.set_ylim(-20, 5)
        self.ax_beta.set_title('Sideslip Angle β(t)', fontsize=11, fontweight='bold')
        self.ax_beta.set_xlabel('Time (s)')
        self.ax_beta.set_ylabel('β (degrees)')
        self.ax_beta.grid(True, alpha=0.3)
        self.ax_beta.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
        self.ax_beta.axhline(y=-10, color='r', linestyle='--', linewidth=1, 
                            label='Safe Limit', alpha=0.7)
        self.ax_beta.axhline(y=-15, color='r', linestyle='--', linewidth=2, 
                            label='Danger Zone', alpha=0.7)
        self.ax_beta.legend(fontsize=8)
        
        # Roll rate plot
        self.ax_roll.set_xlim(0, 10)
        self.ax_roll.set_ylim(-5, 50)
        self.ax_roll.set_title('Roll Rate p(t)', fontsize=11, fontweight='bold')
        self.ax_roll.set_xlabel('Time (s)')
        self.ax_roll.set_ylabel('p (deg/s)')
        self.ax_roll.grid(True, alpha=0.3)
        self.ax_roll.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
        self.ax_roll.axhline(y=30, color='orange', linestyle='--', linewidth=1, 
                            label='Example Rate', alpha=0.7)
        self.ax_roll.legend(fontsize=8)
        
    def setup_force_diagram(self):
        """Setup yaw moment balance diagram"""
        self.ax_yaw.set_xlim(0, 10)
        self.ax_yaw.set_ylim(-0.008, 0.004)
        self.ax_yaw.set_title('Yaw Moment Components', fontsize=11, fontweight='bold')
        self.ax_yaw.set_xlabel('Time (s)')
        self.ax_yaw.set_ylabel('ΔCn (dimensionless)')
        self.ax_yaw.grid(True, alpha=0.3)
        self.ax_yaw.axhline(y=0, color='k', linestyle='-', linewidth=1)
        
    def setup_force_diagram(self):
        """Setup yaw moment components visualization"""
        self.ax_forces.clear()
        self.ax_forces.set_xlim(-1, 1)
        self.ax_forces.set_ylim(-1, 1)
        self.ax_forces.set_aspect('equal')
        self.ax_forces.axis('off')
        self.ax_forces.set_title('Yaw Moment Balance', fontsize=11, fontweight='bold')
        
    def update_force_diagram(self):
        """Update force balance visualization"""
        self.ax_forces.clear()
        self.ax_forces.set_xlim(-1.2, 1.2)
        self.ax_forces.set_ylim(-1.2, 1.2)
        self.ax_forces.set_aspect('equal')
        self.ax_forces.axis('off')
        
        # Calculate moment components
        p_rad = np.radians(self.roll_rate)
        pb_2V = (p_rad * self.b) / (2 * self.V_fps)
        
        # Dynamic adverse yaw from roll rate
        Cn_dynamic = self.C_np_aug * pb_2V if self.roll_damper_on else self.C_np_bare * pb_2V
        
        # Static proverse yaw from aileron
        Cn_aileron = self.dCn_dail * abs(self.aileron_input) * 7  # 7 deg max deflection
        
        # Restoring moment from directional stability
        Cn_beta = self.C_n_beta * self.beta
        
        # Net yaw moment
        Cn_net = Cn_dynamic + Cn_aileron + Cn_beta
        
        # Draw force arrows
        arrow_scale = 300
        
        # Dynamic adverse (downward = adverse to right roll)
        if Cn_dynamic != 0:
            arrow_length = abs(Cn_dynamic) * arrow_scale
            color = 'red' if Cn_dynamic < 0 else 'green'
            self.ax_forces.arrow(0, 0.5, -arrow_length if Cn_dynamic < 0 else arrow_length, 0,
                               head_width=0.1, head_length=0.05, fc=color, ec=color, linewidth=2)
            self.ax_forces.text(0, 0.65, f'Dynamic Adverse\nCnₚ×(pb/2V)\n{Cn_dynamic:.5f}',
                              ha='center', fontsize=8, fontweight='bold', color=color)
        
        # Static proverse (upward = proverse helps right roll)
        if Cn_aileron != 0:
            arrow_length = Cn_aileron * arrow_scale
            self.ax_forces.arrow(0, 0, arrow_length, 0,
                               head_width=0.1, head_length=0.05, fc='blue', ec='blue', linewidth=2)
            self.ax_forces.text(0, 0.15, f'Static Proverse\ndCn/dail×δ\n{Cn_aileron:.5f}',
                              ha='center', fontsize=8, fontweight='bold', color='blue')
        
        # Restoring moment from β
        if abs(self.beta) > 0.1:
            arrow_length = abs(Cn_beta) * arrow_scale
            color = 'purple'
            direction = 1 if Cn_beta > 0 else -1
            self.ax_forces.arrow(0, -0.5, direction * arrow_length, 0,
                               head_width=0.1, head_length=0.05, fc=color, ec=color, linewidth=2)
            self.ax_forces.text(0, -0.65, f'Restoring (Cnβ×β)\n{Cn_beta:.5f}',
                              ha='center', fontsize=8, fontweight='bold', color=color)
        
        # Net moment
        net_arrow_length = abs(Cn_net) * arrow_scale * 1.5
        net_color = 'darkred' if Cn_net < 0 else 'darkgreen'
        if abs(Cn_net) > 0.0001:
            self.ax_forces.arrow(0, -0.9, -net_arrow_length if Cn_net < 0 else net_arrow_length, 0,
                               head_width=0.15, head_length=0.08, fc=net_color, ec='black', 
                               linewidth=3, alpha=0.7)
            self.ax_forces.text(0, -1.05, f'NET: {Cn_net:.5f}',
                              ha='center', fontsize=9, fontweight='bold', color=net_color)
        
        self.ax_forces.set_title('Yaw Moment Balance', fontsize=11, fontweight='bold')
        
    def setup_parameter_display(self):
        """Setup parameter display"""
        self.ax_params.axis('off')
        
    def update_parameter_display(self):
        """Update parameter text display"""
        self.ax_params.clear()
        self.ax_params.axis('off')
        
        info_text = f"""AIRCRAFT PARAMETERS:
        
V = {self.V_KTAS} KEAS
Wingspan b = {self.b:.1f} ft

STABILITY DERIVATIVES:
Cnβ = {self.C_n_beta:.4f} /deg
Clβ = {self.C_l_beta:.4f} /deg
Cnₚ (bare) = {self.C_np_bare:.3f} /rad
Cnₚ (aug) = {self.C_np_aug:.3f} /rad

CURRENT STATE:
φ = {self.phi:.1f}°
p = {self.roll_rate:.1f} °/s
β = {self.beta:.1f}°

Roll Damper: {'ON' if self.roll_damper_on else 'OFF'}
Aileron: {self.aileron_input*100:.0f}%
"""
        
        self.ax_params.text(0.05, 0.95, info_text, transform=self.ax_params.transAxes,
                          fontsize=9, verticalalignment='top', family='monospace',
                          bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
    def equations_of_motion(self, dt):
        """Compute lateral-directional dynamics"""
        # Convert to radians
        p_rad = np.radians(self.roll_rate)
        beta_rad = np.radians(self.beta)
        
        # Non-dimensional roll rate
        pb_2V = (p_rad * self.b) / (2 * self.V_fps)
        
        # Roll acceleration from aileron input
        aileron_deflection = self.aileron_input * 7  # max 7 degrees
        roll_moment = self.dCl_dail * aileron_deflection
        
        # Roll damping (includes synthetic from roll damper if on)
        C_lp_total = self.C_lp_bare
        if self.roll_damper_on:
            C_lp_total += -0.5  # synthetic roll damping
        
        roll_damping = C_lp_total * pb_2V
        
        # Net roll acceleration (simplified, ignoring inertia details)
        p_dot_rad = (roll_moment + roll_damping) * 50  # scaling factor for visualization
        p_dot = np.degrees(p_dot_rad)
        
        # Update roll rate
        self.roll_rate += p_dot * dt
        self.roll_rate = np.clip(self.roll_rate, -100, 100)  # limit
        
        # Update roll angle
        self.phi += self.roll_rate * dt
        
        # Yaw moments creating sideslip
        # 1. Dynamic adverse yaw from roll rate
        C_np_effective = self.C_np_aug if self.roll_damper_on else self.C_np_bare
        Cn_dynamic = C_np_effective * pb_2V
        
        # 2. Static proverse yaw from aileron
        Cn_aileron = self.dCn_dail * aileron_deflection
        
        # 3. Restoring moment from directional stability
        Cn_beta = self.C_n_beta * self.beta
        
        # Net yaw moment
        Cn_net = Cn_dynamic + Cn_aileron + Cn_beta
        
        # Sideslip rate (simplified)
        # When Cn_net < 0, aircraft yaws adversely, creating negative β
        beta_dot = -Cn_net * 500  # scaling factor
        
        # Update sideslip
        self.beta += beta_dot * dt
        self.beta = np.clip(self.beta, -20, 10)
        
        # Store data
        self.time_history.append(self.time)
        self.beta_history.append(self.beta)
        self.roll_rate_history.append(self.roll_rate)
        self.yaw_moment_history.append(Cn_net)
        
        # Update time
        self.time += dt
        
    def update_plots(self):
        """Update all time history plots"""
        if len(self.time_history) > 1:
            # Beta plot
            self.ax_beta.clear()
            self.ax_beta.set_xlim(max(0, self.time-10), self.time+1)
            self.ax_beta.set_ylim(-20, 5)
            self.ax_beta.plot(self.time_history, self.beta_history, 'b-', linewidth=2)
            self.ax_beta.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
            self.ax_beta.axhline(y=-10, color='orange', linestyle='--', linewidth=1, 
                                label='Caution', alpha=0.7)
            self.ax_beta.axhline(y=-15, color='r', linestyle='--', linewidth=2, 
                                label='Danger', alpha=0.7)
            self.ax_beta.set_title('Sideslip Angle β(t)', fontsize=11, fontweight='bold')
            self.ax_beta.set_xlabel('Time (s)')
            self.ax_beta.set_ylabel('β (degrees)')
            self.ax_beta.grid(True, alpha=0.3)
            self.ax_beta.legend(fontsize=8)
            
            # Roll rate plot
            self.ax_roll.clear()
            self.ax_roll.set_xlim(max(0, self.time-10), self.time+1)
            self.ax_roll.set_ylim(-5, 50)
            self.ax_roll.plot(self.time_history, self.roll_rate_history, 'g-', linewidth=2)
            self.ax_roll.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
            self.ax_roll.axhline(y=30, color='orange', linestyle='--', linewidth=1, 
                                label='30°/s Example', alpha=0.7)
            self.ax_roll.set_title('Roll Rate p(t)', fontsize=11, fontweight='bold')
            self.ax_roll.set_xlabel('Time (s)')
            self.ax_roll.set_ylabel('p (deg/s)')
            self.ax_roll.grid(True, alpha=0.3)
            self.ax_roll.legend(fontsize=8)
            
            # Yaw moment plot
            self.ax_yaw.clear()
            self.ax_yaw.set_xlim(max(0, self.time-10), self.time+1)
            self.ax_yaw.set_ylim(-0.008, 0.004)
            self.ax_yaw.plot(self.time_history, self.yaw_moment_history, 'r-', linewidth=2)
            self.ax_yaw.axhline(y=0, color='k', linestyle='-', linewidth=1)
            self.ax_yaw.fill_between(self.time_history, 0, self.yaw_moment_history, 
                                    where=np.array(self.yaw_moment_history) < 0,
                                    alpha=0.3, color='red', label='Adverse')
            self.ax_yaw.fill_between(self.time_history, 0, self.yaw_moment_history,
                                    where=np.array(self.yaw_moment_history) > 0,
                                    alpha=0.3, color='green', label='Proverse')
            self.ax_yaw.set_title('Net Yaw Moment ΔCn', fontsize=11, fontweight='bold')
            self.ax_yaw.set_xlabel('Time (s)')
            self.ax_yaw.set_ylabel('ΔCn')
            self.ax_yaw.grid(True, alpha=0.3)
            self.ax_yaw.legend(fontsize=8)
            
    def animate(self, frame):
        """Animation update function"""
        if self.running:
            # Update dynamics
            self.equations_of_motion(self.dt)
            
            # Update all visualizations
            self.draw_f14()
            self.update_plots()
            self.update_force_diagram()
            self.update_parameter_display()
            
        return []
    
    def start_simulation(self):
        """Start the simulation"""
        self.running = True
        
    def stop_simulation(self):
        """Stop the simulation"""
        self.running = False
        
    def reset_simulation(self):
        """Reset simulation to initial conditions"""
        self.running = False
        self.roll_rate = 0
        self.beta = 0
        self.phi = 0
        self.time = 0
        self.aileron_input = 0
        self.time_history = []
        self.beta_history = []
        self.roll_rate_history = []
        self.yaw_moment_history = []
        
        # Redraw everything
        self.draw_f14()
        self.setup_time_plots()
        self.update_force_diagram()
        self.update_parameter_display()
        plt.draw()
        
    def run(self):
        """Run the interactive simulator"""
        # Add control sliders
        plt.subplots_adjust(bottom=0.25)
        
        # Aileron input slider
        ax_aileron = plt.axes([0.15, 0.15, 0.3, 0.03])
        slider_aileron = Slider(ax_aileron, 'Lateral Stick', -1.0, 1.0, 
                               valinit=0, valstep=0.1, color='blue')
        
        def update_aileron(val):
            self.aileron_input = slider_aileron.val
        slider_aileron.on_changed(update_aileron)
        
        # Roll damper toggle
        ax_damper = plt.axes([0.55, 0.15, 0.15, 0.04])
        button_damper = Button(ax_damper, 'Toggle Roll Damper', color='lightgray')
        
        def toggle_damper(event):
            self.roll_damper_on = not self.roll_damper_on
            button_damper.label.set_text(f"Roll Damper: {'ON' if self.roll_damper_on else 'OFF'}")
            button_damper.color = 'lightgreen' if self.roll_damper_on else 'lightcoral'
            plt.draw()
        button_damper.on_clicked(toggle_damper)
        
        # Start/Stop button
        ax_start = plt.axes([0.15, 0.08, 0.1, 0.04])
        button_start = Button(ax_start, 'Start', color='lightgreen')
        
        def start_stop(event):
            if self.running:
                self.stop_simulation()
                button_start.label.set_text('Start')
                button_start.color = 'lightgreen'
            else:
                self.start_simulation()
                button_start.label.set_text('Stop')
                button_start.color = 'lightcoral'
            plt.draw()
        button_start.on_clicked(start_stop)
        
        # Reset button
        ax_reset = plt.axes([0.27, 0.08, 0.1, 0.04])
        button_reset = Button(ax_reset, 'Reset', color='lightyellow')
        button_reset.on_clicked(lambda event: self.reset_simulation())
        
        # Velocity slider
        ax_velocity = plt.axes([0.55, 0.08, 0.3, 0.03])
        slider_velocity = Slider(ax_velocity, 'Airspeed (KEAS)', 100, 250, 
                                valinit=135, valstep=5, color='green')
        
        def update_velocity(val):
            self.V_KTAS = slider_velocity.val
            self.V_fps = self.V_KTAS * 1.68781
        slider_velocity.on_changed(update_velocity)
        
        # Start animation
        self.anim = FuncAnimation(self.fig, self.animate, interval=50, blit=False)
        
        plt.show()

# Create and run simulator
if __name__ == "__main__":
    print("="*70)
    print("F-14A LATERAL-DIRECTIONAL STABILITY SIMULATOR")
    print("="*70)
    print("\nThis simulator demonstrates the adverse yaw problem discussed in")